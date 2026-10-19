"""DeepAgent responder: a LangChain Deep Agent on Azure OpenAI.

Tools: ``web_search`` (DuckDuckGo) and ``fetch_page`` for quick lookups,
``write_todos`` for planning multi-step requests, and a ``researcher``
subagent (via the ``task`` tool) for in-depth research. A harness profile
hides the virtual-filesystem tools and the general-purpose subagent, which a
voice conversation doesn't need.

Replies stream token by token (the session speaks them sentence by sentence);
tool calls are reported as ``ToolCall`` events. Memory is the realtime
conversation itself: each turn replays the session's items (what the user
actually heard after barge-in, plus earlier tool calls/results).
"""

import json
import logging
from datetime import date
from typing import Any, AsyncIterator

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage

from app import config
from app.agent.llm import build_llm
from app.agent.tools import fetch_page, web_search
from app.pipeline.base import ToolCall, TurnContext

logger = logging.getLogger("deepagent.agent")

VOICE_PROMPT = """You are a helpful voice assistant. Everything you write is spoken aloud by a text-to-speech engine.
- Answer in one to three short, natural, conversational sentences unless the user asks for detail.
- Write plain spoken sentences only: no markdown, lists, headings, emojis, code, or URLs.
- For current events, news, weather, prices, sports, or facts you are unsure about, use web_search; if the snippets don't contain the actual answer, read the most promising result with fetch_page. Don't search for small talk or things you already know.
- If answering needs more than two searches or pages (comparisons, multi-part questions, in-depth topics), delegate to the researcher subagent with the task tool instead of searching yourself, then summarize its findings briefly.
- When you use web results, give the answer directly; mention the source only if it matters.
- Today's date is {today}."""

PLANNING_PROMPT = (
    "You have a write_todos tool for planning. Use it only for genuinely multi-step requests "
    "(three or more distinct steps), never for simple questions, and keep it updated as you work."
)

RESEARCHER = {
    "name": "researcher",
    "description": (
        "Researches a question that needs several web searches or reading multiple pages "
        "(comparisons, multi-part questions, in-depth topics) and returns a concise factual "
        "summary. Not for simple lookups."
    ),
    "system_prompt": (
        "You are an efficient web researcher working for a voice assistant, so speed matters. "
        "Use web_search to find sources and fetch_page to read the most relevant ones, running "
        "independent searches in parallel. Use at most four tool calls in total, then stop. "
        "Reply with a concise plain-text summary (under 120 words) of the facts that answer the "
        "question, including key numbers and dates."
    ),
}

FILESYSTEM_TOOLS = frozenset({"ls", "read_file", "write_file", "edit_file", "delete", "glob",
                              "grep", "execute"})
VOICE_PROFILE = HarnessProfile(
    excluded_tools=FILESYSTEM_TOOLS,
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
)

# Spoken right away when the agent starts a tool call without saying anything itself.
ACKS = {
    "web_search": "Let me look that up.",
    "fetch_page": "Let me check that page.",
    "task": "Let me research that, it may take a moment.",
    "write_todos": "Let me plan that out.",
}
DEFAULT_ACK = "One moment."
MAX_TOOL_OUTPUT_CHARS = 4000


def _profile_key(llm) -> str:
    # deepagents resolves profiles by the model's provider (and identifier, if any).
    from deepagents._models import get_model_identifier, get_model_provider

    provider, identifier = get_model_provider(llm), get_model_identifier(llm)
    return f"{provider}:{identifier}" if identifier else provider


def _text(message: BaseMessage) -> str:
    return str(message.text or "")


def items_to_messages(items: list[dict]) -> list[BaseMessage]:
    """Realtime conversation items -> LangChain messages for the agent."""
    messages: list[BaseMessage] = []
    for item in items:
        kind = item.get("type")
        if kind == "message":
            text = " ".join(
                (part.get("text") or part.get("transcript") or "").strip()
                for part in item.get("content") or []
            ).strip()
            if not text:
                continue
            if item.get("role") == "user":
                messages.append(HumanMessage(text))
            elif item.get("role") == "assistant":
                messages.append(AIMessage(text))
        elif kind == "mcp_call" and item.get("output") is not None:
            try:
                args = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            messages.append(AIMessage("", tool_calls=[
                {"id": item["id"], "name": item.get("name") or "tool", "args": args}]))
            messages.append(ToolMessage(item["output"], tool_call_id=item["id"]))
    return messages


class DeepAgentResponder:
    def __init__(self) -> None:
        self._llm = build_llm()
        register_harness_profile(_profile_key(self._llm), VOICE_PROFILE)
        self._tools = [web_search, fetch_page]
        self._agent_key = ""
        self._agent: Any = None

    def _agent_for(self, instructions: str):
        today = date.today().isoformat()
        key = f"{today}\n{instructions}"
        if key != self._agent_key:
            prompt = VOICE_PROMPT.format(today=today)
            if instructions:
                prompt += f"\n\n{instructions}"
            self._agent = create_deep_agent(
                model=self._llm, tools=self._tools, system_prompt=prompt,
                subagents=[{**RESEARCHER, "tools": self._tools}],
                # deepagents 0.7 no longer adds planning by default.
                middleware=[TodoListMiddleware(system_prompt=PLANNING_PROMPT)],
            )
            self._agent_key = key
        return self._agent

    async def respond(self, turn: TurnContext) -> AsyncIterator[str | ToolCall]:
        messages = items_to_messages(turn.items)
        if not messages:
            return
        agent = self._agent_for(turn.instructions)
        spoke = False          # anything said this turn (suppresses the canned ack)
        streamed_step = False  # text streamed for the current model step
        calls: dict[str, ToolCall] = {}
        stream = agent.astream({"messages": messages}, stream_mode=["messages", "updates"],
                               config={"recursion_limit": config.AGENT_RECURSION_LIMIT})
        async for mode, payload in stream:
            if mode == "messages":
                chunk, meta = payload
                # Main agent's model tokens only (LangGraph already drops subagent
                # tokens without subgraphs=True; the node check is belt and braces).
                if meta.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk):
                    continue
                text = _text(chunk)
                if text:
                    spoke = streamed_step = True
                    yield text
                continue

            for node, data in payload.items():
                if node not in ("model", "tools") or not isinstance(data, dict):
                    continue
                out = data.get("messages") or []
                for message in out if isinstance(out, list) else [out]:
                    if isinstance(message, AIMessage):
                        text = _text(message).strip()
                        if text and not streamed_step:  # model didn't stream: speak it whole
                            spoke = True
                            yield text
                        streamed_step = False
                        yield " "  # keep sentences from separate steps apart
                        if message.tool_calls and not spoke:
                            spoke = True
                            yield ACKS.get(message.tool_calls[0]["name"], DEFAULT_ACK) + " "
                        for tc in message.tool_calls:
                            call = ToolCall(tc["id"], tc["name"], json.dumps(tc.get("args") or {}))
                            calls[call.call_id] = call
                            logger.info("Tool call %s(%s)", call.name, call.arguments[:200])
                            yield call
                    elif isinstance(message, ToolMessage):
                        started = calls.get(message.tool_call_id) or ToolCall(
                            message.tool_call_id, message.name or "tool", "{}")
                        output = _text(message).strip()[:MAX_TOOL_OUTPUT_CHARS]
                        yield ToolCall(started.call_id, started.name, started.arguments,
                                       output=output,
                                       error=output if message.status == "error" else None)
