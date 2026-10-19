"""DeepAgent responder: history conversion (always) and the live Azure agent (opt-in).

Live test: AGENT_TESTS=1 pytest tests/test_agent.py -s
(needs the compute's managed identity to reach Azure OpenAI, and internet for DuckDuckGo).
"""

import os
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.responder import items_to_messages
from tests.test_realtime_protocol import Conn, assert_valid, user_text


def msg(role, part_type, text):
    key = "text" if part_type in ("input_text", "output_text") else "transcript"
    return {"type": "message", "role": role, "content": [{"type": part_type, key: text}]}


def test_items_become_agent_messages():
    items = [
        msg("user", "input_text", "Start the conversation."),
        msg("assistant", "output_audio", "Hi there!"),
        msg("user", "input_audio", "What's the weather in Paris?"),
        {"type": "mcp_call", "id": "item_1", "name": "web_search",
         "arguments": '{"query": "Paris weather"}', "output": "Sunny, 21C"},
        msg("assistant", "output_audio", "Let me look that up. It's sunny and 21 degrees."),
        # Skipped: an empty (unintelligible) user turn, a cancelled tool call, a
        # reply truncated to nothing by barge-in.
        msg("user", "input_audio", ""),
        {"type": "mcp_call", "id": "item_2", "name": "web_search", "arguments": "{}",
         "output": None},
        msg("assistant", "output_audio", ""),
    ]
    messages = items_to_messages(items)
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage, AIMessage,
                                           ToolMessage, AIMessage]
    call = messages[3].tool_calls[0]
    assert (call["id"], call["name"], call["args"]) == ("item_1", "web_search", {"query": "Paris weather"})
    assert messages[4].tool_call_id == "item_1" and messages[4].content == "Sunny, 21C"


@pytest.mark.skipif(os.environ.get("AGENT_TESTS") != "1",
                    reason="set AGENT_TESTS=1 to call Azure OpenAI and DuckDuckGo")
def test_live_agent_searches_and_answers():
    from app.agent.responder import DeepAgentResponder
    from app.server import app

    with TestClient(app) as client:
        client.app.state.pipeline.responder = DeepAgentResponder()
        with client.websocket_connect("/v1/realtime") as ws:
            conn = Conn(ws)
            conn.until("session.created")
            for question, expect_tool in (("Hi! How are you?", False),
                                          ("What is the latest news about NASA today?", True)):
                started = time.monotonic()
                user_text(conn, question)
                conn.send({"type": "response.create"})
                events = conn.until("response.done")
                response = events[-1]["response"]
                tools = [o for o in response["output"] if o["type"] == "mcp_call"]
                reply = response["output"][0]["content"][0]["transcript"]
                print(f"\n{question!r} -> {time.monotonic() - started:.1f}s, "
                      f"tools={[t['arguments'] for t in tools]}\n  {reply}")
                assert response["status"] == "completed" and reply
                assert bool(tools) == expect_tool
                if expect_tool:
                    assert {t["name"] for t in tools} & {"web_search", "fetch_page", "task"}
                    assert all(t["output"] for t in tools)
            assert_valid(conn.events)
