"""Builders for OpenAI Realtime GA server events.

Only the subset the browser client (and the @openai/agents SDK inside it)
consumes. Field shapes must satisfy BOTH the official ``openai`` Python types
and the SDK's own zod schemas — an event the SDK fails to parse is treated as
"generic": the UI still sees it, but the SDK skips audio playback, response
sequencing and tool calls. tests/test_realtime_protocol.py checks both.
"""

import secrets

from app.pipeline.base import SAMPLE_RATE

PCM_FORMAT = {"type": "audio/pcm", "rate": SAMPLE_RATE}


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(10)}"


def event(type_: str, **fields) -> dict:
    return {"type": type_, "event_id": new_id("event"), **fields}


def error(message: str, *, code: str, client_event_id: str | None = None,
          kind: str = "invalid_request_error") -> dict:
    # error.event_id links the failure to the client event (the SDK uses it to
    # release a pending response.create).
    return event("error", error={
        "type": kind, "code": code, "message": message, "param": None,
        "event_id": client_event_id,
    })


def default_session(session_id: str, instructions: str) -> dict:
    return {
        "type": "realtime",
        "object": "realtime.session",
        "id": session_id,
        "model": "deepagent-voice",
        "output_modalities": ["audio"],
        "instructions": instructions,
        "audio": {
            "input": {
                "format": dict(PCM_FORMAT),
                "transcription": None,
                "noise_reduction": None,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600,
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"format": dict(PCM_FORMAT), "voice": "alloy", "speed": 1.0},
        },
        "tools": [],
        "tool_choice": "auto",
        "max_output_tokens": "inf",
    }


def response_obj(response_id: str, conversation_id: str, voice: str, *, status: str,
                 output: list | None = None, status_details: dict | None = None) -> dict:
    return {
        "object": "realtime.response",
        "id": response_id,
        "conversation_id": conversation_id,
        "status": status,
        "status_details": status_details,
        "output": output or [],
        "output_modalities": ["audio"],
        "max_output_tokens": "inf",
        "audio": {"output": {"format": dict(PCM_FORMAT), "voice": voice}},
        "usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        "metadata": None,
    }


def assistant_item(item_id: str, *, status: str, transcript: str | None) -> dict:
    content = [] if transcript is None else [{"type": "output_audio", "transcript": transcript}]
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def user_audio_item(item_id: str, transcript: str | None = None) -> dict:
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": [{"type": "input_audio", "transcript": transcript}],
    }


# `openai` types require usage on transcription.completed; the SDK only accepts
# the "tokens" variant, so report a zeroed one.
TRANSCRIPTION_USAGE = {
    "type": "tokens", "total_tokens": 0, "input_tokens": 0, "output_tokens": 0,
    "input_token_details": {"text_tokens": 0, "audio_tokens": 0},
}


def mcp_call_item(item_id: str, *, name: str, arguments: str) -> dict:
    """A server-executed tool call. `output` must always be present (string or
    null): the SDK's strict item parse throws if it is missing."""
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "mcp_call",
        "status": "in_progress",
        "server_label": "deepagent",
        "name": name,
        "arguments": arguments,
        "output": None,
        "error": None,
    }


def finish_mcp_call(item: dict, *, output: str | None, error: str | None) -> None:
    item["status"] = "incomplete" if error else "completed"
    item["output"] = output
    item["error"] = {"type": "tool_execution_error", "message": error} if error else None
