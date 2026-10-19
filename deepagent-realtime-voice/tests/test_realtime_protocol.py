"""Protocol tests for /v1/realtime, driven the way the browser SDK drives it.

Every event the server emits is validated against both the official ``openai``
Python realtime types and (when installed by scripts/setup_web.sh) the
@openai/agents SDK's own zod schemas, which the browser actually enforces.
"""

import asyncio
import base64
import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai.types.realtime import RealtimeServerEvent
from pydantic import TypeAdapter
from starlette.websockets import WebSocketDisconnect

from app.pipeline.base import ToolCall
from app.pipeline.mock import GREETING_REPLY, MockSTT
from app.server import app

SDK_DIST = (Path(os.environ.get("DRA_CACHE", "/mnt/cache/deepagent-rtv"))
            / "web-node/node_modules/@openai/agents-realtime/dist")
SCHEMA_CHECK = Path(__file__).with_name("sdk_schema_check.mjs")
SERVER_EVENT = TypeAdapter(RealtimeServerEvent)

# Shape the SDK sends right after connecting (openaiRealtimeBase._getMergedSessionConfig)
# with the options s2s-realtime-client.js passes.
SDK_SESSION_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "instructions": "Be brief.",
        "model": "s2s-local",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "noise_reduction": None,
                "transcription": {"model": "whisper-1", "language": "auto"},
                "turn_detection": {"type": "server_vad", "interrupt_response": True},
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "Aiden", "speed": 1},
        },
    },
}

RESPONSE_SEQUENCE = [
    "response.created",
    "response.output_item.added",
    "conversation.item.added",
    "response.content_part.added",
    *["response.output_audio_transcript.delta", "response.output_audio.delta"] * 3,
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.content_part.done",
    "response.output_item.done",
    "conversation.item.done",
    "response.done",
]


class SlowResponder:
    """Speaks one sentence, then stalls so the test can cancel mid-response."""

    async def respond(self, turn):
        yield "First sentence. "
        await asyncio.sleep(10)
        yield "Never spoken."


class ToolResponder:
    """Scripted agent: acknowledgement, one tool call, then the answer."""

    stall = False

    async def respond(self, turn):
        yield "Let me look that up. "
        yield ToolCall("call_1", "web_search", '{"query": "weather"}')
        if self.stall:
            await asyncio.sleep(10)
        yield ToolCall("call_1", "web_search", '{"query": "weather"}', output="Sunny, 21C")
        yield "It is sunny."


class Conn:
    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []

    def send(self, event: dict) -> None:
        self.ws.send_text(json.dumps(event))

    def recv(self) -> dict:
        event = self.ws.receive_json()
        self.events.append(event)
        return event

    def until(self, type_: str) -> list[dict]:
        out = []
        while True:
            event = self.recv()
            out.append(event)
            if event["type"] == type_:
                return out


def collapse(types: list[str]) -> list[str]:
    return [t for i, t in enumerate(types) if i == 0 or t != types[i - 1]]


def assert_valid(events: list[dict]) -> None:
    for event in events:
        SERVER_EVENT.validate_python(event)
    node = shutil.which("node")
    if not node or not SDK_DIST.is_dir():
        pytest.skip("SDK schemas not installed (run scripts/setup_web.sh)")
    result = subprocess.run([node, str(SCHEMA_CHECK), str(SDK_DIST)], input=json.dumps(events),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def open_session(client: TestClient, *, slow: bool = False):
    if slow:
        client.app.state.pipeline = dataclasses.replace(client.app.state.pipeline,
                                                        responder=SlowResponder())
    return client.websocket_connect("/v1/realtime")


def tone(seconds: float, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(24_000 * seconds)) / 24_000
    return (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    return bytes(2 * int(24_000 * seconds))


def stream_audio(conn: Conn, pcm: bytes, chunk_ms: int = 40) -> None:
    """Send audio the way the browser does: one append per ~40 ms frame."""
    step = 48 * chunk_ms
    for i in range(0, len(pcm), step):
        conn.send({"type": "input_audio_buffer.append",
                   "audio": base64.b64encode(pcm[i:i + step]).decode()})


def user_text(conn: Conn, text: str) -> None:
    conn.send({"type": "conversation.item.create",
               "item": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]}})
    added, done = conn.recv(), conn.recv()
    assert (added["type"], done["type"]) == ("conversation.item.added", "conversation.item.done")
    assert added["item"]["id"] == done["item"]["id"]


def test_greeting_flow(client):
    with open_session(client) as ws:
        conn = Conn(ws)
        assert conn.recv()["type"] == "session.created"

        conn.send(SDK_SESSION_UPDATE)
        session = conn.until("session.updated")[-1]["session"]
        assert session["instructions"] == "Be brief."
        assert session["audio"]["output"]["voice"] == "Aiden"
        # The SDK reads this to decide who cancels on barge-in.
        assert session["audio"]["input"]["turn_detection"]["interrupt_response"] is True

        user_text(conn, "Start the conversation now with a brief greeting.")
        conn.send({"type": "response.create", "event_id": "agents_js_response_create_1"})
        events = conn.until("response.done")
        assert collapse([e["type"] for e in events]) == RESPONSE_SEQUENCE

        response = events[-1]["response"]
        assert response["status"] == "completed"
        item_id = events[1]["item"]["id"]
        deltas = [e for e in events if e["type"] == "response.output_audio.delta"]
        assert all(e["item_id"] == item_id and e["response_id"] == response["id"] for e in deltas)
        pcm = [base64.b64decode(e["delta"]) for e in deltas]
        assert sum(map(len, pcm)) > 24_000 and all(len(p) % 2 == 0 for p in pcm)

        transcript = "".join(e["delta"] for e in events
                             if e["type"] == "response.output_audio_transcript.delta")
        assert transcript == GREETING_REPLY
        assert response["output"][0]["content"][0]["transcript"] == GREETING_REPLY

        user_text(conn, "hello there")
        conn.send({"type": "response.create"})
        done = conn.until("response.done")[-1]
        assert done["response"]["output"][0]["content"][0]["transcript"] == "You said: hello there"

        assert_valid(conn.events)


def test_cancel_mid_response_frees_the_slot(client):
    with open_session(client, slow=True) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "response.create"})
        conn.until("response.output_audio.delta")
        conn.send({"type": "response.cancel"})
        done = conn.until("response.done")[-1]["response"]
        assert done["status"] == "cancelled"
        assert done["status_details"]["reason"] == "client_cancelled"
        assert done["output"][0]["status"] == "incomplete"
        assert done["output"][0]["content"][0]["transcript"] == "First sentence."

        conn.send({"type": "response.create"})
        assert conn.until("response.created")[-1]["response"]["status"] == "in_progress"
        conn.send({"type": "response.cancel"})
        conn.until("response.done")
        assert_valid(conn.events)


def test_overlapping_response_create_is_rejected_with_event_id(client):
    with open_session(client, slow=True) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "response.create", "event_id": "a"})
        conn.until("response.created")
        conn.send({"type": "response.create", "event_id": "b"})
        error = conn.until("error")[-1]["error"]
        assert error["code"] == "conversation_already_has_active_response"
        assert error["event_id"] == "b"
        conn.send({"type": "response.cancel"})
        conn.until("response.done")
        assert_valid(conn.events)


def test_truncate_keeps_only_heard_transcript(client):
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "response.create"})
        item_id = conn.until("response.done")[-1]["response"]["output"][0]["id"]

        conn.send({"type": "conversation.item.truncate", "item_id": item_id,
                   "content_index": 0, "audio_end_ms": 1})
        truncated = conn.until("conversation.item.truncated")[-1]
        assert (truncated["item_id"], truncated["audio_end_ms"]) == (item_id, 1)

        conn.send({"type": "conversation.item.retrieve", "item_id": item_id})
        item = conn.until("conversation.item.retrieved")[-1]["item"]
        assert item["content"][0]["transcript"] == "Hi!"
        assert_valid(conn.events)


def test_errors_for_unknown_events_and_idle_cancel(client):
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "bogus.event", "event_id": "x"})
        assert conn.until("error")[-1]["error"]["code"] == "unknown_event"
        conn.send({"type": "response.cancel"})
        assert conn.until("error")[-1]["error"]["code"] == "response_cancel_not_active"
        conn.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\0\0").decode()})
        conn.send({"type": "input_audio_buffer.clear"})
        conn.until("input_audio_buffer.cleared")
        assert_valid(conn.events)


def test_newest_connection_replaces_the_active_one(client):
    with open_session(client) as first:
        Conn(first).until("session.created")
        with open_session(client) as second:
            Conn(second).until("session.created")
            with pytest.raises(WebSocketDisconnect) as closed:
                first.receive_json()
            assert closed.value.code == 4000


USER_TURN = [
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.added",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.done",
]


def test_voice_turn_is_detected_transcribed_and_answered(client):
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send(SDK_SESSION_UPDATE)
        conn.until("session.updated")

        stream_audio(conn, silence(0.5) + tone(1.0) + silence(1.0))
        events = conn.until("response.done")
        types = [e["type"] for e in events]
        assert types[:len(USER_TURN)] == USER_TURN
        assert types[len(USER_TURN)] == "response.created"

        started, stopped = events[0], events[1]
        assert started["item_id"] == stopped["item_id"]
        # Tone starts at 500 ms; the 300 ms prefix pad moves the turn start earlier.
        assert started["audio_start_ms"] == 200
        assert 1480 <= stopped["audio_end_ms"] <= 1520

        user_item = events[3]["item"]
        assert (user_item["role"], user_item["content"][0]["type"]) == ("user", "input_audio")
        assert events[4]["transcript"] == MockSTT.text
        assert events[5]["item"]["content"][0]["transcript"] == MockSTT.text
        reply = events[-1]["response"]["output"][0]["content"][0]["transcript"]
        assert reply == f"You said: {MockSTT.text}"
        assert_valid(conn.events)


def test_speaking_over_the_assistant_cancels_its_reply(client):
    with open_session(client, slow=True) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send(SDK_SESSION_UPDATE)
        conn.until("session.updated")
        conn.send({"type": "response.create"})
        conn.until("response.output_audio.delta")

        stream_audio(conn, tone(0.5))
        events = conn.until("response.done")
        types = [e["type"] for e in events]
        assert types.index("input_audio_buffer.speech_started") < types.index("response.done")
        assert events[-1]["response"]["status"] == "cancelled"
        assert events[-1]["response"]["status_details"]["reason"] == "turn_detected"
        assert_valid(conn.events)


def test_short_noise_does_not_start_a_turn(client):
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        stream_audio(conn, tone(0.12) + silence(1.0))
        conn.send({"type": "input_audio_buffer.clear"})
        assert [e["type"] for e in conn.until("input_audio_buffer.cleared")] == [
            "input_audio_buffer.cleared"]


def test_empty_transcript_gets_no_reply(client, monkeypatch):
    monkeypatch.setattr(MockSTT, "text", "")
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        stream_audio(conn, tone(0.6) + silence(1.0))
        conn.until("conversation.item.done")
        conn.send({"type": "bogus.event"})
        assert conn.recv()["type"] == "error"  # no response.created was queued before it
        assert_valid(conn.events)


def use_responder(client, responder) -> None:
    client.app.state.pipeline = dataclasses.replace(client.app.state.pipeline, responder=responder)


def test_agent_tool_calls_become_mcp_call_items(client):
    use_responder(client, ToolResponder())
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        user_text(conn, "What's the weather?")
        user_id = conn.events[-1]["item"]["id"]
        conn.send({"type": "response.create"})
        events = conn.until("response.done")

        tool_events = [e for e in events if e.get("item", {}).get("type") == "mcp_call"]
        assert [e["type"] for e in tool_events] == [
            "response.output_item.added", "conversation.item.added",
            "response.output_item.done", "conversation.item.done"]
        added, done = tool_events[0], tool_events[2]
        assert added["output_index"] == done["output_index"] == 1
        assert added["item"]["output"] is None and added["item"]["status"] == "in_progress"
        assert done["item"]["output"] == "Sunny, 21C" and done["item"]["status"] == "completed"
        # The tool item sits between the user's question and the reply that used it.
        assert tool_events[1]["previous_item_id"] == user_id

        # The acknowledgement is spoken before the tool runs.
        types = [e["type"] for e in events]
        first_audio = types.index("response.output_audio.delta")
        assert first_audio < types.index("response.output_item.done")

        response = events[-1]["response"]
        assert [o["type"] for o in response["output"]] == ["message", "mcp_call"]
        assert response["output"][0]["content"][0]["transcript"] == (
            "Let me look that up. It is sunny.")
        assert_valid(conn.events)


def test_cancel_during_a_tool_call_closes_the_tool_item(client):
    responder = ToolResponder()
    responder.stall = True
    use_responder(client, responder)
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "response.create"})
        conn.until("response.output_item.added")  # message
        conn.until("response.output_item.added")  # mcp_call
        conn.send({"type": "response.cancel"})
        events = conn.until("response.done")
        tool_done = next(e for e in events if e["type"] == "response.output_item.done"
                         and e["item"]["type"] == "mcp_call")
        assert tool_done["item"]["status"] == "incomplete"
        assert tool_done["item"]["output"] is None
        assert tool_done["item"]["error"]["message"] == "Tool call cancelled."
        assert events[-1]["response"]["status"] == "cancelled"
        assert_valid(conn.events)


class FailingResponder:
    async def respond(self, turn):
        yield "Partial thought "
        raise RuntimeError("Azure is down")


def test_failed_reply_apologises_and_reports_failure(client):
    use_responder(client, FailingResponder())
    with open_session(client) as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "response.create"})
        response = conn.until("response.done")[-1]["response"]
        assert response["status"] == "failed"
        assert response["status_details"]["error"]["message"] == "Azure is down"
        transcript = response["output"][0]["content"][0]["transcript"]
        assert transcript.endswith("Sorry, I ran into a problem answering that.")
        assert_valid(conn.events)
