"""Persistent conversation memory: what is stored, TTL, and restore across reconnects."""

import json

import pytest
from fastapi.testclient import TestClient

from app.realtime.memory import CLIENT_COOKIE, MAX_TOOL_OUTPUT_CHARS, ConversationStore
from app.server import app
from tests.test_realtime_protocol import Conn, user_text

GREETING = "Start the conversation now."


def store(tmp_path, **kw) -> ConversationStore:
    opts = {"ttl_s": 3600, "max_items": 60, "startup_prompt": GREETING, **kw}
    return ConversationStore(str(tmp_path / "memory.sqlite3"), **opts)


def message(role, part_type, text, **extra):
    key = "text" if part_type == "input_text" else "transcript"
    return {"id": f"item_{text[:4]}", "type": "message", "role": role,
            "content": [{"type": part_type, key: text, **extra}]}


def test_only_finished_text_and_tool_results_are_kept(tmp_path):
    s = store(tmp_path)
    s.save("browser1", [
        message("user", "input_text", GREETING),              # hidden startup prompt
        message("assistant", "output_audio", "Hi there!"),    # ...and its greeting
        message("user", "input_audio", "Weather in Paris?", audio="UklGRg=="),
        {"id": "item_tool", "type": "mcp_call", "name": "web_search", "arguments": "{}",
         "output": "x" * 5000},
        {"id": "item_cancelled", "type": "mcp_call", "name": "web_search", "arguments": "{}",
         "output": None},
        message("assistant", "output_audio", "It's sunny."),
        message("user", "input_audio", ""),                   # unintelligible turn
    ])
    kept = s.load("browser1")
    assert [i["id"] for i in kept] == ["item_Weat", "item_tool", "item_It's"]
    assert "audio" not in kept[0]["content"][0]
    assert len(kept[1]["output"]) == MAX_TOOL_OUTPUT_CHARS


def test_cap_ttl_and_clear(tmp_path):
    s = store(tmp_path, max_items=2)
    s.save("b", [message("user", "input_audio", f"turn {n}") for n in range(5)])
    assert [i["content"][0]["transcript"] for i in s.load("b")] == ["turn 3", "turn 4"]
    assert s.load("other") == []
    s.clear("b")
    assert s.load("b") == []
    expired = store(tmp_path, ttl_s=-1)
    expired.save("b", [message("user", "input_audio", "old")])
    assert expired.load("b") == []


@pytest.fixture
def client(tmp_path):
    with TestClient(app) as c:
        c.app.state.memory = store(tmp_path)
        yield c


def connect(client, cookie):
    return client.websocket_connect("/v1/realtime", headers={"cookie": f"{CLIENT_COOKIE}={cookie}"})


def test_conversation_is_restored_for_the_same_browser_only(client):
    with connect(client, "browser-aaaa") as ws:
        conn = Conn(ws)
        conn.until("session.created")
        user_text(conn, "My name is Samarth.")
        user_item = conn.events[-1]["item"]["id"]
        conn.send({"type": "response.create"})
        conn.until("response.done")

    with connect(client, "browser-aaaa") as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "conversation.item.retrieve", "item_id": user_item})
        item = conn.until("conversation.item.retrieved")[-1]["item"]
        assert item["content"][0]["text"] == "My name is Samarth."

    with connect(client, "browser-bbbb") as ws:
        conn = Conn(ws)
        conn.until("session.created")
        conn.send({"type": "conversation.item.retrieve", "item_id": user_item})
        assert conn.until("error")[-1]["error"]["code"] == "item_not_found"

    response = client.delete("/api/memory", cookies={CLIENT_COOKIE: "browser-aaaa"})
    assert response.json() == {"cleared": True}
    assert client.app.state.memory.load("browser-aaaa") == []


def test_browsers_get_a_stable_client_cookie(client):
    first = client.get("/api/config")
    cookie = first.cookies.get(CLIENT_COOKIE)
    assert cookie and len(cookie) >= 16
    again = client.get("/api/config", cookies={CLIENT_COOKIE: cookie})
    assert CLIENT_COOKIE not in again.cookies
    assert json.loads(again.text)["s2sUrl"] == "/v1/realtime"


def test_forgetting_mid_conversation_also_clears_the_live_session(client):
    with connect(client, "browser-cccc") as ws:
        conn = Conn(ws)
        conn.until("session.created")
        user_text(conn, "Remember the number 42.")
        user_item = conn.events[-1]["item"]["id"]
        conn.send({"type": "response.create"})
        conn.until("response.done")

        client.delete("/api/memory", cookies={CLIENT_COOKIE: "browser-cccc"})
        conn.send({"type": "conversation.item.retrieve", "item_id": user_item})
        assert conn.until("error")[-1]["error"]["code"] == "item_not_found"
    # The session's closing save must not resurrect what was forgotten.
    assert client.app.state.memory.load("browser-cccc") == []
