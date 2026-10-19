"""Conversation memory that survives reconnects.

Keyed by a per-browser cookie (set on every HTTP response, and sent by the
browser with the same-origin WebSocket handshake). Stores the realtime
conversation items the agent replays as context — transcripts and tool
calls/results, never audio — in SQLite on the persistent local disk.
"""

import copy
import json
import sqlite3
import threading
import time
from pathlib import Path

CLIENT_COOKIE = "dra_client"
MAX_TOOL_OUTPUT_CHARS = 2000


class ConversationStore:
    def __init__(self, path: str, *, ttl_s: float, max_items: int, startup_prompt: str) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl_s = ttl_s
        self._max_items = max_items
        self._startup_prompt = startup_prompt.strip()
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS conversations ("
                       "client_id TEXT PRIMARY KEY, items TEXT NOT NULL, updated_at REAL NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5.0)

    def load(self, client_id: str) -> list[dict]:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT items, updated_at FROM conversations WHERE client_id = ?",
                             (client_id,)).fetchone()
        if row is None or time.time() - row[1] > self._ttl_s:
            return []
        return json.loads(row[0])

    def save(self, client_id: str, items: list[dict]) -> None:
        kept = self._persistable(items)[-self._max_items:]
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO conversations (client_id, items, updated_at) VALUES (?, ?, ?) "
                       "ON CONFLICT(client_id) DO UPDATE SET items = excluded.items, "
                       "updated_at = excluded.updated_at",
                       (client_id, json.dumps(kept), time.time()))

    def clear(self, client_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM conversations WHERE client_id = ?", (client_id,))

    def _persistable(self, items: list[dict]) -> list[dict]:
        """Finished transcripts and tool results only. The hidden startup-greeting
        prompt (sent on every connection) and the greeting it produced are skipped."""
        kept: list[dict] = []
        skip_greeting_reply = False
        for item in items:
            kind, role = item.get("type"), item.get("role")
            if kind == "message" and role == "user" and self._is_startup_prompt(item):
                skip_greeting_reply = True
                continue
            if kind == "message" and role == "assistant" and skip_greeting_reply:
                skip_greeting_reply = False
                continue
            if kind == "message" and role in ("user", "assistant") and _has_text(item):
                kept.append(_text_only(item))
            elif kind == "mcp_call" and item.get("output") is not None:
                tool = copy.deepcopy(item)
                tool["output"] = tool["output"][:MAX_TOOL_OUTPUT_CHARS]
                kept.append(tool)
        return kept

    def _is_startup_prompt(self, item: dict) -> bool:
        return bool(self._startup_prompt) and any(
            (part.get("text") or "").strip() == self._startup_prompt
            for part in item.get("content") or [])


def _has_text(item: dict) -> bool:
    return any((p.get("text") or p.get("transcript") or "").strip() for p in item.get("content") or [])


def _text_only(item: dict) -> dict:
    out = {k: v for k, v in item.items() if k != "content"}
    out["content"] = [{k: v for k, v in part.items() if k != "audio"}
                      for part in item.get("content") or [] if part.get("type") != "input_image"]
    return copy.deepcopy(out)
