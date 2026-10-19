"""One realtime conversation over a WebSocket (OpenAI Realtime GA protocol).

The browser speaks to us exactly as it would to the OpenAI Realtime API (via
the @openai/agents SDK). This class owns the per-connection state — session
config, conversation items, input audio, turn detection, the active response
— and drives the pluggable pipeline (VAD -> STT -> responder -> TTS).

User turn (server VAD):

  input_audio_buffer.speech_started -> [barge-in: cancel active response]
  -> input_audio_buffer.speech_stopped -> input_audio_buffer.committed
  -> conversation.item.added (user audio) -> STT
  -> conversation.item.input_audio_transcription.completed
  -> conversation.item.done -> [auto response.create]

Response lifecycle, emitted for every response (also when cancelled or failed,
because the SDK blocks further responses until it sees ``response.done``):

  response.created -> response.output_item.added -> conversation.item.added
  -> response.content_part.added -> (output_audio_transcript.delta,
  output_audio.delta...)* -> output_audio.done -> output_audio_transcript.done
  -> content_part.done -> output_item.done -> conversation.item.done
  -> response.done
"""

import asyncio
import base64
import binascii
import copy
import json
import logging
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from app.pipeline.base import (SAMPLE_RATE, Pipeline, ToolCall, TurnContext, TurnDetection,
                               VADStream)
from app.pipeline.text import SentenceBuffer
from app.realtime import events as ev
from app.realtime.memory import ConversationStore

logger = logging.getLogger("deepagent.realtime")

AUDIO_DELTA_MS = 100
_BYTES_PER_MS = SAMPLE_RATE * 2 // 1000
MAX_INPUT_BUFFER_S = 60
ITEM_TYPES = {"message", "function_call", "function_call_output"}
FAILURE_REPLY = "Sorry, I ran into a problem answering that."


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@dataclass
class _ActiveResponse:
    id: str
    item_id: str
    task: asyncio.Task | None = None
    # A task cancelled before its first step never runs its body (so it would
    # never send response.done). Until it starts, cancellation is cooperative.
    started: bool = False
    finalizing: bool = False
    cancel_reason: str = ""


class RealtimeSession:
    def __init__(self, ws: WebSocket, pipeline: Pipeline, *, instructions: str,
                 memory: ConversationStore | None = None, client_id: str | None = None) -> None:
        self.ws = ws
        self.pipeline = pipeline
        self.session = ev.default_session(ev.new_id("sess"), instructions)
        self.conversation_id = ev.new_id("conv")
        # Persistent memory: earlier turns from this browser are restored as context.
        self._memory = memory if client_id else None
        self._client_id = client_id
        self.items: list[dict] = self._memory.load(client_id) if self._memory else []
        # Per assistant item: [(audio_start_ms, transcript_delta)]. Lets a
        # truncation cut the stored transcript to what the user actually heard.
        self._timelines: dict[str, list[tuple[float, str]]] = {}
        self._heard_limit_ms: dict[str, float] = {}
        # Rolling input audio; _buf_start_ms is the stream time of its first byte.
        self._input_audio = bytearray()
        self._audio_ms = 0.0
        self._buf_start_ms = 0.0
        self._vad: VADStream | None = None
        self._vad_cfg: dict | None = None
        self._speech: tuple[str, float] | None = None  # (item_id, start_ms) while the user talks
        self._active: _ActiveResponse | None = None
        self._background: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()
        self._closed = False

    # ── Transport ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("Session %s opened (%d remembered items)", self.session["id"], len(self.items))
        await self._configure_vad()
        await self.send(ev.event("session.created", session=self.session))
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await self.send(ev.error("Invalid JSON.", code="invalid_json"))
                    continue
                if not isinstance(msg, dict):
                    await self.send(ev.error("Event must be a JSON object.", code="invalid_event"))
                    continue
                await self._dispatch(msg)
        except WebSocketDisconnect:
            pass
        finally:
            self._closed = True
            for task in list(self._background):
                task.cancel()
            await self._cancel_active("client_disconnected")
            await self._remember()
            logger.info("Session %s closed", self.session["id"])

    async def _remember(self) -> None:
        if self._memory is None:
            return
        snapshot = copy.deepcopy(self.items)
        try:
            await asyncio.to_thread(self._memory.save, self._client_id, snapshot)
        except Exception:
            logger.exception("Saving conversation memory failed")

    @property
    def client_id(self) -> str | None:
        return self._client_id

    def forget(self) -> None:
        """Drop the conversation so far (keeping only a reply still being spoken,
        which the running response references)."""
        keep = self._active.item_id if self._active else None
        self.items = [i for i in self.items if i.get("id") == keep]
        self._timelines = {k: v for k, v in self._timelines.items() if k == keep}

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self._closed = True
        try:
            await self.ws.close(code=code, reason=reason)
        except RuntimeError:
            pass

    async def send(self, event: dict) -> None:
        if self._closed:
            return
        # Shielded so cancelling a response mid-send never leaves a half-written frame.
        await asyncio.shield(self._send_now(json.dumps(event)))

    async def _send_now(self, text: str) -> None:
        async with self._send_lock:
            if self._closed:
                return
            try:
                await self.ws.send_text(text)
            except (WebSocketDisconnect, RuntimeError, OSError):
                self._closed = True

    async def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        handler = self._HANDLERS.get(kind)
        if kind != "input_audio_buffer.append":
            logger.debug("<- %s", kind)
        if handler is None:
            await self.send(ev.error(f"Unsupported event type: {kind}", code="unknown_event",
                                     client_event_id=msg.get("event_id")))
            return
        await handler(self, msg)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # ── Client event handlers ──────────────────────────────────────────────

    async def _on_session_update(self, msg: dict) -> None:
        patch = msg.get("session")
        if not isinstance(patch, dict):
            await self.send(ev.error("session must be an object.", code="invalid_value",
                                     client_event_id=msg.get("event_id")))
            return
        sid = self.session["id"]
        _deep_merge(self.session, patch)
        self.session.update(type="realtime", object="realtime.session", id=sid)
        await self._configure_vad()
        await self.send(ev.event("session.updated", session=self.session))

    async def _on_audio_append(self, msg: dict) -> None:
        try:
            chunk = base64.b64decode(msg.get("audio") or "", validate=True)
        except (binascii.Error, ValueError):
            await self.send(ev.error("audio must be base64 PCM16.", code="invalid_value",
                                     client_event_id=msg.get("event_id")))
            return
        chunk = chunk[: len(chunk) - len(chunk) % 2]
        self._input_audio += chunk
        self._audio_ms += len(chunk) / _BYTES_PER_MS
        overflow = len(self._input_audio) - MAX_INPUT_BUFFER_S * SAMPLE_RATE * 2
        if overflow > 0:
            del self._input_audio[:overflow]
            self._buf_start_ms += overflow / _BYTES_PER_MS

        vad = self._vad
        if vad is None or not chunk:
            return
        for kind, ms in await asyncio.to_thread(vad.process, chunk):
            if kind == "start":
                await self._on_speech_start(ms)
            else:
                await self._on_speech_end(ms)

    async def _on_audio_clear(self, msg: dict) -> None:
        self._input_audio.clear()
        self._buf_start_ms = self._audio_ms
        await self.send(ev.event("input_audio_buffer.cleared"))

    async def _on_item_create(self, msg: dict) -> None:
        item = copy.deepcopy(msg.get("item"))
        if not isinstance(item, dict) or item.get("type") not in ITEM_TYPES:
            await self.send(ev.error("Unsupported item type.", code="invalid_value",
                                     client_event_id=msg.get("event_id")))
            return
        item.setdefault("id", ev.new_id("item"))
        item["object"] = "realtime.item"
        if item["type"] == "message":
            item.setdefault("status", "completed")
        previous_id = self._insert(item, msg.get("previous_item_id"))
        await self.send(ev.event("conversation.item.added", item=item, previous_item_id=previous_id))
        await self.send(ev.event("conversation.item.done", item=item, previous_item_id=previous_id))

    async def _on_item_retrieve(self, msg: dict) -> None:
        item = self._find(msg.get("item_id"))
        if item is None:
            await self._item_not_found(msg)
            return
        await self.send(ev.event("conversation.item.retrieved", item=item))

    async def _on_item_delete(self, msg: dict) -> None:
        item = self._find(msg.get("item_id"))
        if item is None:
            await self._item_not_found(msg)
            return
        self.items.remove(item)
        self._timelines.pop(item["id"], None)
        await self.send(ev.event("conversation.item.deleted", item_id=item["id"]))

    async def _on_item_truncate(self, msg: dict) -> None:
        item = self._find(msg.get("item_id"))
        if item is None or item.get("role") != "assistant":
            await self._item_not_found(msg)
            return
        end_ms = max(0.0, float(msg.get("audio_end_ms") or 0))
        self._heard_limit_ms[item["id"]] = end_ms
        heard = self._heard_transcript(item["id"], end_ms)
        for part in item.get("content") or []:
            if part.get("type") == "output_audio":
                part["transcript"] = heard
        await self.send(ev.event("conversation.item.truncated", item_id=item["id"],
                                 content_index=int(msg.get("content_index") or 0),
                                 audio_end_ms=int(end_ms)))

    async def _on_response_create(self, msg: dict) -> None:
        if self._active is not None:
            await self.send(ev.error(
                f"Conversation already has an active response in progress: {self._active.id}.",
                code="conversation_already_has_active_response",
                client_event_id=msg.get("event_id"),
            ))
            return
        self._start_response()

    async def _on_response_cancel(self, msg: dict) -> None:
        if self._active is None:
            await self.send(ev.error("Cancellation failed: no active response found.",
                                     code="response_cancel_not_active",
                                     client_event_id=msg.get("event_id")))
            return
        await self._cancel_active("client_cancelled")

    _HANDLERS = {
        "session.update": _on_session_update,
        "input_audio_buffer.append": _on_audio_append,
        "input_audio_buffer.clear": _on_audio_clear,
        "conversation.item.create": _on_item_create,
        "conversation.item.retrieve": _on_item_retrieve,
        "conversation.item.delete": _on_item_delete,
        "conversation.item.truncate": _on_item_truncate,
        "response.create": _on_response_create,
        "response.cancel": _on_response_cancel,
    }

    # ── User turns (server VAD -> STT) ─────────────────────────────────────

    def _turn_detection(self) -> dict | None:
        return self.session["audio"]["input"].get("turn_detection")

    async def _configure_vad(self) -> None:
        cfg = self._turn_detection()
        if cfg == self._vad_cfg and (self._vad is not None or cfg is None):
            return
        if self._speech is not None:
            await self._on_speech_end(self._audio_ms)  # don't strand an open turn
        self._vad_cfg = copy.deepcopy(cfg)
        enabled = isinstance(cfg, dict) and cfg.get("type") in ("server_vad", "semantic_vad")
        self._vad = self.pipeline.vad(TurnDetection.from_session(cfg)) if enabled else None

    async def _on_speech_start(self, start_ms: float) -> None:
        item_id = ev.new_id("item")
        self._speech = (item_id, start_ms)
        await self.send(ev.event("input_audio_buffer.speech_started",
                                 audio_start_ms=int(start_ms), item_id=item_id))
        if (self._turn_detection() or {}).get("interrupt_response", True):
            await self._cancel_active("turn_detected")

    async def _on_speech_end(self, end_ms: float) -> None:
        if self._speech is None:
            return
        item_id, start_ms = self._speech
        self._speech = None
        await self.send(ev.event("input_audio_buffer.speech_stopped",
                                 audio_end_ms=int(end_ms), item_id=item_id))
        pcm = self._slice_input(start_ms, end_ms)
        item = ev.user_audio_item(item_id)
        previous_id = self._insert(item, None)
        await self.send(ev.event("input_audio_buffer.committed", item_id=item_id,
                                 previous_item_id=previous_id))
        await self.send(ev.event("conversation.item.added", item=copy.deepcopy(item),
                                 previous_item_id=previous_id))
        self._spawn(self._transcribe(item, pcm, previous_id))

    async def _transcribe(self, item: dict, pcm: bytes, previous_id: str | None) -> None:
        transcription = self.session["audio"]["input"].get("transcription") or {}
        try:
            text = await self.pipeline.stt.transcribe(pcm, transcription.get("language") or "auto")
        except Exception as exc:
            logger.exception("Transcription of %s failed", item["id"])
            await self.send(ev.event("conversation.item.input_audio_transcription.failed",
                                     item_id=item["id"], content_index=0, error={
                                         "type": "transcription_error", "code": "stt_error",
                                         "message": str(exc), "param": None}))
            await self.send(ev.event("conversation.item.done", item=copy.deepcopy(item),
                                     previous_item_id=previous_id))
            return
        item["content"][0]["transcript"] = text
        logger.info("User said (%.1fs): %r", len(pcm) / _BYTES_PER_MS / 1000, text)
        await self.send(ev.event("conversation.item.input_audio_transcription.completed",
                                 item_id=item["id"], content_index=0, transcript=text,
                                 usage=ev.TRANSCRIPTION_USAGE))
        await self.send(ev.event("conversation.item.done", item=copy.deepcopy(item),
                                 previous_item_id=previous_id))
        # Reply only to real speech, and not while the user has started talking again.
        td = self._turn_detection() or {}
        if text and td.get("create_response", True) and self._speech is None and self._active is None:
            self._start_response()

    def _slice_input(self, start_ms: float, end_ms: float) -> bytes:
        a = int((start_ms - self._buf_start_ms) * _BYTES_PER_MS)
        b = int((end_ms - self._buf_start_ms) * _BYTES_PER_MS)
        a, b = max(0, a - a % 2), min(len(self._input_audio), b - b % 2)
        return bytes(self._input_audio[a:b]) if b > a else b""

    # ── Responses ──────────────────────────────────────────────────────────

    def _start_response(self) -> None:
        """Reserve the response slot and start it (synchronous, so no event can slip in)."""
        voice = self.session["audio"]["output"].get("voice") or "alloy"
        transcription = self.session["audio"]["input"].get("transcription") or {}
        turn = TurnContext(
            instructions=self.session.get("instructions") or "",
            voice=voice,
            language=transcription.get("language") or "auto",
            items=copy.deepcopy(self.items),
        )
        active = _ActiveResponse(ev.new_id("resp"), ev.new_id("item"))
        self._active = active
        active.task = asyncio.create_task(self._run_response(active, voice, turn))

    async def _cancel_active(self, reason: str) -> None:
        active = self._active
        if active is None or active.task is None or active.task.done():
            return
        if not active.cancel_reason and not active.finalizing:
            active.cancel_reason = reason
            if active.started:
                active.task.cancel()
        try:
            await active.task
        except asyncio.CancelledError:
            pass

    async def _run_response(self, active: _ActiveResponse, voice: str, turn: TurnContext) -> None:
        active.started = True
        response_id, item_id = active.id, active.item_id
        item = ev.assistant_item(item_id, status="in_progress", transcript=None)
        previous_id = self.items[-1]["id"] if self.items else None
        self.items.append(item)
        timeline = self._timelines.setdefault(item_id, [])
        part_where = {"response_id": response_id, "item_id": item_id, "output_index": 0,
                      "content_index": 0}
        transcript = ""
        audio_ms = 0.0
        chunk_bytes = AUDIO_DELTA_MS * _BYTES_PER_MS
        # Server-side tool calls (agent tools) as `mcp_call` items: the SDK
        # records them in history but never tries to execute them in the browser.
        tool_items: dict[str, tuple[dict, int, str | None]] = {}  # call_id -> (item, index, previous)

        async def speak(sentence: str) -> None:
            nonlocal transcript, audio_ms
            delta = sentence if not transcript else f" {sentence}"
            timeline.append((audio_ms, delta))
            transcript += delta
            await self.send(ev.event("response.output_audio_transcript.delta", **part_where,
                                     delta=delta))
            pending = bytearray()

            async def flush(data: bytes) -> None:
                nonlocal audio_ms
                await self.send(ev.event("response.output_audio.delta", **part_where,
                                         delta=base64.b64encode(data).decode("ascii")))
                audio_ms += len(data) / _BYTES_PER_MS

            async for pcm in self.pipeline.tts.synthesize(sentence, voice):
                pending += pcm
                while len(pending) >= chunk_bytes:
                    await flush(bytes(pending[:chunk_bytes]))
                    del pending[:chunk_bytes]
            if len(pending) > 1:
                await flush(bytes(pending[: len(pending) - len(pending) % 2]))

        async def finish_tool(tool_item: dict, index: int, previous: str | None) -> None:
            await self.send(ev.event("response.output_item.done", response_id=response_id,
                                     output_index=index, item=copy.deepcopy(tool_item)))
            await self.send(ev.event("conversation.item.done", item=copy.deepcopy(tool_item),
                                     previous_item_id=previous))

        async def tool_activity(call: ToolCall) -> None:
            if call.call_id not in tool_items:
                tool_item = ev.mcp_call_item(ev.new_id("item"), name=call.name,
                                             arguments=call.arguments)
                # History order: user -> tool call/result -> the reply that used it.
                at = self.items.index(item)
                previous = self.items[at - 1]["id"] if at else None
                self.items.insert(at, tool_item)
                tool_items[call.call_id] = (tool_item, len(tool_items) + 1, previous)
                await self.send(ev.event("response.output_item.added", response_id=response_id,
                                         output_index=len(tool_items),
                                         item=copy.deepcopy(tool_item)))
                await self.send(ev.event("conversation.item.added", item=copy.deepcopy(tool_item),
                                         previous_item_id=previous))
            tool_item, index, previous = tool_items[call.call_id]
            if call.output is None and call.error is None:
                return
            ev.finish_mcp_call(tool_item, output=call.output, error=call.error)
            await finish_tool(tool_item, index, previous)

        status, details = "completed", None
        try:
            await self.send(ev.event("response.created", response=ev.response_obj(
                response_id, self.conversation_id, voice, status="in_progress")))
            await self.send(ev.event("response.output_item.added", response_id=response_id,
                                     output_index=0, item=copy.deepcopy(item)))
            await self.send(ev.event("conversation.item.added", item=copy.deepcopy(item),
                                     previous_item_id=previous_id))
            await self.send(ev.event("response.content_part.added", **part_where,
                                     part={"type": "audio", "transcript": ""}))
            if active.cancel_reason:
                raise asyncio.CancelledError
            # The responder runs in its own task so the agent keeps generating
            # while earlier sentences are being synthesized and sent.
            queue: asyncio.Queue = asyncio.Queue()
            end = object()

            async def produce() -> None:
                try:
                    async for out in self.pipeline.responder.respond(turn):
                        queue.put_nowait(out)
                    queue.put_nowait(end)
                except Exception as exc:  # re-raised by the consumer below
                    queue.put_nowait(exc)

            producer = asyncio.create_task(produce())
            try:
                sentences = SentenceBuffer()
                while (chunk := await queue.get()) is not end:
                    if isinstance(chunk, Exception):
                        raise chunk
                    if isinstance(chunk, ToolCall):
                        for sentence in sentences.flush():
                            await speak(sentence)
                        await tool_activity(chunk)
                        continue
                    for sentence in sentences.feed(chunk):
                        await speak(sentence)
                for sentence in sentences.flush():
                    await speak(sentence)
            finally:
                producer.cancel()
        except asyncio.CancelledError:
            status = "cancelled"
            details = {"type": "cancelled", "reason": active.cancel_reason or "client_cancelled"}
        except Exception as exc:
            logger.exception("Response %s failed", response_id)
            status, details = "failed", {"type": "failed", "error": {
                "type": "server_error", "code": "pipeline_error", "message": str(exc)}}
            try:  # don't leave the user in silence
                await speak(FAILURE_REPLY)
            except Exception:
                logger.exception("Could not speak the failure notice")

        active.finalizing = True
        try:
            for tool_item, index, previous in tool_items.values():
                if tool_item["status"] == "in_progress":
                    ev.finish_mcp_call(tool_item, output=None, error=f"Tool call {status}.")
                    await finish_tool(tool_item, index, previous)
            if item_id in self._heard_limit_ms:
                transcript = self._heard_transcript(item_id, self._heard_limit_ms[item_id])
            item["status"] = "completed" if status == "completed" else "incomplete"
            item["content"] = [{"type": "output_audio", "transcript": transcript}]

            await self.send(ev.event("response.output_audio.done", **part_where))
            await self.send(ev.event("response.output_audio_transcript.done", **part_where,
                                     transcript=transcript))
            await self.send(ev.event("response.content_part.done", **part_where,
                                     part={"type": "audio", "transcript": transcript}))
            await self.send(ev.event("response.output_item.done", response_id=response_id,
                                     output_index=0, item=copy.deepcopy(item)))
            await self.send(ev.event("conversation.item.done", item=copy.deepcopy(item),
                                     previous_item_id=previous_id))
            # Free the slot before response.done: the SDK may send the next
            # response.create the moment it sees it.
            self._active = None
            await self.send(ev.event("response.done", response=ev.response_obj(
                response_id, self.conversation_id, voice, status=status,
                output=[copy.deepcopy(item)] + [copy.deepcopy(t) for t, _, _ in tool_items.values()],
                status_details=details)))
        finally:
            if self._active is active:
                self._active = None
        logger.info("Response %s %s (%.1fs audio): %r", response_id, status, audio_ms / 1000,
                    transcript[:120])
        await self._remember()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _find(self, item_id) -> dict | None:
        return next((i for i in self.items if i.get("id") == item_id), None)

    def _insert(self, item: dict, previous_item_id) -> str | None:
        """Insert after ``previous_item_id`` (or append); returns the actual predecessor id."""
        if previous_item_id == "root":
            self.items.insert(0, item)
            return None
        anchor = self._find(previous_item_id) if previous_item_id else None
        if anchor is not None:
            self.items.insert(self.items.index(anchor) + 1, item)
            return anchor["id"]
        previous = self.items[-1]["id"] if self.items else None
        self.items.append(item)
        return previous

    def _heard_transcript(self, item_id: str, end_ms: float) -> str:
        return "".join(d for start, d in self._timelines.get(item_id, []) if start < end_ms).strip()

    async def _item_not_found(self, msg: dict) -> None:
        await self.send(ev.error(f"Item {msg.get('item_id')!r} not found.", code="item_not_found",
                                 client_event_id=msg.get("event_id")))
