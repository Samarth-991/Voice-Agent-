"""
FastAPI server for the DeepAgent realtime voice demo.

Serves the (reused) browser UI from ``web/`` and an OpenAI Realtime
GA-compatible WebSocket at ``/v1/realtime`` that runs the pluggable pipeline
(VAD -> STT -> DeepAgent on Azure OpenAI -> TTS) server-side.

Endpoints:
  WS     /v1/realtime                          -> realtime conversation (one at a time)
  GET    /api/config                           -> client bootstrap (direct mode, pinned URL)
  DELETE /api/memory                           -> forget this browser's remembered conversation
  GET    /vendor/openai-realtime-agents.umd.js -> pinned Agents SDK browser bundle
  /*                                           -> static files from web/
"""

import logging
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.pipeline import build_pipeline
from app.realtime.memory import CLIENT_COOKIE, ConversationStore
from app.realtime.session import RealtimeSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# deepagents probes for its optional [video] extra; `av` (from faster-whisper)
# without Pillow makes that probe log a harmless traceback on every start.
logging.getLogger("deepagents.middleware._video").setLevel(logging.ERROR)
# azure-identity logs every managed-identity token request at INFO.
logging.getLogger("azure").setLevel(logging.WARNING)
logger = logging.getLogger("deepagent.server")

DEFAULT_INSTRUCTIONS = "You are a friendly voice assistant."


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = build_pipeline()
    app.state.memory = (
        ConversationStore(config.MEMORY_DB, ttl_s=config.MEMORY_TTL_H * 3600,
                          max_items=config.MEMORY_MAX_ITEMS, startup_prompt=config.STARTUP_GREETING)
        if config.MEMORY == "persistent" else None
    )
    app.state.active_session = None
    yield


app = FastAPI(title="deepagent-realtime-voice", lifespan=lifespan)


def _client_id(cookies) -> str | None:
    value = cookies.get(CLIENT_COOKIE, "")
    return value if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", value) else None


@app.middleware("http")
async def client_cookie(request: Request, call_next):
    """Give each browser a stable id; the WebSocket handshake sends it back for memory."""
    response = await call_next(request)
    if _client_id(request.cookies) is None:
        response.set_cookie(CLIENT_COOKIE, secrets.token_urlsafe(16), max_age=365 * 86400,
                            httponly=True, samesite="lax")
    return response


@app.websocket(config.REALTIME_PATH)
async def realtime(ws: WebSocket):
    await ws.accept()
    # Single-conversation mode: the newest connection wins, so a stale tab
    # can never lock you out.
    previous: RealtimeSession | None = app.state.active_session
    if previous is not None:
        logger.info("New connection replaces the active conversation")
        await previous.close(code=4000, reason="Replaced by a newer connection")
    session = RealtimeSession(ws, app.state.pipeline, instructions=DEFAULT_INSTRUCTIONS,
                              memory=app.state.memory, client_id=_client_id(ws.cookies))
    app.state.active_session = session
    try:
        await session.run()
    finally:
        if app.state.active_session is session:
            app.state.active_session = None


@app.delete("/api/memory")
def forget_conversation(request: Request):
    """Forget this browser's conversation: the stored memory and, if it is
    talking right now, the live session's history."""
    client_id = _client_id(request.cookies)
    if client_id:
        if app.state.memory is not None:
            app.state.memory.clear(client_id)
        live: RealtimeSession | None = app.state.active_session
        if live is not None and live.client_id == client_id:
            live.forget()
    return {"cleared": True}


@app.get("/api/config")
def api_config():
    """Same shape the original UI expects. Always direct mode, pinned to our own
    realtime endpoint; no load balancer, sign-in, metering, or WebRTC."""
    return {
        # Web search runs inside the DeepAgent (server-side), not as a browser tool.
        "search": False,
        "lb": False,
        "allowDirect": True,
        "s2sUrl": config.REALTIME_URL,
        "rtc": False,
        "iceServers": [],
        "startupGreeting": config.STARTUP_GREETING,
        "auth": False,
        "requireLogin": False,
    }


@app.get("/vendor/openai-realtime-agents.umd.js", include_in_schema=False)
def agents_sdk_bundle():
    """Serve the npm-pinned Agents SDK browser bundle (see scripts/setup_web.sh)."""
    path = config.WEB_DIR / "vendor" / "openai-realtime-agents.umd.js"
    if path.is_file():
        return FileResponse(path, media_type="text/javascript")
    raise HTTPException(status_code=503, detail="Run scripts/setup_web.sh to fetch the Agents SDK bundle.")


# Registered last so the API/WebSocket routes win; html=True serves index.html at "/".
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="static")
