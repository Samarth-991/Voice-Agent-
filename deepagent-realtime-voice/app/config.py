"""Environment-driven settings for the DeepAgent realtime voice server."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"

# Path of our own OpenAI Realtime GA-compatible WebSocket endpoint.
REALTIME_PATH = "/v1/realtime"

# URL handed to the browser. A same-origin path by default so it works behind
# the AzureML proxy / VS Code port forwarding; override with an absolute
# ws(s):// URL to point the UI at a different backend.
REALTIME_URL = os.environ.get("REALTIME_URL", REALTIME_PATH).strip() or REALTIME_PATH

DEFAULT_STARTUP_GREETING = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)
# Set to an empty value to disable the automatic greeting.
STARTUP_GREETING = os.environ.get("STARTUP_GREETING", DEFAULT_STARTUP_GREETING).strip()

# Pipeline engine selection (pluggable).
RESPONDER = os.environ.get("RESPONDER", "deepagent").strip().lower()  # deepagent | mock
TTS_ENGINE = os.environ.get("TTS", "kokoro").strip().lower()  # kokoro | mock
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda").strip()
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))
VAD_ENGINE = os.environ.get("VAD", "silero").strip().lower()  # silero | energy
STT_ENGINE = os.environ.get("STT", "faster-whisper").strip().lower()  # faster-whisper | mock

STT_MODEL = os.environ.get("STT_MODEL", "distil-large-v3").strip()
STT_DEVICE = os.environ.get("STT_DEVICE", "cuda").strip()
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "float16").strip()
# Forced transcription language; distil-large-v3 is English only. Empty = use the UI's choice.
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en").strip()

# DeepAgent: the org's shared Azure OpenAI helper (exposes build_azure_llm()).
LLM_SERVICE_PATH = os.environ.get(
    "LLM_SERVICE_PATH",
    "/home/azureuser/cloudfiles/code/Users/samarth.tandon/workspace/FleetAgent2/llm_service.py",
).strip()
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))
SEARCH_TIMEOUT_S = int(os.environ.get("SEARCH_TIMEOUT_S", "10"))
FETCH_TIMEOUT_S = float(os.environ.get("FETCH_TIMEOUT_S", "8"))
FETCH_MAX_CHARS = int(os.environ.get("FETCH_MAX_CHARS", "6000"))
# The compute's corporate proxy refuses general web sites while direct egress
# works (llm_service.py blanks the proxy env for the same reason).
FETCH_USE_ENV_PROXY = os.environ.get("FETCH_USE_ENV_PROXY", "0").strip() in {"1", "true", "yes"}
AGENT_RECURSION_LIMIT = int(os.environ.get("AGENT_RECURSION_LIMIT", "25"))

# Conversation memory: "persistent" (per browser, survives reconnects) or "connection".
MEMORY = os.environ.get("MEMORY", "persistent").strip().lower()
# On the persistent local disk (cloudfiles' CIFS share is unsafe for SQLite locking).
MEMORY_DB = os.environ.get("MEMORY_DB", "~/.local/share/deepagent-rtv/conversations.sqlite3").strip()
MEMORY_TTL_H = float(os.environ.get("MEMORY_TTL_H", "24"))
MEMORY_MAX_ITEMS = int(os.environ.get("MEMORY_MAX_ITEMS", "60"))
