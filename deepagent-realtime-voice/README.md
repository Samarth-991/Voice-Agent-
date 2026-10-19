# DeepAgent Realtime Voice

Voice conversation UI (reused from `hf-realtime-voice`) backed by our own
agentic harness: a FastAPI server that speaks the **OpenAI Realtime GA**
WebSocket protocol, so the browser client works unchanged, and runs the
pipeline server-side:

`mic → VAD → STT → DeepAgent (Azure OpenAI + web tools + research subagent) → TTS → speaker`

## Setup (AzureML compute)

```bash
bash scripts/setup_env.sh   # venv + model caches on /mnt/cache (ephemeral: rerun after compute restart)
bash scripts/run.sh         # serves http://0.0.0.0:7860
```

Forward port **7860** in VS Code (Ports tab) and open <http://localhost:7860>
(mic access requires localhost or HTTPS). Optional settings go in `.env`.

| Env | Default | What |
|-----|---------|------|
| `VAD` | `silero` | Turn detection (`silero` on CPU, or `energy` = loudness only) |
| `STT` | `faster-whisper` | Transcription (`faster-whisper` on the GPU, or `mock`) |
| `STT_MODEL` | `distil-large-v3` | faster-whisper model (English only; downloads ~1.5 GB on first start) |
| `STT_LANGUAGE` | `en` | Forced transcription language; empty = use the UI's language choice |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` | `cuda` / `float16` | Whisper placement |
| `RESPONDER` | `deepagent` | Reply engine (`deepagent` = Azure OpenAI Deep Agent + web search, or `mock` echo) |
| `LLM_SERVICE_PATH` | `…/FleetAgent2/llm_service.py` | Shared helper exposing `build_azure_llm()` (managed identity; reads `AZURE_OPENAI_*` env) |
| `SEARCH_MAX_RESULTS` / `SEARCH_TIMEOUT_S` | `5` / `10` | DuckDuckGo search tool |
| `FETCH_TIMEOUT_S` / `FETCH_MAX_CHARS` | `8` / `6000` | `fetch_page` tool |
| `FETCH_USE_ENV_PROXY` | `0` | Route `fetch_page` through `http(s)_proxy` (the compute's proxy refuses general sites) |
| `AGENT_RECURSION_LIMIT` | `25` | Max agent graph steps per reply |
| `MEMORY` | `persistent` | `persistent` (per browser, survives reconnects) or `connection` |
| `MEMORY_DB` / `MEMORY_TTL_H` / `MEMORY_MAX_ITEMS` | `~/.local/share/deepagent-rtv/conversations.sqlite3` / `24` / `60` | Memory store |
| `TTS` | `kokoro` | Speech engine (`kokoro` = Kokoro-82M on the GPU, or `mock` = soft tones per word) |
| `TTS_DEVICE` / `TTS_SPEED` | `cuda` / `1.0` | Kokoro placement and speaking rate |
| `STARTUP_GREETING` | built-in prompt | Hidden prompt for the opening greeting; empty disables it |
| `REALTIME_URL` | `/v1/realtime` | WebSocket URL handed to the browser |

Turn detection uses the session's `turn_detection` (threshold 0.5, 300 ms
prefix padding, **600 ms** end-of-turn silence, 200 ms minimum speech).
Speaking over the assistant cancels its reply (barge-in).

## The agent

`app/agent/responder.py` builds a LangChain **Deep Agent** on the Azure OpenAI
model from `llm_service.py`:

- **Tools:** `web_search` (DuckDuckGo) and `fetch_page` (reads a page's main
  text with trafilatura). `fetch_page` only reaches public internet addresses —
  every hop, including redirects, is resolved and checked — because pages can
  carry prompt injections and this compute exposes credential endpoints on
  loopback/link-local addresses. Tools never raise: failures come back to the
  model as text.
- **Planning:** `write_todos` (LangChain `TodoListMiddleware`; deepagents 0.7
  no longer adds it by default), prompted for genuinely multi-step requests only.
- **Research subagent:** `researcher` (via the `task` tool) with the two web
  tools and a four-call budget, for comparisons and in-depth questions. The
  virtual-filesystem tools and the general-purpose subagent are hidden by a
  harness profile (`VOICE_PROFILE`).
- **Streaming:** the main agent's tokens stream into sentence-level TTS while
  the agent keeps running (the responder runs in its own task). Subagent
  tokens are never spoken.
- **Acknowledgement:** when the agent starts a tool call without saying
  anything, a short line is spoken first ("Let me look that up.", "Let me
  research that, it may take a moment.", …), then the answer. If a reply
  fails, "Sorry, I ran into a problem answering that." is spoken.
- **Tool activity** is sent as `mcp_call` items (server-executed tools in the
  Realtime protocol) and shown as tool rows in the chat history.

The system prompt asks for short, plain spoken sentences; the UI's
Instructions setting is appended. Typical timings: small talk ~1.2 s; first
words of a search answer ~1 s, full answer ~3–4 s; researched answers ~15–20 s.

## Memory

Each turn replays the conversation items (what you actually heard after
barge-in, plus earlier tool calls and results) as the agent's messages. With
`MEMORY=persistent` those items are also saved per browser (a `dra_client`
cookie, sent with the WebSocket handshake) for `MEMORY_TTL_H` hours, so the
assistant remembers you across reconnects and Restart. Only transcripts and
tool results are stored (never audio); the hidden startup-greeting prompt is
skipped. **Settings → Forget our conversation** (`DELETE /api/memory`) clears
both the stored memory and the live session's history.

## Voices

Replies are spoken by Kokoro-82M (24 kHz, ~0.1 s per sentence on the T4),
synthesized sentence by sentence. The Settings voice list keeps the original
names, mapped to similar Kokoro English voices (`app/pipeline/tts.py`):

| UI | Kokoro | UI | Kokoro | UI | Kokoro |
|----|--------|----|--------|----|--------|
| Aiden | am_michael | Ono_Anna | af_bella | Uncle_Fu | bm_george |
| Ryan | am_fenrir | Serena | af_heart | Vivian | bf_emma |
| Dylan | am_puck | Sohee | af_nicole | | |
| Eric | am_eric | | | | |

All mapped voices are warmed up at startup (~16 s total server start).
`transformers` is pinned `<5` because 5.x needs torch ≥ 2.6, and this GPU
driver (CUDA 12.2) runs the cu121 torch 2.5.1 build.

## Layout

| Path | Role |
|------|------|
| `app/server.py` | FastAPI app: `/v1/realtime` WebSocket, `/api/config`, `/api/memory`, client cookie, static `web/` |
| `app/realtime/session.py` | One conversation: protocol state machine, response lifecycle, cancel/truncate |
| `app/realtime/events.py` | Server event builders (shapes match `openai` types **and** the SDK's zod schemas) |
| `app/realtime/memory.py` | `ConversationStore`: per-browser SQLite memory with TTL |
| `app/agent/` | `responder.py` (Deep Agent, planning, researcher), `tools.py` (`web_search`, `fetch_page`), `llm.py` (loads `llm_service.py`) |
| `app/pipeline/base.py` | Pluggable engine interfaces (`VADStream`, `STT`, `Responder`, `TTS`) + `TurnContext` |
| `app/pipeline/vad.py` | Hysteresis turn detector; `SileroVAD` (ONNX, CPU) and `EnergyVAD` |
| `app/pipeline/stt.py` | `FasterWhisperSTT` (GPU fp16, hallucination filter) |
| `app/pipeline/tts.py` | `KokoroTTS` (GPU, UI voice-name mapping) |
| `app/pipeline/mock.py` | Mock responder/STT + tone TTS for protocol testing |
| `web/` | Browser UI adapted from `hf-realtime-voice`: rebranded header/About/footer; relative `/v1/realtime` URL support; server-side tool rows from `mcp_call` items; browser-side tools (Serper search, webcam) disabled (`CLIENT_TOOLS` in `main.js`); "Forget our conversation" in Settings |
| `tests/` | Protocol tests (`pytest`) + SDK schema / end-to-end checks |

One conversation at a time: a new connection replaces the active one.

## Tests

```bash
/mnt/cache/deepagent-rtv/venv/bin/python -m pytest tests -q     # protocol (mock engines)
ENGINE_TESTS=1 /mnt/cache/deepagent-rtv/venv/bin/python -m pytest tests/test_speech_engines.py tests/test_tts.py -s
AGENT_TESTS=1 /mnt/cache/deepagent-rtv/venv/bin/python -m pytest tests/test_agent.py tests/test_tools.py -s
# With the server running: drive it with the real browser SDK from Node
node --experimental-websocket tests/sdk_e2e.mjs \
  /mnt/cache/deepagent-rtv/web-node/node_modules/@openai/agents-realtime/dist
```

The SDK silently downgrades any event that fails its zod schema to "generic"
(no audio, no response sequencing, no tool calls), so the tests validate every
emitted event against those schemas, not just the `openai` Python types.

## Progress

- [x] Scaffold + reused frontend
- [x] Realtime GA WebSocket skeleton (mock pipeline)
- [x] VAD (silero)
- [x] STT (faster-whisper)
- [x] TTS (Kokoro)
- [x] DeepAgent (Azure OpenAI + DuckDuckGo), tool rows in chat
- [x] `fetch_page` tool, token streaming, planning + research subagent
- [x] Persistent memory across reconnects, UI rebrand
- [ ] Camera tool (agent asks the browser for a webcam frame)
