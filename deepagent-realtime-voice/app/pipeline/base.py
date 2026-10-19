"""Pluggable pipeline interfaces.

Each engine is loaded once at startup and shared by sessions, except VAD,
which is stateful and created per session from ``Pipeline.vad``. Engines must
be safe to call from the asyncio event loop: long blocking work (GPU
inference, network calls) belongs in a thread (``asyncio.to_thread``).
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Literal, Protocol

# Wire format on both directions of the realtime socket: PCM16 LE mono 24 kHz.
SAMPLE_RATE = 24_000


@dataclass
class TurnContext:
    """What a responder needs to produce one assistant reply."""

    instructions: str
    voice: str
    language: str
    # Conversation history as OpenAI Realtime GA item dicts, oldest first.
    items: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TurnDetection:
    """Server VAD settings, from session.audio.input.turn_detection."""

    threshold: float = 0.5
    silence_ms: int = 600
    prefix_ms: int = 300

    @classmethod
    def from_session(cls, cfg: dict) -> "TurnDetection":
        return cls(
            threshold=float(cfg.get("threshold", cls.threshold)),
            silence_ms=int(cfg.get("silence_duration_ms", cls.silence_ms)),
            prefix_ms=int(cfg.get("prefix_padding_ms", cls.prefix_ms)),
        )


VADEvent = tuple[Literal["start", "end"], float]


class VADStream(Protocol):
    """Stateful turn detector for one session's input audio."""

    def process(self, pcm: bytes) -> list[VADEvent]:
        """Feed PCM16 audio; returns ("start"|"end", ms since the stream began).

        "start" already includes the prefix padding; "end" is when speech stopped.
        """
        ...


class STT(Protocol):
    async def transcribe(self, pcm: bytes, language: str) -> str:
        """Transcribe one committed PCM16 utterance (``language`` may be "auto")."""
        ...


@dataclass
class ToolCall:
    """Tool activity reported by a responder: yielded once when the call starts
    (``output is None``) and again with the same ``call_id`` when it finishes."""

    call_id: str
    name: str
    arguments: str  # JSON
    output: str | None = None
    error: str | None = None


class Responder(Protocol):
    """Produces the assistant's reply for a turn (the DeepAgent, or a mock)."""

    def respond(self, turn: TurnContext) -> AsyncIterator[str | ToolCall]:
        """Yield reply text chunks (their concatenation is the spoken reply)
        interleaved with tool activity."""
        ...


class TTS(Protocol):
    """Speaks text as PCM16 LE mono audio at ``SAMPLE_RATE``."""

    def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks for ``text`` (resample inside the engine if needed)."""
        ...


@dataclass
class Pipeline:
    responder: Responder
    tts: TTS
    stt: STT
    vad: Callable[[TurnDetection], VADStream]
