"""Text helpers shared by pipeline stages."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceBuffer:
    """Accumulates streamed text and releases whole sentences for TTS."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> list[str]:
        self._pending += text
        parts = _SENTENCE_END.split(self._pending)
        self._pending = parts.pop()
        return [p.strip() for p in parts if p.strip()]

    def flush(self) -> list[str]:
        rest, self._pending = self._pending.strip(), ""
        return [rest] if rest else []
