"""Mock engines for exercising the realtime protocol without real models.

``MockResponder`` returns canned text; ``ToneTTS`` renders each word as a
short soft tone so playback, the orb, and chat bubbles can be checked by ear.
"""

import asyncio
import zlib
from typing import AsyncIterator

import numpy as np

from app.pipeline.base import SAMPLE_RATE, TurnContext

GREETING_REPLY = (
    "Hi! I'm the DeepAgent voice skeleton. "
    "The realtime connection and audio playback are working."
)


def _last_user_message(items: list[dict]) -> tuple[str | None, bool]:
    """(text, spoken) of the newest user message: typed text or an audio transcript."""
    for item in reversed(items):
        if item.get("type") != "message" or item.get("role") != "user":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "input_audio":
                return part.get("transcript"), True
            if part.get("text"):
                return part["text"], False
        return None, False
    return None, False


class MockResponder:
    async def respond(self, turn: TurnContext) -> AsyncIterator[str]:
        text, spoken = _last_user_message(turn.items)
        has_replied = any(
            i.get("type") == "message" and i.get("role") == "assistant" for i in turn.items
        )
        if not spoken and not has_replied:
            yield GREETING_REPLY  # reply to the hidden startup-greeting prompt
        elif text:
            yield f"You said: {text}"
        else:
            yield "I heard you, but couldn't make out any words."


class MockSTT:
    """Returns a fixed transcript for every utterance."""

    text = "hello from the mock transcriber"

    async def transcribe(self, pcm: bytes, language: str) -> str:
        return self.text


class ToneTTS:
    """One soft tone per word, pitch derived from the word, pauses at sentence ends."""

    amplitude = 0.12

    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        for word in text.split():
            yield self._tone(word)
            yield self._silence(0.05)
            await asyncio.sleep(0)
        yield self._silence(0.2)

    def _tone(self, word: str) -> bytes:
        freq = 260 + (zlib.crc32(word.encode()) % 6) * 40
        seconds = 0.09 + 0.025 * min(len(word), 8)
        t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
        wave = np.sin(2 * np.pi * freq * t)
        # 10 ms raised-cosine edges so word boundaries don't click.
        ramp = int(SAMPLE_RATE * 0.01)
        env = np.ones_like(wave)
        env[:ramp] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
        env[-ramp:] = env[:ramp][::-1]
        return (wave * env * self.amplitude * 32767).astype("<i2").tobytes()

    @staticmethod
    def _silence(seconds: float) -> bytes:
        return bytes(2 * int(SAMPLE_RATE * seconds))
