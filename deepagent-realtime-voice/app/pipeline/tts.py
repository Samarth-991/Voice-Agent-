"""Text-to-speech engines."""

import asyncio
import logging
import threading
from typing import AsyncIterator

import numpy as np

from app.pipeline.audio import float_to_pcm16

logger = logging.getLogger("deepagent.tts")

KOKORO_REPO = "hexgrad/Kokoro-82M"
# The UI's voice list comes from the original Qwen3-TTS backend; map each name
# to a similar Kokoro English voice (af/am = American, bf/bm = British).
VOICE_MAP = {
    "Aiden": "am_michael",
    "Ryan": "am_fenrir",
    "Dylan": "am_puck",
    "Eric": "am_eric",
    "Ono_Anna": "af_bella",
    "Serena": "af_heart",
    "Sohee": "af_nicole",
    "Uncle_Fu": "bm_george",
    "Vivian": "bf_emma",
}
DEFAULT_VOICE = "af_heart"


def resolve_voice(name: str) -> str:
    """UI voice name -> Kokoro voice id; native Kokoro ids (e.g. "bf_emma") pass through."""
    if name in VOICE_MAP:
        return VOICE_MAP[name]
    if len(name) > 3 and name[0] in "ab" and name[1] in "fm" and name[2] == "_":
        return name
    return DEFAULT_VOICE


class KokoroTTS:
    """Kokoro-82M on the GPU; outputs 24 kHz natively (the wire rate), one chunk per sentence."""

    def __init__(self, *, device: str, speed: float = 1.0) -> None:
        import torch
        from kokoro import KModel, KPipeline

        logger.info("Loading Kokoro-82M on %s", device)
        self._torch = torch
        model = KModel(repo_id=KOKORO_REPO).to(device).eval()
        # One G2P pipeline per accent, sharing the model ("a" American, "b" British).
        self._pipelines = {lang: KPipeline(lang_code=lang, repo_id=KOKORO_REPO, model=model)
                           for lang in ("a", "b")}
        self._speed = speed
        self._lock = threading.Lock()
        # First use of each voice downloads it and compiles kernels (~1 s); do it now.
        for voice in sorted({DEFAULT_VOICE, *VOICE_MAP.values()}):
            self._synthesize_sync("Ready.", voice)

    async def synthesize(self, text: str, voice: str) -> AsyncIterator[bytes]:
        pcm = await asyncio.to_thread(self._synthesize_sync, text, resolve_voice(voice))
        if pcm:
            yield pcm

    def _synthesize_sync(self, text: str, voice: str) -> bytes:
        pipeline = self._pipelines[voice[0]]
        with self._lock, self._torch.inference_mode():
            chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=self._speed)
                      if audio is not None]
        if not chunks:
            return b""
        audio = np.concatenate([np.asarray(c.cpu() if hasattr(c, "cpu") else c, dtype=np.float32)
                                for c in chunks])
        return float_to_pcm16(audio)
