"""Speech-to-text engines."""

import asyncio
import logging
import threading

import numpy as np

from app.pipeline.audio import pcm16_to_float, resample
from app.pipeline.base import SAMPLE_RATE

logger = logging.getLogger("deepagent.stt")

WHISPER_RATE = 16_000
# Whisper invents text ("Thank you.") on noise; drop segments it isn't confident hold speech.
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0


class FasterWhisperSTT:
    def __init__(self, model: str, *, device: str, compute_type: str, language: str = "") -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper %s on %s (%s)", model, device, compute_type)
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        # Fixed language for language-specific models (e.g. distil-large-v3 is English only).
        self._language = language
        self._lock = threading.Lock()
        self._transcribe_sync(bytes(2 * SAMPLE_RATE), "auto")  # warm up CUDA kernels

    async def transcribe(self, pcm: bytes, language: str) -> str:
        return await asyncio.to_thread(self._transcribe_sync, pcm, language)

    def _transcribe_sync(self, pcm: bytes, language: str) -> str:
        audio = resample(pcm16_to_float(pcm), SAMPLE_RATE, WHISPER_RATE)
        lang = self._language or (None if language in ("", "auto") else language)
        with self._lock:
            segments, _ = self._model.transcribe(
                np.ascontiguousarray(audio),
                language=lang,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            kept = [s.text.strip() for s in segments
                    if not (s.no_speech_prob > NO_SPEECH_PROB and s.avg_logprob < MIN_AVG_LOGPROB)]
        return " ".join(t for t in kept if t).strip()
