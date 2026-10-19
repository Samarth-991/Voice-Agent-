"""TTS: voice mapping (always) and a Kokoro -> Whisper round trip (ENGINE_TESTS=1)."""

import asyncio
import os
import re

import pytest

from app import config
from app.pipeline.tts import DEFAULT_VOICE, VOICE_MAP, resolve_voice


def test_ui_voices_map_to_kokoro_voices():
    assert resolve_voice("Aiden") == "am_michael"
    assert resolve_voice("Vivian") == "bf_emma"
    assert resolve_voice("bm_lewis") == "bm_lewis"  # native Kokoro ids pass through
    assert resolve_voice("alloy") == DEFAULT_VOICE  # unknown -> default
    assert all(re.fullmatch(r"[ab][fm]_[a-z]+", v) for v in VOICE_MAP.values())


@pytest.mark.skipif(os.environ.get("ENGINE_TESTS") != "1",
                    reason="set ENGINE_TESTS=1 to run Kokoro + Whisper on the GPU")
def test_kokoro_speech_round_trips_through_whisper():
    from app.pipeline.stt import FasterWhisperSTT
    from app.pipeline.tts import KokoroTTS

    tts = KokoroTTS(device=config.TTS_DEVICE)
    stt = FasterWhisperSTT(config.STT_MODEL, device=config.STT_DEVICE,
                           compute_type=config.STT_COMPUTE_TYPE, language="en")
    sentence = "The quick brown fox jumps over the lazy dog."

    async def round_trip(voice: str) -> str:
        pcm = b"".join([chunk async for chunk in tts.synthesize(sentence, voice)])
        assert len(pcm) > 2 * 24_000  # over a second of speech
        return await stt.transcribe(pcm, "en")

    for voice in ("Aiden", "Vivian"):  # American and British pipelines
        heard = asyncio.run(round_trip(voice))
        print(f"\n{voice} -> {heard!r}")
        assert "quick brown fox jumps over the lazy dog" in re.sub(r"[^a-z ]", "", heard.lower())
