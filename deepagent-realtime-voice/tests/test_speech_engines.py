"""Real-engine check: recorded speech -> silero VAD -> faster-whisper over /v1/realtime.

Opt-in (loads Whisper on the GPU): ENGINE_TESTS=1 pytest tests/test_speech_engines.py -s
Uses the JFK sample from whisper.cpp, cached under $DRA_CACHE/test-audio.
"""

import os
import re
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config
from app.pipeline.audio import float_to_pcm16, resample
from app.pipeline.base import Pipeline
from app.pipeline.mock import MockResponder, ToneTTS
from app.server import app
from tests.test_realtime_protocol import SDK_SESSION_UPDATE, Conn, assert_valid, silence, stream_audio

pytestmark = pytest.mark.skipif(os.environ.get("ENGINE_TESTS") != "1",
                                reason="set ENGINE_TESTS=1 to run real VAD/STT engines")

JFK_URL = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
JFK_PHRASE = "ask not what your country can do for you"


def jfk_pcm24k() -> bytes:
    path = Path(os.environ.get("DRA_CACHE", "/mnt/cache/deepagent-rtv")) / "test-audio/jfk.wav"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(JFK_URL, path)
    with wave.open(str(path)) as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return float_to_pcm16(resample(audio, rate, 24_000))


def normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower())


def test_recorded_speech_is_segmented_and_transcribed():
    from app.pipeline.stt import FasterWhisperSTT
    from app.pipeline.vad import SileroVAD

    stt = FasterWhisperSTT(config.STT_MODEL, device=config.STT_DEVICE,
                           compute_type=config.STT_COMPUTE_TYPE, language=config.STT_LANGUAGE)
    with TestClient(app) as client:
        client.app.state.pipeline = Pipeline(responder=MockResponder(), tts=ToneTTS(),
                                             stt=stt, vad=SileroVAD)
        with client.websocket_connect("/v1/realtime") as ws:
            conn = Conn(ws)
            conn.until("session.created")
            conn.send(SDK_SESSION_UPDATE)
            conn.until("session.updated")

            started = time.monotonic()
            stream_audio(conn, silence(0.5) + jfk_pcm24k() + silence(2.0))
            # Appends are processed in order, so every VAD event precedes this sentinel's error.
            conn.send({"type": "bogus.sentinel"})
            turns = sum(e["type"] == "input_audio_buffer.speech_stopped"
                        for e in conn.until("error"))
            transcripts = [e["transcript"] for e in conn.events
                           if e["type"] == "conversation.item.input_audio_transcription.completed"]
            while len(transcripts) < turns:
                event = conn.recv()
                if event["type"] == "conversation.item.input_audio_transcription.completed":
                    transcripts.append(event["transcript"])
            elapsed = time.monotonic() - started

    print(f"\n{turns} turn(s) in {elapsed:.1f}s: {transcripts}")
    assert turns >= 1
    assert JFK_PHRASE in normalize(" ".join(transcripts))
    assert_valid(conn.events)
