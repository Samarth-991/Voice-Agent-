"""Build the configured pipeline (engines are created once and shared)."""

import logging

from app import config
from app.pipeline.base import Pipeline

logger = logging.getLogger("deepagent.pipeline")


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not supported; choose one of {allowed}.")
    return value


def build_pipeline() -> Pipeline:
    from app.pipeline import mock, vad

    vad_cls = {"silero": vad.SileroVAD, "energy": vad.EnergyVAD}[
        _choice("VAD", config.VAD_ENGINE, ("silero", "energy"))]

    if _choice("RESPONDER", config.RESPONDER, ("deepagent", "mock")) == "deepagent":
        from app.agent.responder import DeepAgentResponder

        responder = DeepAgentResponder()
    else:
        print("Using mock responder")
        responder = mock.MockResponder()

    if _choice("STT", config.STT_ENGINE, ("faster-whisper", "mock")) == "faster-whisper":
        from app.pipeline.stt import FasterWhisperSTT

        stt = FasterWhisperSTT(config.STT_MODEL, device=config.STT_DEVICE,
                               compute_type=config.STT_COMPUTE_TYPE, language=config.STT_LANGUAGE)
    else:
        stt = mock.MockSTT()

    if _choice("TTS", config.TTS_ENGINE, ("kokoro", "mock")) == "kokoro":
        from app.pipeline.tts import KokoroTTS

        tts = KokoroTTS(device=config.TTS_DEVICE, speed=config.TTS_SPEED)
    else:
        tts = mock.ToneTTS()

    logger.info("Pipeline: vad=%s stt=%s responder=%s tts=%s", config.VAD_ENGINE,
                config.STT_ENGINE, config.RESPONDER, config.TTS_ENGINE)
    return Pipeline(responder=responder, tts=tts, stt=stt, vad=vad_cls)
