"""Turn detection: a shared hysteresis state machine over per-frame speech
probabilities, fed by silero-vad (real) or signal energy (tests)."""

import numpy as np

from app.pipeline.audio import pcm16_to_float, resample
from app.pipeline.base import SAMPLE_RATE, TurnDetection, VADEvent

# Speech must persist this long before a turn starts, so clicks, coughs, or
# speaker echo don't barge in on the assistant.
MIN_SPEECH_MS = 200
# A gap this long while still confirming a start discards the candidate.
MAX_CANDIDATE_GAP_MS = 150


class FrameVAD:
    """Turns per-frame speech probabilities into start/end events."""

    rate: int = SAMPLE_RATE
    frame: int = 480

    def __init__(self, td: TurnDetection) -> None:
        self.td = td
        self.frame_ms = 1000 * self.frame / self.rate
        self._neg_threshold = max(td.threshold - 0.15, 0.01)
        self._pending = np.zeros(0, dtype=np.float32)
        self._t = 0.0
        self._in_speech = False
        self._last_speech_end = 0.0
        self._silence = 0.0
        self._reset_candidate()

    def speech_prob(self, frame: np.ndarray) -> float:
        raise NotImplementedError

    def process(self, pcm: bytes) -> list[VADEvent]:
        audio = resample(pcm16_to_float(pcm), SAMPLE_RATE, self.rate)
        self._pending = np.concatenate([self._pending, audio])
        events: list[VADEvent] = []
        while self._pending.size >= self.frame:
            frame, self._pending = self._pending[: self.frame], self._pending[self.frame:]
            event = self._step(self.speech_prob(frame))
            if event:
                events.append(event)
        return events

    def _reset_candidate(self) -> None:
        self._cand_start: float | None = None
        self._cand_speech = 0.0
        self._cand_gap = 0.0

    def _step(self, p: float) -> VADEvent | None:
        start, self._t = self._t, self._t + self.frame_ms
        if not self._in_speech:
            if p >= self.td.threshold:
                if self._cand_start is None:
                    self._cand_start = start
                self._cand_speech += self.frame_ms
                self._cand_gap = 0.0
                if self._cand_speech >= MIN_SPEECH_MS:
                    self._in_speech, self._silence = True, 0.0
                    self._last_speech_end = self._t
                    begin = max(0.0, self._cand_start - self.td.prefix_ms)
                    self._reset_candidate()
                    return ("start", begin)
            elif self._cand_start is not None:
                self._cand_gap += self.frame_ms
                if self._cand_gap > MAX_CANDIDATE_GAP_MS:
                    self._reset_candidate()
            return None
        if p >= self._neg_threshold:
            self._silence, self._last_speech_end = 0.0, self._t
            return None
        self._silence += self.frame_ms
        if self._silence >= self.td.silence_ms:
            self._in_speech = False
            return ("end", self._last_speech_end)
        return None


class SileroVAD(FrameVAD):
    """silero-vad (ONNX, CPU) at 16 kHz; one model instance per session (it is stateful)."""

    rate = 16_000
    frame = 512

    def __init__(self, td: TurnDetection) -> None:
        from silero_vad import load_silero_vad

        super().__init__(td)
        self._model = load_silero_vad(onnx=True)

    def speech_prob(self, frame: np.ndarray) -> float:
        import torch

        return float(self._model(torch.from_numpy(frame), self.rate).item())


class EnergyVAD(FrameVAD):
    """Loudness-only detector, for tests and model-free debugging."""

    rms_threshold = 0.02

    def speech_prob(self, frame: np.ndarray) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > self.rms_threshold else 0.0
