"""PCM conversion and resampling helpers."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
