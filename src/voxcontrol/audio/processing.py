"""Audio I/O, preprocessing and controlled degradations.

The original signal is never modified in place; every function returns a new array.
WAV is read with the standard library; FLAC needs the optional ``soundfile`` package.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16000


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() == ".flac":
        try:
            import soundfile as sf
        except ImportError as e:
            raise RuntimeError("FLAC files need the optional 'soundfile' package (pip install soundfile)") from e
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x.T.astype(np.float32), int(sr)
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        x = (x - 128.0) / 128.0
    else:
        x /= float(2 ** (8 * width - 1))
    return x.reshape(-1, ch).T, sr


def save_wav(path: str | Path, x: np.ndarray, sr: int = TARGET_SR) -> None:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=0) if x.ndim == 2 else x


def resample(x: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    g = np.gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 0 else x


def trim_silence(x: np.ndarray, sr: int = TARGET_SR, threshold_db: float = -40.0, frame_ms: int = 20) -> np.ndarray:
    f = max(1, int(sr * frame_ms / 1000))
    n = len(x) // f
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[:n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (np.max(rms) + 1e-12))
    active = np.flatnonzero(db > threshold_db)
    if active.size == 0:
        return x
    return x[active[0] * f:(active[-1] + 1) * f]


def preprocess(x: np.ndarray, sr: int, trim: bool = True) -> np.ndarray:
    y = resample(to_mono(x), sr)
    y = normalize_peak(y)
    return trim_silence(y) if trim else y


# ------------------------------------------------------------- degradations
def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator, kind: str = "white") -> np.ndarray:
    """Additive noise scaled to the requested signal-to-noise ratio (dB)."""
    x = np.asarray(x, dtype=np.float64)
    n = rng.standard_normal(len(x))
    if kind == "pink":
        spec = np.fft.rfft(n)
        f = np.arange(len(spec)); f[0] = 1
        n = np.fft.irfft(spec / np.sqrt(f), len(x))
    p_sig = np.mean(x ** 2) + 1e-12
    p_noise = np.mean(n ** 2) + 1e-12
    n *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return (x + n).astype(np.float32)


def reverb(x: np.ndarray, rt60: float = 0.4, sr: int = TARGET_SR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Convolve with a synthetic exponentially decaying noise impulse response."""
    rng = rng or np.random.default_rng(0)
    L = int(rt60 * sr)
    t = np.arange(L) / sr
    ir = rng.standard_normal(L) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    y = np.convolve(x, ir / np.max(np.abs(ir)))[:len(x)]
    return normalize_peak(y.astype(np.float32), float(np.max(np.abs(x))) or 0.9)


def clip(x: np.ndarray, level: float = 0.3) -> np.ndarray:
    return np.clip(x, -level, level).astype(np.float32)


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Time stretch by resampling (changes pitch proportionally, as in speed perturbation)."""
    idx = np.arange(0, len(x), factor)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def dropout(x: np.ndarray, fraction: float, rng: np.random.Generator, segment_ms: int = 50,
            sr: int = TARGET_SR) -> np.ndarray:
    """Zero random segments covering roughly ``fraction`` of the signal (packet loss)."""
    y = np.array(x, dtype=np.float32, copy=True)
    seg = max(1, int(sr * segment_ms / 1000))
    n = len(y) // seg
    for i in rng.choice(n, size=int(round(fraction * n)), replace=False) if n else []:
        y[i * seg:(i + 1) * seg] = 0.0
    return y
