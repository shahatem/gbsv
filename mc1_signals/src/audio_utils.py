from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def project_root(start: str | Path = ".") -> Path:
    """Return the project root when called from a notebook or module."""
    path = Path(start).resolve()
    candidates = [path, *path.parents]
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the mc1_signals project root.")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_audio_mono(
    file_path: str | Path,
    target_sr: int | None = None,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 audio."""
    offset = 0.0 if start_sec is None else float(start_sec)
    audio, sr = librosa.load(
        str(file_path),
        sr=target_sr,
        mono=True,
        offset=offset,
        duration=duration_sec,
    )
    return audio.astype(np.float32), int(sr)


def peak_normalize(signal: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal.astype(np.float32)
    return (signal / max_abs * peak).astype(np.float32)


def to_time_axis(num_samples: int, sr: int) -> np.ndarray:
    return np.arange(num_samples, dtype=np.float64) / float(sr)


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal.astype(np.float32)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(signal, up=up, down=down).astype(np.float32)


def sample_by_interpolation(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample by changing the sampling grid with linear interpolation.

    This is intentionally simple and does not add a new anti-alias filter, which
    is useful when demonstrating what happens if a signal is sampled below the
    Nyquist rate after a fixed application-driven bandwidth assumption.
    """
    duration = len(signal) / float(orig_sr)
    original_time = np.arange(len(signal), dtype=np.float64) / float(orig_sr)
    target_time = np.arange(0, duration, 1 / float(target_sr), dtype=np.float64)
    return np.interp(target_time, original_time, signal).astype(np.float32)


def save_audio(file_path: str | Path, signal: np.ndarray, sr: int) -> Path:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(file_path, signal, sr)
    return file_path
