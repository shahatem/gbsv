from __future__ import annotations

import numpy as np


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def normalized_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = np.sqrt(np.mean(reference**2)) + 1e-12
    return float(rmse(reference, estimate) / denom)


def spectral_cosine_similarity(
    reference: np.ndarray,
    estimate: np.ndarray,
    eps: float = 1e-12,
) -> float:
    ref_mag = np.abs(np.fft.rfft(reference))
    est_mag = np.abs(np.fft.rfft(estimate))
    denom = np.linalg.norm(ref_mag) * np.linalg.norm(est_mag) + eps
    return float(np.dot(ref_mag, est_mag) / denom)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise = reference - estimate
    signal_power = np.mean(reference**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    return float(10 * np.log10(signal_power / noise_power))


def peak_to_sidelobe_ratio(main_peak: float, sidelobe_peak: float, eps: float = 1e-12) -> float:
    return float(20 * np.log10((main_peak + eps) / (sidelobe_peak + eps)))
