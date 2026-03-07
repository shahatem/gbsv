from __future__ import annotations

import numpy as np
from scipy.signal import butter, correlate, correlation_lags, find_peaks, resample_poly, sosfiltfilt


def lowpass_filter(signal: np.ndarray, sr: int, cutoff_hz: float, order: int = 8) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def practical_bandlimit_then_resample(
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int,
    bandwidth_hz: float | None = None,
    transition_ratio: float = 0.9,
) -> np.ndarray:
    """
    Create a practical anti-aliased downsampled version of the signal.
    If bandwidth_hz is provided, filter to that application-driven band first.
    """
    filtered = signal.astype(np.float32)
    if bandwidth_hz is not None:
        safe_cutoff = min(bandwidth_hz, 0.5 * target_sr * transition_ratio)
        filtered = lowpass_filter(filtered, sr=orig_sr, cutoff_hz=safe_cutoff)
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(filtered, up=up, down=down).astype(np.float32)


def moving_average_kernel(size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("Kernel size must be >= 1.")
    return np.ones(size, dtype=np.float32) / float(size)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    center = size // 2
    x = np.arange(size, dtype=np.float64) - center
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def manual_convolve_1d(signal: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    if mode not in {"same", "full", "valid"}:
        raise ValueError("mode must be one of: same, full, valid")
    signal = np.asarray(signal, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    full = np.zeros(len(signal) + len(kernel) - 1, dtype=np.float32)
    flipped_kernel = kernel[::-1]

    for idx in range(len(full)):
        acc = 0.0
        for kernel_idx in range(len(flipped_kernel)):
            signal_idx = idx - kernel_idx
            if 0 <= signal_idx < len(signal):
                acc += signal[signal_idx] * flipped_kernel[kernel_idx]
        full[idx] = acc

    if mode == "full":
        return full
    if mode == "valid":
        start = len(kernel) - 1
        end = start + len(signal) - len(kernel) + 1
        return full[start:end]
    start = (len(kernel) - 1) // 2
    end = start + len(signal)
    return full[start:end]


def amplitude_envelope(
    signal: np.ndarray,
    sr: int,
    frame_ms: float = 50.0,
    hop_ms: float = 10.0,
    smoothing_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    values = []
    times = []
    for start in range(0, len(signal) - frame + 1, hop):
        chunk = signal[start : start + frame]
        values.append(np.sqrt(np.mean(chunk**2)))
        times.append((start + frame / 2) / sr)
    envelope = np.asarray(values, dtype=np.float32)
    time_axis = np.asarray(times, dtype=np.float32)

    if smoothing_ms is not None and smoothing_ms > hop_ms:
        width = max(3, int(round(smoothing_ms / hop_ms)))
        if width % 2 == 0:
            width += 1
        kernel = moving_average_kernel(width)
        envelope = np.convolve(envelope, kernel, mode="same").astype(np.float32)
    return envelope, time_axis


def zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    return ((signal - np.mean(signal)) / (np.std(signal) + 1e-12)).astype(np.float32)


def autocorrelation(signal: np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    centered = signal - np.mean(signal)
    corr = correlate(centered, centered, mode="full")
    lags = correlation_lags(len(centered), len(centered), mode="full")
    non_negative = lags >= 0
    corr = corr[non_negative].astype(np.float64)
    lags = lags[non_negative]
    if normalize and corr[0] != 0:
        corr = corr / corr[0]
    return corr.astype(np.float32), lags.astype(np.int32)


def find_top_autocorr_peaks(
    corr: np.ndarray,
    lags: np.ndarray,
    min_lag: int,
    distance: int,
    prominence: float,
    top_k: int = 5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    valid = lags >= min_lag
    peaks, properties = find_peaks(corr[valid], distance=distance, prominence=prominence)
    peak_values = corr[valid][peaks]
    order = np.argsort(peak_values)[::-1][:top_k]
    peak_indices = peaks[order]
    filtered_lags = lags[valid][peak_indices]
    filtered_values = corr[valid][peak_indices]
    result_properties = {
        key: np.asarray(value)[order] for key, value in properties.items()
    }
    return np.asarray(filtered_lags), {
        "peak_values": np.asarray(filtered_values),
        **result_properties,
    }


def normalized_cross_correlation(template: np.ndarray, signal: np.ndarray) -> np.ndarray:
    template = np.asarray(template, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32)
    template = template - np.mean(template)
    template_norm = np.linalg.norm(template) + 1e-12

    corr = np.zeros(len(signal) - len(template) + 1, dtype=np.float32)
    for idx in range(len(corr)):
        window = signal[idx : idx + len(template)]
        window = window - np.mean(window)
        corr[idx] = float(np.dot(template, window) / (template_norm * (np.linalg.norm(window) + 1e-12)))
    return corr


def local_maxima(values: np.ndarray, distance: int = 1, prominence: float = 0.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return find_peaks(values, distance=distance, prominence=prominence)


def wiener_deconvolution_1d(observed: np.ndarray, kernel: np.ndarray, lam: float) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    n = len(observed)
    kernel_padded = np.zeros(n, dtype=np.float32)
    kernel_padded[: len(kernel)] = kernel
    kernel_padded = np.roll(kernel_padded, -(len(kernel) // 2))
    observed_fft = np.fft.rfft(observed)
    kernel_fft = np.fft.rfft(kernel_padded)
    estimate_fft = np.conj(kernel_fft) * observed_fft / (np.abs(kernel_fft) ** 2 + lam)
    estimate = np.fft.irfft(estimate_fft, n=n)
    return estimate.astype(np.float32)
