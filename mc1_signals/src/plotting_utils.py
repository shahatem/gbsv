from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def save_figure(fig: plt.Figure, file_path: str | Path, dpi: int = 160) -> Path:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    return file_path


def plot_waveforms(
    time_axis: np.ndarray,
    signals: list[np.ndarray],
    labels: list[str],
    title: str,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    for signal, label in zip(signals, labels):
        ax.plot(time_axis[: len(signal)], signal, linewidth=1.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [-]")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    return fig, ax


def plot_spectrum(
    signal: np.ndarray,
    sr: int,
    title: str,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure
    freq = np.fft.rfftfreq(len(signal), d=1 / sr)
    mag = np.abs(np.fft.rfft(signal))
    ax.plot(freq, mag, linewidth=1.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    if label:
        ax.legend()
    return fig, ax


def plot_kernel(kernel: np.ndarray, title: str = "Kernel") -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.stem(np.arange(len(kernel)), kernel, basefmt=" ")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Weight")
    return fig, ax


def plot_metric_curve(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
