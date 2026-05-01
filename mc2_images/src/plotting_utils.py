from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style() -> None:
    """Apply a consistent plotting style for all MC2 notebooks."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10


def save_figure(fig: plt.Figure, file_path: str | Path, dpi: int = 160) -> Path:
    """Save a Matplotlib figure and create parent folders when needed."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def show_image(
    image: np.ndarray,
    title: str,
    cmap: str | None = None,
    figsize: tuple[int, int] = (7, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Display one image with axes hidden."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_title(title)
    ax.axis("off")
    return fig, ax


def show_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    cmap: str | None = None,
    columns: int = 3,
    figsize: tuple[int, int] = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """Display multiple images in a compact grid."""
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes_array = np.atleast_1d(axes).ravel()
    for axis, image, title in zip(axes_array, images, titles):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes_array[len(images) :]:
        axis.axis("off")
    return fig, axes_array


def show_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    title: str,
    color: tuple[float, float, float] = (1.0, 0.2, 0.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay a binary mask on an RGB image."""
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = 0.55 * overlay[mask_bool] + 0.45 * np.array(color)
    return show_image(np.clip(overlay, 0.0, 1.0), title)


def plot_histogram_with_thresholds(
    values: np.ndarray,
    thresholds: dict[str, float],
    title: str,
    bins: int = 60,
    figsize: tuple[int, int] = (10, 4),
    x_label: str = "Intensity",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of pixel values and overlay vertical lines for each named threshold."""
    flat = np.asarray(values).ravel()
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(flat, bins=bins, color="tab:blue", alpha=0.65, edgecolor="white")
    palette = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    for index, (label, threshold) in enumerate(thresholds.items()):
        color = palette[index % len(palette)]
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.6, label=f"{label} = {threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pixel count")
    ax.legend(loc="upper right", fontsize=9)
    return fig, ax


def plot_parameter_sweep(
    sweep_df: pd.DataFrame,
    x_column: str,
    y_column: str,
    hue_column: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
    figsize: tuple[int, int] = (8, 4),
    marker: str = "o",
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot of `y_column` vs `x_column`, one line per `hue_column` value."""
    fig, ax = plt.subplots(figsize=figsize)
    hues: Iterable = sorted(sweep_df[hue_column].unique())
    for hue_value in hues:
        subset = sweep_df[sweep_df[hue_column] == hue_value].sort_values(x_column)
        ax.plot(
            subset[x_column],
            subset[y_column],
            marker=marker,
            linewidth=1.2,
            label=f"{hue_column}={hue_value}",
        )
    ax.set_title(title)
    ax.set_xlabel(x_label or x_column)
    ax.set_ylabel(y_label or y_column)
    ax.legend(loc="best", fontsize=9)
    return fig, ax
