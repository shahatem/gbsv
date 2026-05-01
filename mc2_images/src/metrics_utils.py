from __future__ import annotations

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity


def relative_change_percent(value: float, baseline: float) -> float:
    """Compute relative change from a baseline in percent."""
    if abs(baseline) < 1e-12:
        return 0.0
    return float((value - baseline) / baseline * 100.0)


def mean_structural_similarity(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute structural similarity between two RGB or grayscale images."""
    if reference.ndim == 3:
        return float(
            structural_similarity(
                reference,
                estimate,
                channel_axis=2,
                data_range=1.0,
            )
        )
    return float(structural_similarity(reference, estimate, data_range=1.0))


def psnr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB for images in the range [0, 1]."""
    mse = float(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def edge_density(edge_mask: np.ndarray) -> float:
    """Return the proportion of pixels classified as edges."""
    return float(np.mean(edge_mask > 0))


def connected_edge_components(edge_mask: np.ndarray) -> int:
    """Count 8-connected components of an edge map (excluding the background)."""
    binary = (edge_mask > 0).astype(np.uint8)
    component_count, _ = cv.connectedComponents(binary, connectivity=8)
    return int(max(component_count - 1, 0))


def jaccard_index(reference_mask: np.ndarray, estimate_mask: np.ndarray) -> float:
    """Compute intersection-over-union for two binary masks."""
    reference = reference_mask.astype(bool)
    estimate = estimate_mask.astype(bool)
    union = np.logical_or(reference, estimate).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(reference, estimate).sum()
    return float(intersection / union)


def object_count_stability(reference_count: int, estimate_count: int) -> int:
    """Absolute deviation between two object counts (0 = identical count)."""
    return int(abs(int(estimate_count) - int(reference_count)))
