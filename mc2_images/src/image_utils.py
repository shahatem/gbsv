from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image as RGB float data in the range [0, 1]."""
    image_bgr = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image in [0, 1] to grayscale in [0, 1]."""
    return cv.cvtColor((image * 255).astype(np.uint8), cv.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def crop_region(
    image: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Return a rectangular image region defined in pixel coordinates."""
    return image[top : top + height, left : left + width]


def adjust_contrast_brightness(image: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    """Apply linear contrast and brightness adjustment to an RGB image."""
    return np.clip(image * contrast + brightness, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Add zero-mean Gaussian noise with deterministic random seed."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def create_test_edge_image(size: int = 160) -> np.ndarray:
    """Create a simple image with rectangle and circle edges for algorithm checks."""
    test_image = np.zeros((size, size), dtype=np.float32)
    cv.rectangle(test_image, (30, 35), (125, 120), color=0.75, thickness=-1)
    cv.circle(test_image, (95, 80), 30, color=1.0, thickness=-1)
    return test_image


def gradient_magnitude(gray_image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
    """Return the Sobel gradient magnitude of a grayscale image, normalized to [0, 1]."""
    gray_u8 = (np.clip(gray_image, 0.0, 1.0) * 255).astype(np.uint8)
    if blur_kernel and blur_kernel >= 3:
        gray_u8 = cv.GaussianBlur(gray_u8, (blur_kernel, blur_kernel), sigmaX=0)
    gx = cv.Sobel(gray_u8, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray_u8, cv.CV_32F, 0, 1, ksize=3)
    magnitude = cv.magnitude(gx, gy)
    return cv.normalize(magnitude, None, 0.0, 1.0, cv.NORM_MINMAX)


def canny_steps(gray_image: np.ndarray, blur_kernel: int, low_threshold: int, high_threshold: int) -> dict[str, np.ndarray]:
    """Run Canny edge detection and return three traceable intermediate outputs."""
    gray_u8 = (gray_image * 255).astype(np.uint8)
    blurred = cv.GaussianBlur(gray_u8, (blur_kernel, blur_kernel), sigmaX=0)
    gradient_x = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)
    gradient_mag = cv.magnitude(gradient_x, gradient_y)
    edges = cv.Canny(blurred, low_threshold, high_threshold)
    return {
        "blurred": blurred.astype(np.float32) / 255.0,
        "gradient_magnitude": cv.normalize(gradient_mag, None, 0, 1, cv.NORM_MINMAX),
        "edges": edges > 0,
    }


def canny_steps_with_nms(
    gray_image: np.ndarray,
    blur_kernel: int,
    low_threshold: int,
    high_threshold: int,
) -> dict[str, np.ndarray]:
    """Five-step Canny visualization: input, blurred, gradient, NMS, hysteresis edges."""
    gray_u8 = (np.clip(gray_image, 0.0, 1.0) * 255).astype(np.uint8)
    blurred = cv.GaussianBlur(gray_u8, (blur_kernel, blur_kernel), sigmaX=0)
    gx = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)
    magnitude = cv.magnitude(gx, gy)
    nms = _non_maximum_suppression(magnitude, gx, gy)
    edges = cv.Canny(blurred, low_threshold, high_threshold)
    return {
        "input": gray_u8.astype(np.float32) / 255.0,
        "blurred": blurred.astype(np.float32) / 255.0,
        "gradient_magnitude": cv.normalize(magnitude, None, 0.0, 1.0, cv.NORM_MINMAX),
        "non_max_suppressed": cv.normalize(nms, None, 0.0, 1.0, cv.NORM_MINMAX),
        "edges": edges > 0,
    }


def _non_maximum_suppression(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Quantize gradient direction to 4 bins and keep local maxima along that direction.

    Vectorized: for each of the four orientation bins (0, 45, 90, 135 degrees) the
    relevant pair of neighbours is selected via numpy roll, and the centre pixel is
    kept only when it dominates both neighbours.
    """
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    bin_horizontal = (angle < 22.5) | (angle >= 157.5)
    bin_diag_anti = (angle >= 22.5) & (angle < 67.5)
    bin_vertical = (angle >= 67.5) & (angle < 112.5)
    bin_diag_main = (angle >= 112.5) & (angle < 157.5)
    left = np.roll(magnitude, shift=1, axis=1)
    right = np.roll(magnitude, shift=-1, axis=1)
    up = np.roll(magnitude, shift=1, axis=0)
    down = np.roll(magnitude, shift=-1, axis=0)
    up_right = np.roll(np.roll(magnitude, shift=1, axis=0), shift=-1, axis=1)
    down_left = np.roll(np.roll(magnitude, shift=-1, axis=0), shift=1, axis=1)
    up_left = np.roll(np.roll(magnitude, shift=1, axis=0), shift=1, axis=1)
    down_right = np.roll(np.roll(magnitude, shift=-1, axis=0), shift=-1, axis=1)
    keep = np.zeros_like(magnitude, dtype=bool)
    keep |= bin_horizontal & (magnitude >= left) & (magnitude >= right)
    keep |= bin_diag_anti & (magnitude >= up_right) & (magnitude >= down_left)
    keep |= bin_vertical & (magnitude >= up) & (magnitude >= down)
    keep |= bin_diag_main & (magnitude >= up_left) & (magnitude >= down_right)
    nms = np.where(keep, magnitude, 0.0).astype(np.float32)
    nms[0, :] = 0.0
    nms[-1, :] = 0.0
    nms[:, 0] = 0.0
    nms[:, -1] = 0.0
    return nms


def threshold_mask(gray_image: np.ndarray, threshold: float, invert: bool = False) -> np.ndarray:
    """Create a binary segmentation mask from a grayscale threshold."""
    mask = gray_image < threshold if invert else gray_image > threshold
    return mask.astype(bool)


def otsu_threshold(gray_image: np.ndarray) -> float:
    """Return the data-driven Otsu threshold in [0, 1] for a grayscale image."""
    gray_u8 = (np.clip(gray_image, 0.0, 1.0) * 255).astype(np.uint8)
    threshold_u8, _ = cv.threshold(gray_u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return float(threshold_u8) / 255.0


def refine_mask(mask: np.ndarray, kernel_size: int, operation: str = "open_close") -> np.ndarray:
    """Refine a binary mask with morphology while keeping object shapes stable."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    if operation == "open_close":
        opened = cv.morphologyEx(mask_u8, cv.MORPH_OPEN, kernel)
        refined = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
    elif operation == "close":
        refined = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, kernel)
    else:
        refined = cv.morphologyEx(mask_u8, cv.MORPH_OPEN, kernel)
    return refined.astype(bool)


def connected_component_properties(mask: np.ndarray, min_area: int = 120) -> pd.DataFrame:
    """Measure area, perimeter, and bounding boxes for connected components."""
    component_count, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    records = []
    for label in range(1, component_count):
        area = int(stats[label, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        component_mask = labels == label
        contours, _ = cv.findContours(component_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv.arcLength(contour, True) for contour in contours))
        records.append(_component_record(label, stats[label], area, perimeter))
    return pd.DataFrame.from_records(records)


def _component_record(label: int, stat_row: np.ndarray, area: int, perimeter: float) -> dict[str, float]:
    return {
        "component_id": label,
        "area_px": area,
        "perimeter_px": perimeter,
        "x_px": int(stat_row[cv.CC_STAT_LEFT]),
        "y_px": int(stat_row[cv.CC_STAT_TOP]),
        "width_px": int(stat_row[cv.CC_STAT_WIDTH]),
        "height_px": int(stat_row[cv.CC_STAT_HEIGHT]),
    }


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component."""
    component_count, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if component_count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    return labels == largest_label


def skeleton_pixel_count(mask: np.ndarray) -> int:
    """Count pixels in the skeleton of a binary object mask."""
    return int(np.sum(skeletonize(mask.astype(bool))))


def skeleton_mask(mask: np.ndarray) -> np.ndarray:
    """Return the skeleton of a binary mask as a boolean array."""
    return skeletonize(mask.astype(bool))


def parameter_grid(**axes: Iterable) -> pd.DataFrame:
    """Build a tidy DataFrame with one row per Cartesian-product configuration."""
    keys = list(axes.keys())
    values = [list(axes[key]) for key in keys]
    rows = [dict(zip(keys, combo)) for combo in product(*values)]
    return pd.DataFrame(rows)


def save_table(data_frame: pd.DataFrame, file_path: str | Path) -> Path:
    """Write a DataFrame to CSV and create parent folders when needed."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False)
    return output_path
