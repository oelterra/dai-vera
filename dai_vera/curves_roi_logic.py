from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ROIContourResult:
    """Returned by detect_roi_contour on success."""
    filled_coords: np.ndarray  # (N, 2) row/col in full image space
    boundary_coords: np.ndarray  # (M, 2) row/col in full image space
    long_axis: float
    short_axis: float
    area: int


@dataclass
class CurveData:
    """Time-intensity curve plus optional fitted overlay."""
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    fitted_time: Optional[list[float]] = None
    fitted_curve: Optional[list[float]] = None

    # selected time-window (data coords, not indices)
    range_start: float = 0.0
    range_end: float = 0.0

    def clear(self) -> None:
        self.times.clear()
        self.values.clear()
        self.fitted_time = None
        self.fitted_curve = None

    def undo_last(self) -> None:
        """Remove the last manually added point."""
        if self.times:
            self.times.pop()
            self.values.pop()


# ---------------------------------------------------------------------------
# ROI / contour detection
# ---------------------------------------------------------------------------

def detect_roi_contour(
        image: np.ndarray,
        x: int,
        y: int,
        window: int,
) -> Optional[ROIContourResult]:
    """
    Detect the boundary and filled region of the brightest blob near (x, y).

    Parameters
    ----------
    image  : 2-D float array (single slice, single time-point)
    x, y   : click coordinates in image-pixel space (row, col)
    window : side-length of the search block in pixels

    Returns
    -------
    ROIContourResult or None if no usable region is found.
    """
    half = window // 2
    sx = max(0, x - half)
    sy = max(0, y - half)
    ex = min(image.shape[0], x + half)
    ey = min(image.shape[1], y + half)

    block = image[sx:ex, sy:ey]
    if block.size == 0 or block.max() == block.min():
        return None

    # threshold on full image (matches MATLAB graythresh behaviour)
    level = threshold_otsu(image)
    block_binary = block > level

    labeled = label(block_binary)
    if labeled.max() == 0:
        return None

    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    # filled mask of largest region
    filled_mask = np.zeros_like(block_binary)
    filled_mask[largest.coords[:, 0], largest.coords[:, 1]] = True

    # boundary = filled minus eroded (matches MATLAB bwperim)
    eroded = binary_erosion(filled_mask)
    boundary = filled_mask & ~eroded

    # translate local coords back to full-image space
    filled_coords = largest.coords.copy()
    filled_coords[:, 0] += sx
    filled_coords[:, 1] += sy

    bY, bX = np.where(boundary)
    boundary_coords = np.column_stack((bY + sx, bX + sy))

    props = regionprops(filled_mask.astype(int))[0]

    return ROIContourResult(
        filled_coords=filled_coords,
        boundary_coords=boundary_coords,
        long_axis=props.major_axis_length,
        short_axis=props.minor_axis_length,
        area=largest.area,
    )


# Curve sampling


def sample_curve_from_volume(
        pixels: np.ndarray,
        filled_coords: np.ndarray,
        z_idx: int,
        times: Optional[list[float]] = None,
        height_positive: bool = False,
) -> CurveData:
    """
    Build a time-intensity curve by averaging pixel values inside a ROI mask
    across all time-points of a 4-D volume.

    Parameters
    ----------
    pixels        : (T, Z, H, W) float array
    filled_coords : (N, 2) array of (row, col) indices into the H×W slice
    z_idx         : which Z slice to use
    times         : optional list of real acquisition times (seconds).
                    Falls back to integer indices if not provided.
    height_positive : if True take abs() of mean values (matches MATLAB option)

    Returns
    -------
    CurveData with .times and .values populated.
    """
    T, Z, H, W = pixels.shape

    image_shape = (H, W)
    mask = np.zeros(image_shape, dtype=bool)
    rows = np.clip(filled_coords[:, 0], 0, H - 1)
    cols = np.clip(filled_coords[:, 1], 0, W - 1)
    mask[rows, cols] = True

    if not mask.any():
        return CurveData()

    # resolve time axis
    if times and len(times) == T:
        # t_arr = np.array(times, dtype=float)
        # convert ms → s when values suggest milliseconds
        # if t_arr.max() > 10000:
        def dicom_time_to_seconds(t):
            hh = int(t // 10000)
            mm = int((t % 10000) // 100)
            ss = t % 100
            return hh * 3600 + mm * 60 + ss


        print("---- TIME DEBUG !!!!!!!!!!")
        print("Raw times:", times[:10])
        print("Min:", np.min(t_arr), "Max:", np.max(t_arr))
        print("Diffs:", np.diff(t_arr)[:5])
        print("--------------------")

        t_arr = np.array([dicom_time_to_seconds(t) for t in t_arr])
        t_arr = t_arr - t_arr[0]


    else:
        t_arr = np.arange(T, dtype=float)
        print("---- TIME DEBUG ----")
        print("Raw times:", times[:10])
        print("Min:", np.min(t_arr), "Max:", np.max(t_arr))
        print("Diffs:", np.diff(t_arr)[:5])
        print("--------------------")

    z_clamped = min(max(0, z_idx), Z - 1)
    values = np.array(
        [float(np.mean(pixels[t, z_clamped][mask])) for t in range(T)]
    )

    if height_positive:
        values = np.abs(values)

    return CurveData(
        times=t_arr.tolist(),
        values=values.tolist(),
        range_start=float(t_arr[0]),
        range_end=float(t_arr[-1]),
    )


# Image windowing

def window_to_uint8(
        img: np.ndarray,
        length: float,
        width: float,
) -> np.ndarray:
    """
    Apply a windowing transform and return a uint8 image for display.

    Parameters
    ----------
    img    : 2-D float array
    length : 0–1 slider value controlling window centre offset
    width  : 0–1 slider value controlling window width scale
    """
    lo = np.percentile(img, 1)
    hi = np.percentile(img, 99)
    if hi <= lo:
        hi = lo + 1.0

    w_scale = 0.25 + width * 1.75
    center = (lo + hi) / 2.0 + (length - 0.5) * (hi - lo) * 0.5
    span = (hi - lo) * w_scale
    out = np.clip((img - (center - span / 2)) / span, 0, 1)
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_roi_to_json(roi_dict: dict, path: Optional[str] = None) -> str:
    """
    Serialise a ROI state dict to JSON.

    Parameters
    ----------
    roi_dict : dict (may contain numpy arrays — they will be converted)
    path     : target file path; defaults to ~/pre_lesion_roi.json

    Returns
    -------
    Absolute path of the written file.
    """
    if path is None:
        path = os.path.join(os.path.expanduser("~"), "pre_lesion_roi.json")

    safe = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in roi_dict.items()
    }
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Test / synthetic data
# ---------------------------------------------------------------------------

def make_test_volume(
        T: int = 24,
        Z: int = 10,
        H: int = 256,
        W: int = 256,
) -> dict:
    """
    Generate a synthetic CTP volume with a gamma-variate bolus planted at a
    known location so the full pipeline can be exercised without real DICOM data.

    Returns
    -------
    dict with keys: pixels (T,Z,H,W), times, zs, shape
    """
    t = np.arange(T, dtype=float)
    t_shift = t - 4
    t_shift = np.maximum(t_shift, 0)

    bolus = 1.0 * (t_shift ** 2.5) * np.exp(-t_shift / 1.5)
    bolus = bolus / bolus.max() * 300  # 300 HU peak

    pixels = np.random.normal(-50, 20, (T, Z, H, W)).astype(np.float32)
    for t_i in range(T):
        pixels[t_i, :, 120:136, 120:136] += float(bolus[t_i])

    return {
        "pixels": pixels,
        "times": list(np.arange(T) * 2.0),  # 2 s spacing
        "zs": list(range(Z)),
        "shape": (T, Z, H, W),
    }