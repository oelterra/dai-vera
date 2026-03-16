"""
--------------
Python translation of the MATLAB functions:
    getContour, getContourIndices, getROIOverlayed,
    getLongAndShortAxis, getWindowPerimeter, getCloseContour,
    getAllPixelsInClosedContour, getROI

No GUI imports. All functions operate on numpy arrays.
Coordinate convention matches MATLAB: X = row, Y = col.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
from scipy.spatial.distance import pdist


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ContourResult:
    """
    Mirrors the outputs of MATLAB getContour.

    boundary_rows / boundary_cols  — perimeter pixels in full-image space
                                     (used for overlay at 1500 HU)
    filled_rows / filled_cols      — all pixels inside the contour
                                     (used for HU sampling)
    long_axis   : major axis length in pixels
    short_axis  : minor axis length in pixels
    radius_cm   : (long + short) / 2 * avg_pixel_spacing_mm / 10
    area_cm2    : num_pixels * avg_pixel_spacing_mm^2 / 100
    """
    boundary_rows: np.ndarray
    boundary_cols: np.ndarray
    filled_rows:   np.ndarray
    filled_cols:   np.ndarray
    long_axis:     float
    short_axis:    float
    radius_cm:     float
    area_cm2:      float


@dataclass
class ROIObject:
    """
    Python equivalent of MATLAB roiObject / getROI struct.
    Fields match the MATLAB struct exactly.
    """
    study_name:          str
    num_time_points:     int
    num_slices:          int

    # coordinates
    x: int
    y: int
    z: int
    t: int

    # curves
    curve:               list[float] = field(default_factory=list)
    time_points:         list[float] = field(default_factory=list)
    fitted_curve:        list[float] = field(default_factory=list)
    fitted_time_points:  list[float] = field(default_factory=list)
    auc_fitted:          float = 0.0

    # boundary (for overlay)
    roi_x_boundary:      list[int] = field(default_factory=list)   # rows
    roi_y_boundary:      list[int] = field(default_factory=list)   # cols

    # radii / flow (filled by downstream Bernoulli step)
    rs1: float = 0.0
    rs2: float = 0.0
    rs3: float = 0.0
    entry_q: float = 0.0
    exit_q:  float = 0.0

    extra_points:         list = field(default_factory=list)
    number_of_stenosis:   int  = 0


# ---------------------------------------------------------------------------
# Search-window helpers
# ---------------------------------------------------------------------------

def get_search_window(x: int, y: int, window_size: int) -> tuple[int, int]:
    """
    Return the top-left corner (sx, sy) of a square search window
    centred on (x, y).  Matches MATLAB getSearchWindow.
    """
    half = window_size // 2
    sx = x - half
    sy = y - half
    return sx, sy


def get_window_perimeter(sx: int, sy: int,
                          width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the perimeter pixel coordinates of a rectangle.
    Matches MATLAB getWindowPerimeter.
    Returns (rows, cols) as 1-D arrays.
    """
    rows, cols = [], []
    for i in range(sx, sx + height + 1):
        for j in range(sy, sy + width + 1):
            if i == sx or i == sx + height or j == sy or j == sy + width:
                rows.append(i)
                cols.append(j)
    return np.array(rows), np.array(cols)


# ---------------------------------------------------------------------------
# Axis measurement  - getLongAndShortAxis
# ---------------------------------------------------------------------------

def get_long_and_short_axis(binary_image: np.ndarray) -> tuple[float, float]:
    """
    Return (long_axis, short_axis) in pixels.

    MATLAB uses regionprops MajorAxisLength / MinorAxisLength plus
    bwferet for the absolute max/min Feret diameters.
    We use skimage regionprops which gives the same ellipse-fit axes.
    The Feret values are only used internally in MATLAB for maxX/minX
    which are not consumed by the Python pipeline, so we skip them.
    """
    labeled = label(binary_image)
    if labeled.max() == 0:
        return 0.0, 0.0
    props = regionprops(labeled)[0]
    return props.major_axis_length, props.minor_axis_length

# equivalent of bwferet in matlab bwferet(binaryImage, 'Max/MinFeretProperties')
def bwferet(binary_image: np.ndarray) -> tuple[float, float]:
    """
    Python equivalent of MATLAB bwferet(binaryImage,'Max/MinFeretProperties').

    Returns:
        max_feret : longest distance between any two points on the object
        min_feret : shortest distance between any two points on the object
    """
    # find contours
    contours = find_contours(binary_image, 0.5)
    if len(contours) == 0:
        return 0.0, 0.0
    contour = contours[0]  # assume single largest object

    # compute pairwise distances
    distances = pdist(contour)
    if distances.size == 0:
        return 0.0, 0.0

    max_feret = distances.max()
    min_feret = distances.min()
    return max_feret, min_feret

# ---------------------------------------------------------------------------
# Core contour detection (getContourIndices)
# ---------------------------------------------------------------------------

def get_contour_indices(
    sx: int,
    sy: int,
    search_window_size: int,
    image_2d: np.ndarray,
    x: int,
    y: int,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Python translation of MATLAB getContourIndices.

    Steps:
      1. Extract the (search_window_size × search_window_size) block.
      2. Otsu threshold on the FULL image (matches MATLAB graythresh(image2D)).
      3. Binarise the block.
      4. bwperim equivalent: boundary = filled XOR eroded  (skimage).
      5. Label blobs; keep the largest (matches MATLAB sort descend, take 1).
      6. Translate local coords back to full-image space.
      7. Measure long/short axis on the local contour image.

    Returns
    -------
    boundary_rows, boundary_cols : perimeter pixels in full-image space
    long_axis, short_axis        : axis lengths in pixels
    num_pixels                   : number of pixels inside the filled contour
    """
    H, W = image_2d.shape

    # clamp block to image bounds
    r0 = max(0, sx)
    c0 = max(0, sy)
    r1 = min(H, sx + search_window_size)
    c1 = min(W, sy + search_window_size)

    block = image_2d[r0:r1, c0:c1]

    if block.size == 0 or block.max() == block.min():
        # fallback: return window perimeter (matches MATLAB else branch)
        p_rows, p_cols = get_window_perimeter(x, y, search_window_size, search_window_size)
        return p_rows, p_cols, 0.0, 0.0, 0

    # Otsu on FULL image (MATLAB: graythresh(image2D))
    # level = threshold_otsu(image_2d)
    # block_binary = block > level
    # # norm_image = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-6)
    # # level = threshold_otsu(norm_image)
    # # block_binary = block > (level * (block.max() - block.min()) + block.min())

    # # bwperim: boundary = filled minus 4-connected erosion
    # eroded   = binary_erosion(block_binary)
    # boundary = block_binary & ~eroded

    # # label boundary blobs, keep largest (MATLAB: sort descend, take index 1)
    # labeled, n_blobs = label(boundary, return_num=True)
    # if n_blobs == 0:
    #     p_rows, p_cols = get_window_perimeter(x, y, search_window_size, search_window_size)
    #     return p_rows, p_cols, 0.0, 0.0, 0

    # props = regionprops(labeled)
    # largest = max(props, key=lambda r: r.area)

    # # contour image (local coords)
    # contour_image = labeled == largest.label

    # # translate to full-image coords (MATLAB: X = round(X + sx))
    # local_rows, local_cols = np.where(contour_image)
    # boundary_rows = np.round(local_rows + r0).astype(int)
    # boundary_cols = np.round(local_cols + c0).astype(int)

    # # axis lengths
    # long_axis, short_axis = get_long_and_short_axis(contour_image)

    # # num_pixels = pixels INSIDE the filled contour
    # # MATLAB getNumberOfPixels: imfill then counts non-filled pixels (inverted)
    # # Equivalent: count pixels inside the filled mask
    # filled = binary_fill_holes(contour_image)
    # num_pixels = int(np.sum(filled))

    # return boundary_rows, boundary_cols, long_axis, short_axis, num_pixels

    # 1. Threshold the full image
    level = threshold_otsu(image_2d)
    block_binary = block > level

    # 2. Label connected regions in the block
    labeled = label(block_binary)
    if labeled.max() == 0:
        # fallback: return window perimeter
        return get_window_perimeter(x, y, search_window_size, search_window_size) + (0.0, 0.0, 0)

    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    # 3. Create filled mask of largest region
    filled_mask = np.zeros_like(block_binary, dtype=bool)
    filled_mask[largest.coords[:,0], largest.coords[:,1]] = True

    # 4. Compute boundary: filled minus eroded
    eroded = binary_erosion(filled_mask)
    boundary_mask = filled_mask & ~eroded

    # 5. Map local coords back to full image
    filled_coords = largest.coords.copy()
    filled_coords[:,0] += r0
    filled_coords[:,1] += c0

    bY, bX = np.where(boundary_mask)
    boundary_coords = np.column_stack((bY + r0, bX + c0))

    filled_rows = filled_coords[:, 0]
    filled_cols = filled_coords[:, 1]

    boundary_rows = boundary_coords[:, 0]
    boundary_cols = boundary_coords[:, 1]
    # --- END INSERT ---

    # Compute axis lengths
    long_axis, short_axis = get_long_and_short_axis(filled_mask)

    # Count number of pixels inside the filled contour
    num_pixels = int(np.sum(filled_mask))

    return boundary_rows, boundary_cols, long_axis, short_axis, num_pixels


# ---------------------------------------------------------------------------
# ROI overlay (getROIOverlayed)
# ---------------------------------------------------------------------------

def get_roi_overlayed(
    image_2d: np.ndarray,
    boundary_rows: np.ndarray,
    boundary_cols: np.ndarray,
    overlay_value: float = 1500.0,
) -> np.ndarray:
    """
    Burn boundary pixels into a copy of the image at overlay_value HU.
    Matches MATLAB getROIOverlayed exactly.

    Returns a copy — does NOT modify the input array.
    """
    out = image_2d.copy().astype(np.float32)
    H, W = out.shape
    valid = (
        (boundary_rows >= 0) & (boundary_rows < H) &
        (boundary_cols >= 0) & (boundary_cols < W)
    )
    out[boundary_rows[valid], boundary_cols[valid]] = overlay_value
    return out


# ---------------------------------------------------------------------------
# Closed-contour helpers (manual polygon path — getCloseContour branch)
# ---------------------------------------------------------------------------

def get_all_pixels_in_closed_contour(
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return all (row, col) pixel indices that lie inside (or on) the polygon
    defined by (x_pts, y_pts).  Matches MATLAB getAllPixelsInClosedContour
    which uses inpolygon on a 512×512 grid.

    x_pts = row coordinates of polygon vertices
    y_pts = col coordinates of polygon vertices
    """
    H, W = image_shape
    # skimage.draw.polygon returns (rows, cols) inside the polygon
    rr, cc = polygon(x_pts, y_pts, shape=(H, W))
    return rr, cc


# ---------------------------------------------------------------------------
# Main entry point — getContour
# ---------------------------------------------------------------------------

def get_contour(
    x: int | np.ndarray,
    y: int | np.ndarray,
    search_window_size: int,
    slice_idx: int,
    time_point_idx: int,
    four_d_image_set: np.ndarray,
    pixel_spacing: tuple[float, float],
) -> ContourResult:
    """
    Python translation of MATLAB getContour.

    Parameters
    ----------
    x, y               : click coordinates (scalars) OR arrays of polygon vertices
    search_window_size : side length of search window in pixels
    slice_idx          : 0-based Z index
    time_point_idx     : 0-based T index
    four_d_image_set   : (T, Z, H, W) float array
    pixel_spacing      : (row_spacing_mm, col_spacing_mm)

    Returns
    -------
    ContourResult with boundary pixels, filled pixels, radii and area.
    """
    T, Z, H, W = four_d_image_set.shape
    image_2d = four_d_image_set[
        time_point_idx,
        slice_idx,
    ].copy().astype(np.float32)

    avg_spacing_mm = (pixel_spacing[0] + pixel_spacing[1]) / 2.0

    # ── branch 1: polygon vertices supplied (manual contour) ────────────────
    if np.ndim(x) > 0 and np.size(x) > 1:
        # close the contour (MATLAB: x = [x', x(1)])
        x_closed = np.append(np.asarray(x), x[0])
        y_closed = np.append(np.asarray(y), y[0])

        # boundary = the supplied polygon vertices
        boundary_rows = x_closed.astype(int)
        boundary_cols = y_closed.astype(int)

        # filled region via inpolygon equivalent
        # filled_rows, filled_cols = get_all_pixels_in_closed_contour(
        #     x_closed, y_closed, (H, W)
        # )
        filled_rows, filled_cols = polygon(boundary_rows, boundary_cols, shape=image_2d.shape)


        num_pixels       = len(filled_rows)
        area_cm2         = num_pixels * avg_spacing_mm ** 2 / 100.0
        long_axis, short_axis = 0.0, 0.0
        radius_cm        = 0.0

    # ── branch 2: single click point (automatic contour detection) ──────────
    else:
        xi = int(x)
        yi = int(y)
        sx, sy = get_search_window(xi, yi, search_window_size)

        boundary_rows, boundary_cols, long_axis, short_axis, num_pixels = \
            get_contour_indices(sx, sy, search_window_size, image_2d, xi, yi)

        radius_mm  = (long_axis + short_axis) / 2.0 * avg_spacing_mm
        radius_cm  = radius_mm / 10.0
        area_cm2   = num_pixels * avg_spacing_mm ** 2 / 100.0

        # filled region: flood-fill the boundary to get interior pixels
        # Build a small mask covering just the boundary, fill it, extract interior
        H, W = image_2d.shape
        if len(boundary_rows) > 0:
            # r0, c0 = boundary_rows.min(), boundary_cols.min()
            # r1, c1 = boundary_rows.max() + 1, boundary_cols.max() + 1
            # local_h = r1 - r0
            # local_w = c1 - c0

            # local_mask = np.zeros((local_h, local_w), dtype=bool)
            # local_mask[boundary_rows - r0, boundary_cols - c0] = True

            # filled_local = binary_fill_holes(local_mask)
            # fr_local, fc_local = np.where(filled_local)
            # filled_rows = fr_local + r0
            # filled_cols = fc_local + c0
            filled_rows, filled_cols = polygon(boundary_rows, boundary_cols, shape=(H, W))

        else:
            filled_rows = np.array([], dtype=int)
            filled_cols = np.array([], dtype=int)

    return ContourResult(
        boundary_rows = boundary_rows,
        boundary_cols = boundary_cols,
        filled_rows   = filled_rows,
        filled_cols   = filled_cols,
        long_axis     = long_axis,
        short_axis    = short_axis,
        radius_cm     = radius_cm,
        area_cm2      = area_cm2,
    )


# ---------------------------------------------------------------------------
# ROI object builder (getROI)
# ---------------------------------------------------------------------------

def get_roi(
    study_name:         str,
    num_time_points:    int,
    num_slices:         int,
    x: int, y: int, z: int, t: int,
    sampled_curve:      list[float],
    time_points:        list[float],
    fitted_curve:       list[float],
    fitted_time_points: list[float],
    roi_x_boundary:     list[int],
    roi_y_boundary:     list[int],
    auc_fitted:         float = 0.0,
    extra_points:       list  = None,
    number_of_stenosis: int   = 0,
    rs1: float = 0.0,
    rs2: float = 0.0,
    rs3: float = 0.0,
    entry_q: float = 0.0,
    exit_q:  float = 0.0,
) -> ROIObject:
    """
    Python translation of MATLAB getROI.
    Packages all ROI data into a single ROIObject.
    """
    return ROIObject(
        study_name         = study_name,
        num_time_points    = num_time_points,
        num_slices         = num_slices,
        x = x, y = y, z = z, t = t,
        curve              = list(sampled_curve),
        time_points        = list(time_points),
        fitted_curve       = list(fitted_curve),
        fitted_time_points = list(fitted_time_points),
        auc_fitted         = float(auc_fitted),
        roi_x_boundary     = list(roi_x_boundary),
        roi_y_boundary     = list(roi_y_boundary),
        extra_points       = extra_points or [],
        number_of_stenosis = number_of_stenosis,
        rs1 = rs1, rs2 = rs2, rs3 = rs3,
        entry_q = entry_q, exit_q = exit_q,
    )