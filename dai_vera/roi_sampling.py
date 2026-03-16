"""
roi_sampling.py
---------------
Python translation of MATLAB getBestSample and its helpers:
    getBestSample, getCoordinates, getWindowPerimeter,
    getInterpolatedCurve

No GUI imports. All functions operate on numpy arrays.
Coordinate convention matches MATLAB: X = row, Y = col.

Key algorithm (scalar-click branch):
  - For every time point t:
      - Slide a (roiM x roiN) sample patch across the
        (windowSize x windowSize) search window.
      - At each position compute mean HU over the patch pixels.
      - Keep the position with the highest mean → bestCurve[t].
  - If T < 25, interpolate up to 25 points.

Key algorithm (polygon branch):
  - Fill the polygon to get all interior pixels.
  - For every time point t, average HU over those pixels → bestCurve[t].
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def get_coordinates(x: int, y: int, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return all (row, col) pixel indices in the rectangle
    [x .. x+m] × [y .. y+n] inclusive.

    Matches MATLAB getCoordinates exactly (1-pixel inclusive on both ends).
    """
    rows, cols = [], []
    for i in range(x, x + m + 1):
        for j in range(y, y + n + 1):
            rows.append(i)
            cols.append(j)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def get_window_perimeter(
    sx: int, sy: int, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the perimeter pixel coordinates of the rectangle whose
    top-left corner is (sx, sy) with the given width and height.

    Matches MATLAB getWindowPerimeter exactly.
    Returns (rows, cols).
    """
    rows, cols = [], []
    for i in range(sx, sx + height + 1):
        for j in range(sy, sy + width + 1):
            if i == sx or i == sx + height or j == sy or j == sy + width:
                rows.append(i)
                cols.append(j)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def get_search_window(x: int, y: int, window_size: int) -> tuple[int, int]:
    """
    Return top-left corner (sx, sy) of a square search window
    centred on (x, y).  Matches MATLAB getSearchWindow.
    """
    half = window_size // 2
    return x - half, y - half


# ---------------------------------------------------------------------------
# Interpolation  (getInterpolatedCurve)
# ---------------------------------------------------------------------------

def get_interpolated_curve(
    time_point_values: np.ndarray,
    best_curve: np.ndarray,
    target_points: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate best_curve to target_points evenly spaced time points.

    Matches MATLAB getInterpolatedCurve behaviour:
      - Input:  time_point_values (T,), best_curve (T,)
      - Output: (interpolated_values, interpolated_times) both length target_points

    Uses linear interpolation (MATLAB default interp1).
    """
    t_in  = np.asarray(time_point_values, dtype=float)
    v_in  = np.asarray(best_curve,        dtype=float)

    t_out = np.linspace(t_in[0], t_in[-1], target_points)

    if len(t_in) < 2:
        # nothing to interpolate
        return v_in, t_in

    f = interp1d(t_in, v_in, kind="linear", fill_value="extrapolate")
    v_out = f(t_out)

    return v_out, t_out


# Main entry: getBestSample

def get_best_sample(
    x: int | np.ndarray,
    y: int | np.ndarray,
    window_size: int,
    roi_m: int,
    roi_n: int,
    slice_idx: int,
    four_d_image_set: np.ndarray,
    time_point_values: np.ndarray,
) -> dict:
    """
    Python translation of MATLAB getBestSample.

    Parameters
    ----------
    x, y              : scalar click coords (row, col) OR polygon vertex arrays
    window_size       : search window side length in pixels  (e.g. 20 for 1×1,
                        40 for 2×2 — matches MATLAB windowSize)
    roi_m, roi_n      : sample patch dimensions in pixels    (e.g. 2, 2)
    slice_idx         : 0-based Z index
    four_d_image_set  : (T, Z, H, W) float array
    time_point_values : 1-D array of acquisition times (seconds), length T

    Returns
    -------
    dict with keys:
        best_curve                  : (T,) or (25,) raw sampled HU values
        interpolated_sampled_points : (≥25,) interpolated HU values
        interpolated_time_points    : matching time axis
        search_window_rows          : perimeter rows of search window
        search_window_cols          : perimeter cols of search window
        roi_rows                    : sample-patch perimeter rows (scalar branch)
        roi_cols                    : sample-patch perimeter cols (scalar branch)
    """
    T, Z, H, W = four_d_image_set.shape
    time_point_values = np.asarray(time_point_values, dtype=float)
    n_time = len(time_point_values)

    # ── Branch 1: polygon / manual contour ──────────────────────────────────
    if np.ndim(x) > 0 and np.size(x) > 1:
        x_arr = np.asarray(x, dtype=int)
        y_arr = np.asarray(y, dtype=int)

        # fill the polygon to get interior pixels (MATLAB: imfill(binaryImage,'holes'))
        binary_image = np.zeros((H, W), dtype=bool)
        valid = (x_arr >= 0) & (x_arr < H) & (y_arr >= 0) & (y_arr < W)
        binary_image[x_arr[valid], y_arr[valid]] = True
        filled = binary_fill_holes(binary_image)
        all_x, all_y = np.where(filled)

        best_curve = np.zeros(n_time, dtype=float)
        for t in range(n_time):
            img = four_d_image_set[t, slice_idx]
            # clamp to image bounds
            rx = np.clip(all_x, 0, H - 1)
            ry = np.clip(all_y, 0, W - 1)
            best_curve[t] = float(np.mean(img[rx, ry]))

        search_window_rows = x_arr
        search_window_cols = y_arr
        roi_rows           = x_arr
        roi_cols           = y_arr

    # ── Branch 2: single click — sliding sample patch ────────────────────────
    else:
        xi = int(x)
        yi = int(y)
        sx, sy = get_search_window(xi, yi, window_size)

        search_window_rows, search_window_cols = get_window_perimeter(
            sx, sy, window_size, window_size
        )

        best_curve = np.zeros(n_time, dtype=float)
        roi_rows   = np.array([], dtype=int)
        roi_cols   = np.array([], dtype=int)

        for t in range(n_time):
            img = four_d_image_set[t, slice_idx]
            max_avg = -4096.0
            best_X, best_Y = xi, yi # track best patch 

            # slide sample patch (roiM × roiN) through search window
            # MATLAB: X = sx : sx+windowSize-roiM,  Y = sy : sy+windowSize-roiN
            for X in range(sx, sx + window_size - roi_m + 1):
                for Y in range(sy, sy + window_size - roi_n + 1):

                    # collect patch pixel coords
                    patch_rows, patch_cols = get_coordinates(X, Y, roi_m, roi_n)

                    # clamp to image bounds
                    pr = np.clip(patch_rows, 0, H - 1)
                    pc = np.clip(patch_cols, 0, W - 1)

                    avg = float(np.mean(img[pr, pc]))

                    if avg > max_avg:
                        max_avg = avg
                        best_X, best_Y = X, Y

                    # capture sample-patch perimeter at the click position
                    # MATLAB: if (X == x && Y == y)
                    # if X == xi and Y == yi:
                    #     roi_rows, roi_cols = get_window_perimeter(X, Y, roi_n, roi_m)

                # After sliding all positions, record the perimeter of best patch
                roi_rows, roi_cols = get_window_perimeter(best_X, best_Y, roi_n, roi_m)
                best_curve[t] = max_avg

                # print("t, max_avg:", t, max_avg)
                # print("BEST PATCH top-left: ", best_X, best_Y)
                # print("roi_rows:", roi_rows)
                # print("roi_cols:", roi_cols)
                # print("img slice min/max:", img.min(), img.max())
                    

    # ── Interpolation  (MATLAB: if timePoints < 25) ──────────────────────────
    if n_time < 25:
        interp_values, interp_times = get_interpolated_curve(
            time_point_values, best_curve, target_points=25
        )
    else:
        interp_values = best_curve.copy()
        interp_times  = time_point_values.copy()

    return {
        "best_curve":                   best_curve,
        "interpolated_sampled_points":  interp_values,
        "interpolated_time_points":     interp_times,
        "search_window_rows":           search_window_rows,
        "search_window_cols":           search_window_cols,
        "roi_rows":                     roi_rows,
        "roi_cols":                     roi_cols,
    }