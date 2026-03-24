"""
roi_curve_processing.py
-----------------------
Baseline detection and curve preprocessing for gamma variate fitting.

Key functions:
    find_baseline()       - Auto-detect baseline position
    preprocess_curve()    - Subtract baseline and prepare for fitting
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict


def find_baseline(tdc: np.ndarray, threshold: float = 10.0) -> int:
    """
    Auto-detect baseline position (0-based index of last baseline point).
    
    Baseline ends when signal increases by more than `threshold` above
    the mean of the initial flat region.
    
    Parameters
    ----------
    tdc : np.ndarray
        Time-density curve (HU values)
    threshold : float
        Threshold for detecting contrast arrival (HU)
    
    Returns
    -------
    int
        0-based index of the last baseline point
        (Returns 0 if no clear baseline detected)
    """
    tdc = np.asarray(tdc, dtype=float).flatten()
    
    if len(tdc) < 3:
        return 0
    
    # Start with first 2-3 points
    baseline_mean = float(np.mean(tdc[:min(3, len(tdc))]))
    
    # Find where signal exceeds baseline by threshold
    for i in range(len(tdc)):
        if tdc[i] > baseline_mean + threshold:
            # Return the point BEFORE the increase
            return max(0, i - 1)
    
    # No clear contrast arrival - use first point
    return 0


def preprocess_curve(
    raw_tdc: np.ndarray,
    time_points: np.ndarray,
    washout_point: Optional[int] = None,
    baseline_override: Optional[int] = None,
) -> Dict:
    """
    Preprocess curve for gamma variate fitting.
    
    Steps:
        1. Determine baseline position (auto-detect or use override)
        2. Subtract baseline (mean of baseline points)
        3. Determine recirculation start (washout point)
    
    Parameters
    ----------
    raw_tdc : np.ndarray
        Raw HU time-density curve
    time_points : np.ndarray
        Time points (seconds)
    washout_point : int, optional
        1-based index of last point before recirculation
        (e.g., 8 means fit points 1-8, exclude 9+)
        If None or 0, use all points
    baseline_override : int, optional
        1-based COUNT of baseline points to use
        (e.g., 2 means use points 1 and 2 as baseline)
        If None, auto-detect
    
    Returns
    -------
    dict
        {
            'subtracted_curve': baseline-subtracted HU curve,
            'baseline_position': 0-based index of last baseline point,
            'baseline_value': baseline HU value (what was subtracted),
            'recirculation_start': number of points to use for fitting,
            'raw_curve': original curve (for reference)
        }
    """
    raw_tdc = np.asarray(raw_tdc, dtype=float).flatten()
    time_points = np.asarray(time_points, dtype=float).flatten()
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Determine baseline position
    # ─────────────────────────────────────────────────────────────────
    if baseline_override is not None and baseline_override > 0:
        # User specified: baseline_override is COUNT of baseline points
        # e.g., 2 means points 1-2 (indices 0-1)
        baseline_position_0based = int(baseline_override) - 1
    else:
        # Auto-detect
        baseline_position_0based = _get_baseline_from_delta_diff(raw_tdc)

    
    # Ensure we have at least 1 baseline point
    baseline_position_0based = max(0, min(baseline_position_0based, len(raw_tdc) - 1))
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Calculate and subtract baseline
    # ─────────────────────────────────────────────────────────────────
    # Baseline value = mean of points from 0 to baseline_position (inclusive)
    baseline_count = baseline_position_0based + 1
    baseline_value = float(np.mean(raw_tdc[:baseline_count]))
    
    # Subtract baseline from entire curve
    subtracted_curve = raw_tdc - baseline_value
    subtracted_curve = np.maximum(subtracted_curve, 0)
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Determine recirculation start (how many points to fit)
    # ─────────────────────────────────────────────────────────────────
    if washout_point is not None and washout_point > 0:
        # washout_point is 1-based INDEX of last point to fit
        # e.g., 8 means fit points 1-8 (indices 0-7)
        recirculation_start = int(washout_point)

    else:
        # Use all points
        recirculation_start = len(raw_tdc)
    
    # Ensure we don't exceed array bounds
    recirculation_start = min(recirculation_start, len(raw_tdc))
    
    return {
    'subtracted_curve': subtracted_curve,
    'baseline_position': baseline_position_0based,
    'baseline_value': baseline_value,
    'recirculation_start': recirculation_start,
    'truncated_curve': subtracted_curve[:recirculation_start],
    'truncated_time': time_points[:recirculation_start],
    'raw_curve': raw_tdc,
}


# ─────────────────────────────────────────────────────────────────────
# Testing / Validation
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test data
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])
    signal = np.array([50, 52, 54, 165, 285, 245, 190, 145, 115, 105, 100, 98, 96])
    
    print("Test 1: Auto-detect baseline, use all points")
    result = preprocess_curve(signal, time)
    print(f"  Baseline position: {result['baseline_position']} (0-based)")
    print(f"  Baseline value: {result['baseline_value']:.2f}")
    print(f"  Peak (subtracted): {np.max(result['subtracted_curve']):.2f}")
    print(f"  Recirculation start: {result['recirculation_start']}")
    print()
    
    print("Test 2: Baseline=2 (points 1-2), washout=8 (fit points 1-8)")
    result = preprocess_curve(signal, time, washout_point=8, baseline_override=2)
    print(f"  Baseline position: {result['baseline_position']} (0-based)")
    print(f"  Baseline value: {result['baseline_value']:.2f}")
    print(f"  Subtracted curve: {result['subtracted_curve']}")
    print(f"  Peak (subtracted): {np.max(result['subtracted_curve']):.2f}")
    print(f"  Recirculation start: {result['recirculation_start']}")
    print(f"  Points to fit: {result['subtracted_curve'][:result['recirculation_start']]}")

# """
# curve_processing
# -------------------
# Python translation of MATLAB baseline/washout curve processing functions:
#     findBaseline
#     getBaseLineFromDeltaDiff      ← active method (others kept for reference)
#     getBaseLineFromThreshold
#     getBaseLineFromUpwardTrend
#     getBaseLineFromFindChangePts
#     subtractBaseline
#     getStartingPointOfRecirculationPhase  (washout)

# No GUI imports. All functions operate on 1-D numpy arrays.

# MATLAB index convention: 1-based.
# Python convention here:   0-based.
# All public functions return 0-based indices.
# """

# from __future__ import annotations

# import numpy as np
# from scipy.ndimage import label as scipy_label


# # ---------------------------------------------------------------------------
# # Public entry point
# # ---------------------------------------------------------------------------

# def find_baseline(tdc: np.ndarray) -> int:
#     """
#     Return the 0-based index of the last baseline frame
#     (i.e. the frame just before contrast arrives).

#     Matches MATLAB findBaseline — delegates to getBaseLineFromDeltaDiff.

#     Parameters
#     ----------
#     tdc : 1-D array of HU values (time-density curve), any orientation.

#     Returns
#     -------
#     0-based baseline position index.
#     """
#     tdc = _flatten(tdc)
#     return _get_baseline_from_delta_diff(tdc)


# # ---------------------------------------------------------------------------
# # Active method: getBaseLineFromDeltaDiff
# # ---------------------------------------------------------------------------

def _get_baseline_from_delta_diff(tdc: np.ndarray) -> int:
    """
    Slides a 3-point window through the curve.
    Baseline = startingIndex when tdc[end] - tdc[start] >= 50 HU.
    Falls back to 15% of curve length if threshold never reached.

    MATLAB is 1-based; all arithmetic below is converted to 0-based.

    MATLAB logic:
        startingIndex = 1, endingIndex = 3  (1-based)
        while true
            if tdc(endingIndex) - tdc(startingIndex) >= 50  → baseline = startingIndex
            else  startingIndex++, endingIndex++
            if endingIndex == lengthOfTDC  → break
        if condition: baseline = startingIndex
        else:         baseline = ceil(length * 15/100)
    """
    n = len(tdc)

    if n < 3:
        return 0

    if n == 3:
        # MATLAB sets startingIndex=1, endingIndex=3 but never enters the
        # while-loop (condition stays false), so falls through to the
        # 15% fallback.
        return int(np.ceil(n * 15 / 100)) - 1   # convert to 0-based

    # n > 3: sliding window
    # MATLAB startingIndex / endingIndex are 1-based → subtract 1 here
    start = 0   # MATLAB startingIndex = 1  → 0-based = 0
    end   = 2   # MATLAB endingIndex   = 3  → 0-based = 2
    condition = False

    while True:
        if tdc[end] - tdc[start] >= 50:
            condition = True
            break
        else:
            start += 1
            end   += 1

        # MATLAB: if endingIndex == lengthOfTDC → break
        if end == n - 1:   # 0-based equivalent of endingIndex == lengthOfTDC
            break

    if condition:
        return start          # 0-based
    else:
        # MATLAB: ceil(lengthOfTDC * 15 / 100)  → 1-based, convert to 0-based
        return int(np.ceil(n * 15 / 100)) - 1


# # ---------------------------------------------------------------------------
# # Reference methods (not active but preserved for future use)
# # ---------------------------------------------------------------------------

# def _get_baseline_from_threshold(tdc: np.ndarray) -> int:
#     """
#     MATLAB getBaseLineFromThreshold.
#     Threshold = 10% of max value.
#     Returns 0-based index.
#     """
#     threshold = tdc.max() * 0.10
#     for i in range(len(tdc) - 1):
#         if (tdc[i + 1] - tdc[i]) > threshold:
#             return i + 1   # MATLAB returns i+1 (1-based i+1 → 0-based i+1)
#     return 0


# def _get_baseline_from_upward_trend(tdc: np.ndarray) -> int:
#     """
#     MATLAB getBaseLineFromUpwardTrend.
#     Finds the first run of 3+ consecutive upward steps.
#     Returns 0-based index.
#     """
#     d_tdc = np.diff(tdc) > 0          # boolean differences
#     labeled, n_blobs = scipy_label(d_tdc)

#     if n_blobs == 0:
#         return int(np.ceil(len(tdc) * 20 / 100)) - 1

#     # find blobs with area >= 3
#     for blob_id in range(1, n_blobs + 1):
#         indices = np.where(labeled == blob_id)[0]
#         if len(indices) >= 3:
#             start_idx = int(indices[0])
#             end_idx   = int(indices[-1])
#             # MATLAB: ceil((startingIndex + endingIndex) / 2)
#             # indices here are into diff array (0-based) → add 1 for 1-based
#             mid = int(np.ceil((start_idx + 1 + end_idx + 1) / 2))
#             return mid - 1   # back to 0-based

#     return int(np.ceil(len(tdc) * 20 / 100)) - 1


# def _get_baseline_from_find_change_pts(tdc: np.ndarray) -> int:
#     """
#     MATLAB getBaseLineFromFindChangePts.
#     Uses a simple variance-based change-point detection as a substitute
#     for MATLAB's findchangepts (Signal Processing Toolbox).
#     Returns 0-based index.
#     """
#     peak_idx = int(np.argmax(tdc))
#     if peak_idx < 2:
#         return peak_idx

#     segment = tdc[1:peak_idx]   # MATLAB tdc(2:idx-1)
#     if len(segment) < 2:
#         return peak_idx - 1

#     # simple change-point: largest absolute difference in the segment
#     diffs = np.abs(np.diff(segment))
#     change_pt = int(np.argmax(diffs))   # 0-based within segment
#     # translate back: segment starts at index 1 of tdc
#     return change_pt + 1   # 0-based in tdc


# # ---------------------------------------------------------------------------
# # Baseline subtraction
# # ---------------------------------------------------------------------------

# def subtract_baseline(tdc: np.ndarray, baseline_position: int) -> np.ndarray:
#     """
#     Subtract the mean of all frames up to and including baseline_position
#     from the entire curve.

#     Parameters
#     ----------
#     tdc               : 1-D HU curve
#     baseline_position : 0-based index of last baseline frame

#     Returns
#     -------
#     Baseline-subtracted curve (same length as tdc).
#     """
#     tdc = _flatten(tdc).copy()
#     n_baseline = baseline_position + 1   # inclusive
#     baseline_mean = float(np.mean(tdc[:n_baseline]))
#     return tdc - baseline_mean


# # ---------------------------------------------------------------------------
# # Washout / recirculation truncation
# # ---------------------------------------------------------------------------

# def get_starting_point_of_recirculation_phase(
#     tdc: np.ndarray,
#     washout_point: int | None = None,
# ) -> int:
#     """
#     Return the 0-based index of the first recirculation frame so the
#     fitting can be truncated at washout_point - 1.

#     Two modes
#     ---------
#     1. Manual (washout_point supplied by user):
#        The user enters a 1-based time-point number in the UI.
#        We convert to 0-based and return it directly.

#     2. Automatic (washout_point is None):
#        Finds the first trough after the peak — the point where the
#        curve stops falling and starts rising again (recirculation).
#        Algorithm:
#          - Find peak index.
#          - Scan forward from peak; record the running minimum.
#          - Recirculation starts at the first index after the minimum
#            where the value rises above the minimum by > 5% of peak value.
#          - Falls back to the last index if no rise is found.

#     Returns
#     -------
#     0-based index of the start of the recirculation phase.
#     Fitting should use tdc[:recirculation_start].
#     """
#     tdc = _flatten(tdc)
#     n   = len(tdc)

#     # ── manual mode ──────────────────────────────────────────────────────────
#     if washout_point is not None:
#         # user enters 1-based point number → convert to 0-based
#         idx = int(washout_point) - 1
#         return max(0, min(idx, n - 1))

#     # ── automatic mode ───────────────────────────────────────────────────────
#     peak_idx   = int(np.argmax(tdc))
#     peak_value = float(tdc[peak_idx])

#     if peak_idx >= n - 1:
#         return n - 1

#     # scan forward from peak for the trough then first rise
#     running_min       = float(tdc[peak_idx])
#     running_min_idx   = peak_idx
#     threshold_rise    = peak_value * 0.05   # 5% of peak = "real" rise

#     for i in range(peak_idx + 1, n):
#         val = float(tdc[i])
#         if val < running_min:
#             running_min     = val
#             running_min_idx = i
#         elif val - running_min > threshold_rise and i > running_min_idx:
#             # curve is rising again → recirculation starts here
#             return i

#     return n - 1   # no recirculation detected → use full curve


# # ---------------------------------------------------------------------------
# # Convenience: full pre-processing pipeline
# # ---------------------------------------------------------------------------

# def preprocess_curve(
#     raw_tdc: np.ndarray,
#     time_points: np.ndarray,
#     washout_point: int | None = None,
#     baseline_override: int | None = None,
# ) -> dict:
#     """
#     Run the full MATLAB pre-processing pipeline in one call:
#         1. findBaseline
#         2. subtractBaseline
#         3. getStartingPointOfRecirculationPhase
#         4. Truncate curve + time_points to washout

#     Parameters
#     ----------
#     raw_tdc           : raw HU time-density curve (1-D)
#     time_points       : acquisition times in seconds (1-D, same length)
#     washout_point     : 1-based user-supplied washout frame (or None = auto)
#     baseline_override : 1-based user-supplied baseline frame (or None = auto)

#     Returns
#     -------
#     dict with keys:
#         baseline_position   : 0-based index
#         baseline_mean       : HU value subtracted
#         subtracted_curve    : full baseline-subtracted curve
#         recirculation_start : 0-based index
#         truncated_curve     : curve clipped at recirculation_start
#         truncated_times     : matching time axis
#     """
#     tdc        = _flatten(raw_tdc).copy()
#     t_arr      = np.asarray(time_points, dtype=float)

#     # 1. baseline
#     if baseline_override is not None:
#         baseline_pos = int(baseline_override) - 1   # user is 1-based
#     else:
#         baseline_pos = find_baseline(tdc)

#     baseline_mean    = float(np.mean(tdc[:baseline_pos + 1]))
#     subtracted_curve = tdc - baseline_mean

#     # 2. washout / recirculation
#     recirc_start = get_starting_point_of_recirculation_phase(
#         subtracted_curve, washout_point
#     )

#     # 3. truncate
#     truncated_curve = subtracted_curve[:recirc_start]
#     truncated_times = t_arr[:recirc_start]

#     return {
#         "baseline_position":   baseline_pos,
#         "baseline_mean":       baseline_mean,
#         "subtracted_curve":    subtracted_curve,
#         "recirculation_start": recirc_start,
#         "truncated_curve":     truncated_curve,
#         "truncated_times":     truncated_times,
#     }


# # ---------------------------------------------------------------------------
# # Internal helper
# # ---------------------------------------------------------------------------

# def _flatten(tdc: np.ndarray) -> np.ndarray:
#     """Ensure tdc is a 1-D float array regardless of input orientation."""
#     return np.asarray(tdc, dtype=float).flatten()