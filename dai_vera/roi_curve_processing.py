""" roi_curve_processing.py
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

    # Subtract baseline from entire curve.
    # FIX: Do NOT clamp to zero — the original np.maximum(subtracted, 0)
    # was flattening pre-peak points to a straight line at 0, which
    # caused the gamma variate fitter to receive a degenerate curve.
    subtracted_curve = raw_tdc - baseline_value

    # ─────────────────────────────────────────────────────────────────
    # 3. Determine recirculation start (how many points to fit)
    # ─────────────────────────────────────────────────────────────────
    if washout_point is not None and washout_point > 0:
        # washout_point is 1-based INDEX of last point to fit
        recirculation_start = int(washout_point)
    else:
        # Use all points
        recirculation_start = len(raw_tdc)

    # Ensure we don't exceed array bounds
    recirculation_start = min(recirculation_start, len(raw_tdc))

    return {
        'subtracted_curve':   subtracted_curve,
        'baseline_position':  baseline_position_0based,
        'baseline_value':     baseline_value,
        'recirculation_start': recirculation_start,
        'truncated_curve':    subtracted_curve[:recirculation_start],
        'truncated_time':     time_points[:recirculation_start],
        'raw_curve':          raw_tdc,
    }


# ─────────────────────────────────────────────────────────────────────
# Active baseline method: getBaseLineFromDeltaDiff
# ─────────────────────────────────────────────────────────────────────

def _get_baseline_from_delta_diff(tdc: np.ndarray) -> int:
    """
    Slides a 3-point window through the curve.
    Baseline = startingIndex when tdc[end] - tdc[start] >= 50 HU.
    Falls back to 15% of curve length if threshold never reached.

    MATLAB is 1-based; all arithmetic below is converted to 0-based.
    """
    n = len(tdc)

    if n < 3:
        return 0

    if n == 3:
        return int(np.ceil(n * 15 / 100)) - 1   # 0-based

    # n > 3: sliding window
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

        if end == n - 1:
            break

    if condition:
        return start          # 0-based
    else:
        return int(np.ceil(n * 15 / 100)) - 1


# ─────────────────────────────────────────────────────────────────────
# Testing / Validation
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    time   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])
    signal = np.array([50, 52, 54, 165, 285, 245, 190, 145, 115, 105, 100, 98, 96])

    print("Test 1: Auto-detect baseline, use all points")
    result = preprocess_curve(signal, time)
    print(f"  Baseline position: {result['baseline_position']} (0-based)")
    print(f"  Baseline value: {result['baseline_value']:.2f}")
    print(f"  Subtracted curve: {result['subtracted_curve']}")
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