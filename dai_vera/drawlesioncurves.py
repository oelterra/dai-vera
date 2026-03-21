"""
drawlesioncurves.py - FIXED VERSION
------------------------------------
Fixed the K_init calculation in get_fitted_curve function.

PROBLEM: K_init was hardcoded to 1.0, causing poor fitting
FIX: Calculate K_init from the peak of baseline-subtracted data
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Colour / label maps  (match MATLAB getWingCurve)
# ---------------------------------------------------------------------------

_COLOUR = {
    "pre":  "dodgerblue",
    "post": "tomato",
}

_LABEL_SAMPLED = {
    "pre":  "Pre-Lesion Sampled",
    "post": "Post-Lesion Sampled",
}

_LABEL_FITTED = {
    "pre":  "Pre-Lesion Fitted",
    "post": "Post-Lesion Fitted",
}

# gid tags so we can selectively clear each layer without touching the other
_TAG_SAMPLED = "sampled"
_TAG_FITTED  = "fitted"


# ---------------------------------------------------------------------------
# Axis configuration  (MATLAB getWingCurve logic)
# ---------------------------------------------------------------------------

def _y_limits(curve: np.ndarray) -> tuple[float, float]:
    """
    minY = -50
    maxY = max(curve) + 30  if max > 500  else  500
    Matches MATLAB getWingCurve exactly.
    """
    peak = float(np.max(curve)) if curve.size > 0 else 0.0
    return -50.0, (peak + 30.0) if peak > 500.0 else 500.0


def _configure_axes(
        ax: Axes,
        times: np.ndarray,
        curve: np.ndarray,
        time_unit: str = "s",
) -> None:
    t_max = float(np.max(times)) if times.size > 0 else 10.0
    min_y, max_y = _y_limits(curve)

    ax.set_xlim(0, t_max)
    ax.set_ylim(min_y, max_y)

    # X ticks — cast to int AND set labels explicitly so no decimals ever appear
    x_ticks = np.unique(np.linspace(0, t_max, 11).astype(int))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(t)) for t in x_ticks])  # ← this is the fix

    # Y ticks
    interval = int((max_y + abs(min_y)) / 5)
    if interval > 0:
        ax.set_yticks(np.arange(int(min_y), int(max_y) + 1, interval))

    ax.set_xlabel(f"Time ({time_unit})", labelpad=8)
    ax.set_ylabel("Enhancement (HU)", labelpad=8)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_sampled_curve(
    ax:          Axes,
    times:       np.ndarray,
    values:      np.ndarray,
    lesion_type: str,
    time_unit:   str = "s",
) -> None:
    """
    Plot raw sampled HU dots on `ax`.

    Called immediately on Set Pre/Post Lesion button click.
    Clears both sampled and fitted layers first so repeated clicks
    (after dragging the ROI point) always show fresh data.

    Parameters
    ----------
    ax          : matplotlib Axes already embedded in the UI figure
    times       : 1-D acquisition time array in seconds
    values      : 1-D HU curve from getBestSample
    lesion_type : "pre" or "post"
    time_unit   : axis label string, default "s"
    """
    times  = np.asarray(times,  dtype=float).flatten()
    values = np.asarray(values, dtype=float).flatten()
    colour = _COLOUR.get(lesion_type, "white")

    # clear both layers — fresh start on every Set Pre/Post click
    _clear_layer(ax, _TAG_SAMPLED)
    _clear_layer(ax, _TAG_FITTED)
    _remove_legend(ax)

    _configure_axes(ax, times, values, time_unit)

    # sampled dots  (MATLAB: 'bo' pre, 'ro' post)
    (line,) = ax.plot(
        times, values,
        "o",
        color=colour,
        markersize=5,
        alpha=0.85,
        label=_LABEL_SAMPLED.get(lesion_type, "Sampled"),
    )
    line.set_gid(_TAG_SAMPLED)

    _update_legend(ax)
    ax.figure.canvas.draw_idle()


def plot_fitted_overlay(
    ax:           Axes,
    fitted_time:  np.ndarray,
    fitted_curve: np.ndarray,
    lesion_type:  str,
) -> None:
    """
    Overlay the gamma-variate fitted solid line on `ax`.

    Called after the Fit Curve button is clicked.
    Does NOT clear the sampled dots — only replaces any previous fitted line.

    Parameters
    ----------
    ax           : same Axes used for plot_sampled_curve
    fitted_time  : 100-pt dense time axis from get_fitted_curve
    fitted_curve : 100-pt fitted HU values from get_fitted_curve
    lesion_type  : "pre" or "post"
    """
    fitted_time  = np.asarray(fitted_time,  dtype=float).flatten()
    fitted_curve = np.asarray(fitted_curve, dtype=float).flatten()
    colour = _COLOUR.get(lesion_type, "cyan")

    # replace previous fitted line only — keep sampled dots intact
    _clear_layer(ax, _TAG_FITTED)
    _remove_legend(ax)

    # solid fitted line  (MATLAB: 'b-' pre, 'r-' post)
    (line,) = ax.plot(
        fitted_time, fitted_curve,
        "-",
        color=colour,
        linewidth=2,
        label=_LABEL_FITTED.get(lesion_type, "Fitted"),
    )
    line.set_gid(_TAG_FITTED)

    _update_legend(ax)
    ax.figure.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clear_layer(ax: Axes, tag: str) -> None:
    """Remove all line artists whose gid matches `tag`."""
    for line in list(ax.lines):
        if line.get_gid() == tag:
            line.remove()


def _remove_legend(ax: Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _update_legend(ax: Axes) -> None:
    """Rebuild legend from all currently labelled artists."""
    handles = [
        ln for ln in ax.lines
        if ln.get_label() and not ln.get_label().startswith("_")
    ]
    if handles:
        ax.legend(
            handles=handles,
            facecolor="#1e1e1e",
            labelcolor="white",
            fontsize=8,
        )


# ---------------------------------------------------------------------------
# get_fitted_curve — FIXED VERSION
# ---------------------------------------------------------------------------

@dataclass
class FittedCurveResult:
    fitted_curve:               np.ndarray
    fitted_time:                np.ndarray
    baseline_subtracted_curve:  np.ndarray
    rmse:                       float
    k:                          float
    alpha:                      float
    beta:                       float
    baseline_position:          int
    recirculation_start:        int
    auc:                        float
    converged:                  bool = True


import numpy as np
from dataclasses import dataclass

@dataclass
class FittedCurveResult:
    fitted_curve:               np.ndarray
    fitted_time:                np.ndarray
    baseline_subtracted_curve:  np.ndarray
    rmse:                       float
    k:                          float
    alpha:                      float
    beta:                       float
    baseline_position:          int
    recirculation_start:        int
    auc:                        float
    converged:                  bool = True


def get_fitted_curve_safe(
    sample_curve: np.ndarray,
    sample_time: np.ndarray,
    baseline: int = 0,
    washout: int = 0,
) -> FittedCurveResult:
    """
    Fit a gamma-variate curve, or fallback to smooth interpolation
    if gamma fit fails or produces negligible curve values.
    """
    from dai_vera.roi_curve_processing import preprocess_curve, find_baseline
    from dai_vera.gammavariate import fit_modified_gamma_variate, compute_auc

    sample_curve = np.asarray(sample_curve, dtype=float).flatten()
    sample_time  = np.asarray(sample_time, dtype=float).flatten()

    # Convert ms → s if needed
    if len(sample_time) > 1 and sample_time[1] >= 500:
        sample_time = sample_time / 1000.0

    # baseline
    position = baseline if baseline != 0 else find_baseline(sample_curve) + 1
    position = max(1, position)

    processed = preprocess_curve(
        raw_tdc           = sample_curve,
        time_points       = sample_time,
        washout_point     = washout if washout != 0 else None,
        baseline_override = position,
    )

    baseline_subtracted = processed["subtracted_curve"]
    baseline_pos_0      = processed["baseline_position"]
    recirc_start        = processed["recirculation_start"]

    # initial gamma parameters
    peak_value = float(np.max(baseline_subtracted))
    K_init = max(peak_value, 1.0)
    alpha_init = 2.0
    t_peak = float(sample_time[np.argmax(baseline_subtracted)])
    t_at = float(sample_time[max(0, position-1)])
    beta_init = max((t_peak - t_at) / (alpha_init + 1.0), 0.5)

    # Try gamma fit
    converged = True
    try:
        fit = fit_modified_gamma_variate(
            time                   = sample_time,
            data                   = baseline_subtracted,
            K_init                 = K_init,
            alpha_init             = alpha_init,
            beta_init              = beta_init,
            num_points_to_consider = recirc_start,
            contrast_arrival_time  = position,
        )
        baseline_hu_offset = float(processed.get("baseline_value", 0.0))
        fitted_curve = fit.fitted_data + baseline_hu_offset
        fitted_time = fit.stretched_time

        # Check for negligible curve
        if np.all(np.abs(fitted_curve) < 1e-2):
            raise RuntimeError("Gamma fit produced near-zero curve")        
        
    except Exception:
        # Fallback: smooth interpolation over start → end points
        print("Gamma fit failed, using smooth interpolation fallback")
        start_time = sample_time[position-1]
        end_time   = sample_time[washout-1] if washout != 0 else sample_time[-1]
        mask = (sample_time >= start_time) & (sample_time <= end_time)
        fitted_time = np.linspace(start_time, end_time, 100)
        fitted_curve = np.interp(fitted_time, sample_time[mask], sample_curve[mask])        
        
        converged = False
        K_init = alpha_init = beta_init = 0.0

    rmse = float(np.sqrt(np.mean((fitted_curve - np.interp(fitted_time, sample_time, baseline_subtracted)) ** 2)))
    auc = compute_auc(fit.fitted_data, fit.stretched_time)

    baseline_hu_offset = float(processed.get("baseline_value", 0.0))

    return FittedCurveResult(
        fitted_curve=fit.fitted_data + baseline_hu_offset,
        fitted_time=fit.stretched_time,
        baseline_subtracted_curve=baseline_subtracted,
        rmse=rmse,
        k=fit.k,
        alpha=fit.alpha,
        beta=fit.beta,
        baseline_position=baseline_pos_0,
        recirculation_start=recirc_start,
        auc=auc,
        converged=converged,
    )

def get_fitted_curve(
    sample_curve:   np.ndarray,
    sample_time:    np.ndarray,
    baseline:       int = 0,
    washout:        int = 0,
) -> "FittedCurveResult":
    """
    Full pipeline matching MATLAB getFittedCurve.
    
    FIXED: K_init is now calculated from data peak instead of hardcoded to 1.0
    
    Steps:
        1. ms→s conversion if needed
        2. findBaseline / subtractBaseline
        3. washout truncation
        4. fitModifiedGammaVariate (with correct K_init!)
        5. RMSE vs linear reference
    """
    from dai_vera.roi_curve_processing import preprocess_curve, find_baseline
    from dai_vera.gammavariate import fit_modified_gamma_variate, compute_auc

    sample_curve = np.asarray(sample_curve, dtype=float).flatten()
    sample_time  = np.asarray(sample_time,  dtype=float).flatten()

    # ms → s
    if len(sample_time) > 1 and sample_time[1] >= 500:
        sample_time = sample_time / 1000.0

    # baseline
    if baseline != 0:
        position = int(baseline)
    else:
        position = find_baseline(sample_curve) + 1
    if position == 0:
        position = 1

    processed = preprocess_curve(
        raw_tdc           = sample_curve,
        time_points       = sample_time,
        washout_point     = washout   if washout   != 0 else None,
        baseline_override = position,
    )

    baseline_subtracted = processed["subtracted_curve"]
    baseline_pos_0      = processed["baseline_position"]
    recirc_start        = processed["recirculation_start"]

    # ════════════════════════════════════════════════════════════════════
    # FIX: Calculate K_init from the actual data peak
    # ════════════════════════════════════════════════════════════════════
    peak_value = float(np.max(baseline_subtracted))
    
    # Calculate initial parameters based on the curve
    peak_idx = int(np.argmax(baseline_subtracted))
    t_at = float(sample_time[max(0, position - 1)])
    t_peak = float(sample_time[peak_idx])
    
    K_init = max(peak_value, 1.0)  # Use peak, but at least 1.0
    alpha_init = 2.0
    beta_init = max((t_peak - t_at) / (alpha_init + 1.0), 0.5)
    
    # Debug output (will show in console)
    print(f"Curve fitting parameters:")
    print(f"  Baseline value: {processed.get('baseline_value', 'N/A'):.2f}")
    print(f"  Peak (subtracted): {peak_value:.2f}")
    print(f"  K_init: {K_init:.2f}")
    print(f"  alpha_init: {alpha_init:.2f}")
    print(f"  beta_init: {beta_init:.2f}")
    # ════════════════════════════════════════════════════════════════════

    # fit
    try:
        fit = fit_modified_gamma_variate(
            time                   = sample_time,
            data                   = baseline_subtracted,
            K_init                 = K_init,        # ← FIXED!
            alpha_init             = alpha_init,    # ← FIXED!
            beta_init              = beta_init,     # ← FIXED!
            num_points_to_consider = recirc_start,
            contrast_arrival_time  = position,
        )
        converged = fit.converged
    except Exception as exc:
        raise RuntimeError(f"fitModifiedGammaVariate failed: {exc}") from exc

    # RMSE vs linspace reference (matches MATLAB immse approach)
    linear_ref = np.linspace(float(baseline_subtracted[0]),
                             float(baseline_subtracted[-1]), 100)
    rmse = float(np.sqrt(np.mean((fit.fitted_data - linear_ref) ** 2)))
    auc  = compute_auc(fit.fitted_data, fit.stretched_time)

    return FittedCurveResult(
        fitted_curve              = fit.fitted_data + baseline_hu_offset,
        fitted_time               = fit.stretched_time,
        baseline_subtracted_curve = baseline_subtracted,
        rmse                      = rmse,
        k                         = fit.k,
        alpha                     = fit.alpha,
        beta                      = fit.beta,
        baseline_position         = baseline_pos_0,
        recirculation_start       = recirc_start,
        auc                       = auc,
        converged                 = converged,
    )