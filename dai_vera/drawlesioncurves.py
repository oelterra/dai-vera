""" drawlesioncurves.py
--------------------
Curve plotting helpers and gamma-variate fitting pipeline.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from matplotlib.axes import Axes


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

_TAG_SAMPLED = "sampled"
_TAG_FITTED  = "fitted"


# ---------------------------------------------------------------------------
# Result dataclass  (defined ONCE)
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

    x_ticks = np.unique(np.linspace(0, t_max, 11).astype(int))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(t)) for t in x_ticks])

    interval = int((max_y + abs(min_y)) / 5)
    if interval > 0:
        ax.set_yticks(np.arange(int(min_y), int(max_y) + 1, interval))

    ax.set_xlabel(f"Time ({time_unit})", labelpad=8)
    ax.set_ylabel("Enhancement (HU)", labelpad=8)


# ---------------------------------------------------------------------------
# Public plotting API
# ---------------------------------------------------------------------------

def plot_sampled_curve(ax, times, values, lesion_type, time_unit="s"):
    times  = np.asarray(times, dtype=float).flatten()
    values = np.asarray(values, dtype=float).flatten()

    _clear_layer(ax, _TAG_SAMPLED)
    _clear_layer(ax, _TAG_FITTED)
    _remove_legend(ax)

    _configure_axes(ax, times, values, time_unit)

    # purple curve
    line_path, = ax.plot(
        times, values, "-",
        color="mediumpurple",
        linewidth=2,
        alpha=0.9
    )
    line_path.set_gid(_TAG_SAMPLED)

    # green points
    line_pts, = ax.plot(
        times, values, "o",
        color="limegreen",
        markersize=6,
        alpha=1.0,
        label="Sampled Points"
    )
    line_pts.set_gid(_TAG_SAMPLED)

    _update_legend(ax)
    ax.figure.canvas.draw_idle()

def plot_fitted_overlay(ax, fitted_time, fitted_curve, lesion_type):
    fitted_time  = np.asarray(fitted_time).flatten()
    fitted_curve = np.asarray(fitted_curve).flatten()

    # Clean up existing layers
    _clear_layer(ax, _TAG_SAMPLED)
    _clear_layer(ax, _TAG_FITTED)
    _remove_legend(ax)

    # Plot ONLY the fitted line
    line_path, = ax.plot(
        fitted_time,
        fitted_curve,
        "-", # This ensures a solid line
        color="mediumpurple",
        linewidth=2.5,
        label="Fitted Curve"
    )
    line_path.set_gid(_TAG_FITTED)

    # Update UI elements
    _update_legend(ax)
    ax.figure.canvas.draw_idle()


# Internal helpers
# ---------------------------------------------------------------------------

def _clear_layer(ax: Axes, tag: str) -> None:
    for line in list(ax.lines):
        if line.get_gid() == tag:
            line.remove()


def _remove_legend(ax: Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _update_legend(ax: Axes) -> None:
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
# get_fitted_curve_safe
# ---------------------------------------------------------------------------

def get_fitted_curve(
        sample_curve: np.ndarray,
        sample_time: np.ndarray,
        baseline: int = 0,
        washout: int = 0,
) -> FittedCurveResult:
    from dai_vera.roi_curve_processing import preprocess_curve, find_baseline
    from dai_vera.gammavariate import fit_modified_gamma_variate, compute_auc

    # Ensure 1D arrays (Python equivalent of handling isrow/iscolumn)
    sample_curve = np.asarray(sample_curve, dtype=float).flatten()
    sample_time = np.asarray(sample_time, dtype=float).flatten()

    # 1. Baseline Logic
    if baseline != 0:
        position = int(baseline)
    else:
        # findBaseline(sampleCurve)
        position = find_baseline(sample_curve) + 1

    if position <= 0:
        position = 1

    # 2. Convert time (ms -> s)
    if len(sample_time) > 1 and sample_time[1] >= 500:
        sample_time = sample_time / 1000.0

    # 3. Baseline Subtraction & Washout Point
    # This matches the MATLAB call to subtractBaseline and getStartingPoint...
    processed = preprocess_curve(
        raw_tdc=sample_curve,
        time_points=sample_time,
        washout_point=washout if washout != 0 else None,
        baseline_override=position,
    )

    baseline_subtracted = processed["subtracted_curve"]
    baseline_hu_offset = float(processed.get("baseline_value", 0.0))
    recirc_pos = processed["recirculation_start"]

    # 4. Initial Guesses (Matching your MATLAB code: k=1, alpha=5, beta=1.5)
    k_init, alpha_init, beta_init = 1.0, 5.0, 1.5

    # 5. Curve Fitting
    try:
        fit = fit_modified_gamma_variate(
            time=sample_time,
            data=baseline_subtracted,
            K_init=k_init,
            alpha_init=alpha_init,
            beta_init=beta_init,
            num_points_to_consider=recirc_pos,
            contrast_arrival_time=position,
        )
        converged = fit.converged
    except Exception as exc:
        raise RuntimeError(f"fitModifiedGammaVariate failed: {exc}")

    # 6. RMSE Logic (Matches MATLAB: sqrt(immse(fit, linspace(start, end))))
    # This compares the fit to a straight line from the first to last subtracted point.
    linear_ref = np.linspace(
        float(baseline_subtracted[0]),
        float(baseline_subtracted[-1]),
        len(fit.fitted_data)
    )
    rmse = float(np.sqrt(np.mean((fit.fitted_data - linear_ref) ** 2)))

    auc = compute_auc(fit.fitted_data, fit.stretched_time)

    return FittedCurveResult(
        fitted_curve=fit.fitted_data + baseline_hu_offset,
        fitted_time=fit.stretched_time,
        baseline_subtracted_curve=baseline_subtracted,
        rmse=rmse,
        k=fit.k,
        alpha=fit.alpha,
        beta=fit.beta,
        baseline_position=position,
        recirculation_start=recirc_pos,
        auc=auc,
        converged=converged,
    )

# ---------------------------------------------------------------------------
# get_fitted_curve  (primary path used by curves_roi_page)
# ---------------------------------------------------------------------------

def get_fitted_curve(
        sample_curve: np.ndarray,
        sample_time: np.ndarray,
        baseline: int = 0,
        washout: int = 0,
) -> FittedCurveResult:
    from dai_vera.roi_curve_processing import preprocess_curve, find_baseline
    from dai_vera.gammavariate import fit_modified_gamma_variate, compute_auc

    # Ensure 1D arrays (Python equivalent of handling isrow/iscolumn)
    sample_curve = np.asarray(sample_curve, dtype=float).flatten()
    sample_time = np.asarray(sample_time, dtype=float).flatten()

    # 1. Baseline Logic
    if baseline != 0:
        position = int(baseline)
    else:
        # findBaseline(sampleCurve)
        position = find_baseline(sample_curve) + 1

    if position <= 0:
        position = 1

    # 2. Convert time (ms -> s)
    if len(sample_time) > 1 and sample_time[1] >= 500:
        sample_time = sample_time / 1000.0

    # 3. Baseline Subtraction & Washout Point
    # This matches the MATLAB call to subtractBaseline and getStartingPoint...
    processed = preprocess_curve(
        raw_tdc=sample_curve,
        time_points=sample_time,
        washout_point=washout if washout != 0 else None,
        baseline_override=position,
    )

    baseline_subtracted = processed["subtracted_curve"]
    baseline_hu_offset = float(processed.get("baseline_value", 0.0))
    recirc_pos = processed["recirculation_start"]

    # 4. Initial Guesses (Matching your MATLAB code: k=1, alpha=5, beta=1.5)
    k_init, alpha_init, beta_init = 1.0, 5.0, 1.5

    # 5. Curve Fitting
    try:
        fit = fit_modified_gamma_variate(
            time=sample_time,
            data=baseline_subtracted,
            K_init=k_init,
            alpha_init=alpha_init,
            beta_init=beta_init,
            num_points_to_consider=recirc_pos,
            contrast_arrival_time=position,
        )
        converged = fit.converged
    except Exception as exc:
        raise RuntimeError(f"fitModifiedGammaVariate failed: {exc}")

    # 6. RMSE Logic (Matches MATLAB: sqrt(immse(fit, linspace(start, end))))
    # This compares the fit to a straight line from the first to last subtracted point.
    linear_ref = np.linspace(
        float(baseline_subtracted[0]),
        float(baseline_subtracted[-1]),
        len(fit.fitted_data)
    )
    rmse = float(np.sqrt(np.mean((fit.fitted_data - linear_ref) ** 2)))

    auc = compute_auc(fit.fitted_data, fit.stretched_time)

    return FittedCurveResult(
        fitted_curve=fit.fitted_data + baseline_hu_offset,
        fitted_time=fit.stretched_time,
        baseline_subtracted_curve=baseline_subtracted,
        rmse=rmse,
        k=fit.k,
        alpha=fit.alpha,
        beta=fit.beta,
        baseline_position=position,
        recirculation_start=recirc_pos,
        auc=auc,
        converged=converged,
    )