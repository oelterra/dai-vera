"""
gamma_variate.py
----------------
Python translation of MATLAB fitModifiedGammaVariate.

Model:  ct(t) = K * (t - t_AT)^alpha * exp(-(t - t_AT) / beta)
        where t_AT = contrast arrival time, and (t - t_AT) is clamped to 0.

MATLAB uses lsqcurvefit (Optimization Toolbox).
Python equivalent: scipy.optimize.curve_fit (Levenberg-Marquardt / trust-region).

No GUI imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GammaVariateFitResult:
    """
    Mirrors the outputs of MATLAB fitModifiedGammaVariate.

    fitted_data        : HU values on the dense stretched time axis
    stretched_time     : dense time axis (100 points, linspace of input range)
    k                  : estimated scale parameter
    alpha              : estimated shape parameter
    beta               : estimated time-constant parameter
    contrast_arrival_idx : 0-based index into stretched_time closest to t_AT
    converged          : False if scipy raised OptimizeWarning
    """
    fitted_data:          np.ndarray
    stretched_time:       np.ndarray
    k:                    float
    alpha:                float
    beta:                 float
    contrast_arrival_idx: int
    abs_diff_sum:         float
    converged:            bool = True


# ---------------------------------------------------------------------------
# Modified gamma variate model
# ---------------------------------------------------------------------------

def _mgv(time, K, alpha, beta, t_at_value):
    """
    Strict translation of MATLAB's inner mgv(initialParameters, time)
    """
    tAT = time - t_at_value
    tAT = np.where(tAT < 0, 0, tAT) # Clamp negative to 0
    
    # MATLAB: ct(i) = K * tATalpha(i) * tATexp(i)
    # We use np.clip to prevent overflow errors in exp during optimization
    exponent = -tAT / max(beta, 1e-6)
    return K * (tAT**alpha) * np.exp(np.clip(exponent, -100, 100))


# ---------------------------------------------------------------------------
# Peak-boost loop  (MATLAB while loop)
# ---------------------------------------------------------------------------

def _boost_to_peak(
    params: np.ndarray,
    stretched_time: np.ndarray,
    t_at: float,
    peak_data: float,
    thresh: float = 5.0,
) -> np.ndarray:
    """
    If the fitted peak undershoots the data peak by more than `thresh` HU,
    iteratively scale K upward until it catches up.

    Matches MATLAB:
        factor = 1.05
        while peakData - peakFit > thresh:
            K *= factor
            factor += 0.05
    """
    params = params.copy()
    fitted = _mgv(stretched_time, params[0], params[1], params[2], t_at)
    peak_fit = float(np.max(fitted))
    factor = 1.05

    while peak_data - peak_fit > thresh:
        params[0] *= factor
        fitted    = _mgv(stretched_time, params[0], params[1], params[2], t_at)
        peak_fit  = float(np.max(fitted))
        factor   += 0.05

    return params


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def fit_modified_gamma_variate(
    time: np.ndarray,
    data: np.ndarray,
    K_init: float,
    alpha_init: float,
    beta_init: float,
    num_points_to_consider: int,
    contrast_arrival_time: int,
) -> GammaVariateFitResult:
    """
    Python translation of MATLAB fitModifiedGammaVariate.

    Parameters
    ----------
    time                    : 1-D array of acquisition times (seconds)
    data                    : 1-D baseline-subtracted HU curve (same length)
    K_init                  : initial K  (scale)
    alpha_init              : initial alpha (shape)
    beta_init               : initial beta (time constant)
    num_points_to_consider  : number of time points to use for fitting
                              (0 = use all — matches MATLAB numOfDataPointsToConsider)
    contrast_arrival_time   : 1-based index of contrast arrival
                              (matches MATLAB convention; converted internally)

    Returns
    -------
    GammaVariateFitResult
    """
    time = np.asarray(time, dtype=float).flatten()
    data = np.asarray(data, dtype=float).flatten()

    # ── 1. Discard washout phase (MATLAB: if numOfDataPointsToConsider > 0) ─
    if num_points_to_consider > 0:
        times = time[:num_points_to_consider]
        datas = data[:num_points_to_consider]
    else:
        times = time.copy()
        datas = data.copy()

    logger.debug("fitModifiedGammaVariate: len(times)=%d, len(datas)=%d, "
                 "contrastArrivalTime=%d", len(times), len(datas), contrast_arrival_time)

    # ── 2. Resolve t_AT from the 1-based contrast_arrival_time index ─────────
    # MATLAB: tAT = time - time(contrastArrivalTime)
    # contrastArrivalTime is 1-based → convert to 0-based
    cat_idx_0 = max(0, min(int(contrast_arrival_time) - 1, len(times) - 1))
    t_at      = float(times[cat_idx_0])

    logger.debug("t_AT = %.4f s (index %d)", t_at, cat_idx_0)

    # ── 3. Fit with scipy curve_fit  (≈ lsqcurvefit) ─────────────────────────
    # Fix t_at as a constant by wrapping the model in a lambda.
    def model(t, K, alpha, beta):
        # guard: beta must be > 0 to avoid division by zero
        beta = max(beta, 1e-6)
        return _mgv(t, K, alpha, beta, t_at)

    p0     = [K_init, alpha_init, beta_init]
    # MATLAB: lowerBound=-Inf, upperBound=Inf → no bounds in scipy either
    converged = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            estimated, _ = curve_fit(
                model, times, datas,
                p0=p0,
                maxfev=3000,
                ftol=1e-10,
                xtol=1e-10,
            )
    except (OptimizeWarning, RuntimeError) as exc:
        logger.warning("curve_fit did not converge: %s — using initial params", exc)
        estimated = np.array(p0, dtype=float)
        converged = False

    logger.debug("Estimated K=%.4f  alpha=%.4f  beta=%.4f",
                 estimated[0], estimated[1], estimated[2])

    # ── 4. Build stretched (dense) time axis ──────────────────────────────────
    # MATLAB: stretchedTime = linspace(time(1), time(end))  → 100 points
    stretched_time = np.linspace(float(time[0]), float(time[-1]), 100)

    # ── 5. Find closest index in stretched_time to t_AT ──────────────────────
    # MATLAB: absoluteDifferenceValues = abs(stretchedTime - time(contrastArrivalTime))
    #         closestIndex = find(abs == min(abs))  → take first if multiple
    abs_diff = np.abs(stretched_time - t_at)
    closest_idx = int(np.argmin(abs_diff))   # already 0-based

    # update t_at to the exact value on the stretched grid
    t_at_stretched = float(stretched_time[closest_idx])

    # ── 6. Compute fitted curve on stretched time ─────────────────────────────
    fitted_data = _mgv(stretched_time, estimated[0], estimated[1], estimated[2],
                       t_at_stretched)

    # ── 7. Peak-boost loop ────────────────────────────────────────────────────
    peak_data    = float(np.max(data))
    estimated    = _boost_to_peak(estimated, stretched_time, t_at_stretched, peak_data)
    fitted_data  = _mgv(stretched_time, estimated[0], estimated[1], estimated[2],
                        t_at_stretched)
    
    # We calculate what the curve would be AT the original sample points
    fitted_at_samples = _mgv(time, estimated[0], estimated[1], estimated[2], t_at)
    # Sum of absolute differences between raw data and the fit
    abs_diff_sum = float(np.sum(np.abs(data - fitted_at_samples)))

    logger.debug("Final K=%.4f  alpha=%.4f  beta=%.4f", *estimated)

    return GammaVariateFitResult(
        fitted_data          = fitted_data,
        stretched_time       = stretched_time,
        k                    = float(estimated[0]),
        alpha                = float(estimated[1]),
        beta                 = float(estimated[2]),
        contrast_arrival_idx = closest_idx,
        abs_diff_sum         = abs_diff_sum, # <--- Return this to the UI
        converged            = converged,
    )


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------

def compute_auc(fitted_data: np.ndarray, stretched_time: np.ndarray) -> float:
    """
    Compute the area under the fitted gamma variate curve using
    the trapezoidal rule. what MATLAB would compute via trapz().
    """
    return float(np.trapezoid(fitted_data, stretched_time))


# ---------------------------------------------------------------------------
# Default initial parameters helper
# ---------------------------------------------------------------------------

def get_default_initial_params(
    tdc: np.ndarray,
    time: np.ndarray,
    contrast_arrival_idx: int,
) -> tuple[float, float, float]:
    """
    Generate reasonable starting parameters for the gamma variate fit
    based on the observed curve, so the UI can pre-populate K / alpha / beta.

    Rules of thumb matching MATLAB behaviour:
        K     ~ peak HU value
        alpha ~ 2.0  (typical bolus shape)
        beta  ~ time_to_peak / (alpha + 1)  (mode of gamma variate)
    """
    tdc  = np.asarray(tdc,  dtype=float).flatten()
    time = np.asarray(time, dtype=float).flatten()

    peak_idx  = int(np.argmax(tdc))
    peak_hu   = float(tdc[peak_idx])
    t_at      = float(time[max(0, contrast_arrival_idx)])
    t_peak    = float(time[peak_idx])

    K     = max(peak_hu, 1.0)
    alpha = 2.0
    beta  = max((t_peak - t_at) / (alpha + 1.0), 0.1)

    return K, alpha, beta