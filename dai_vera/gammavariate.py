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

def _mgv(t, K, alpha, beta, t_at):
    t = np.asarray(t, dtype=float)

    # Shift time
    tAT = t - t_at

    # Clamp BEFORE power (CRITICAL)
    tAT = np.maximum(tAT, 0)

    # Safe computation
    with np.errstate(invalid='ignore'):
        y = K * (tAT ** alpha) * np.exp(-tAT / beta)

    # Replace NaNs just in case
    y = np.nan_to_num(y)

    return y
# ---------------------------------------------------------------------------
# Peak-boost loop  (MATLAB while loop)
# ---------------------------------------------------------------------------

# def _boost_to_peak(
#     params: np.ndarray,
#     stretched_time: np.ndarray,
#     t_at: float,
#     peak_data: float,
#     thresh: float = 5.0,
# ) -> np.ndarray:
#     """
#     If the fitted peak undershoots the data peak by more than `thresh` HU,
#     iteratively scale K upward until it catches up.

#     Matches MATLAB:
#         factor = 1.05
#         while peakData - peakFit > thresh:
#             K *= factor
#             factor += 0.05
#     """
#     params = params.copy()
#     fitted = _mgv(stretched_time, params[0], params[1], params[2], t_at)
#     peak_fit = float(np.max(fitted))
#     factor = 1.05

#     while peak_data - peak_fit > thresh:
#         params[0] *= factor
#         fitted    = _mgv(stretched_time, params[0], params[1], params[2], t_at)
#         peak_fit  = float(np.max(fitted))
#         factor   += 0.05

#     return params


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

    # 2. Get t_AT value (convert 1-based to 0-based index)
    cat_idx_0 = max(0, min(int(contrast_arrival_time) - 1, len(times) - 1))
    t_at = float(times[cat_idx_0])  # This is the TIME VALUE, not index
    
    logger.debug(f"t_AT = {t_at:.4f} s (index {cat_idx_0})")

    # 3. Define model with FIXED t_at
    def model(t, K, alpha, beta):
        beta = max(beta, 1e-6)

        # Use SAME t_AT time value everywhere (matches MATLAB behavior better)
        tAT = t - t_at

        tAT = np.where(tAT < 0, 0, tAT)

        tAT_safe = np.where(tAT == 0, 1e-12, tAT)

        return K * (tAT_safe ** alpha) * np.exp(np.clip(-tAT / beta, -100, 100))
    
    # 4. Fit the model
    p0 = [K_init, alpha_init, beta_init]
    converged = True
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            estimated, _ = curve_fit(
                model, times, datas,
                p0=p0,
                bounds=([0, 0, 1e-6], [np.inf, 10, np.inf]),
                maxfev=3000,
                ftol=1e-10,
                xtol=1e-10,
            )
    except (OptimizeWarning, RuntimeError) as exc:
        logger.warning(f"curve_fit did not converge: {exc}")
        estimated = np.array(p0, dtype=float)
        converged = False

    logger.debug(f"Estimated K={estimated[0]:.4f}, alpha={estimated[1]:.4f}, beta={estimated[2]:.4f}")

    # 5. Build stretched time axis
    stretched_time = np.linspace(float(time[0]), float(time[-1]), 100)
    
    # 6. Find closest index to t_AT in stretched_time
    abs_diff = np.abs(stretched_time - t_at)
    closest_idx = int(np.argmin(abs_diff))
    t_at_stretched = float(stretched_time[closest_idx])

    # 7. Compute fitted curve on stretched time
    fitted_data = model(stretched_time, estimated[0], estimated[1], estimated[2])

    # 8. Peak-boost loop
    peak_data = float(np.max(data))  # Use ORIGINAL data, not truncated
    peak_fit = float(np.max(fitted_data))
    thresh = 5.0
    factor = 1.05
    
    while peak_data - peak_fit > thresh:
        estimated[0] *= factor
        fitted_data = model(stretched_time, estimated[0], estimated[1], estimated[2])
        peak_fit = float(np.max(fitted_data))
        factor += 0.05

    logger.debug(f"Final K={estimated[0]:.4f}, alpha={estimated[1]:.4f}, beta={estimated[2]:.4f}")

    # 9. Calculate abs_diff_sum on ORIGINAL sample times
    fitted_at_samples = model(time, estimated[0], estimated[1], estimated[2])
    abs_diff_sum = float(np.sum(np.abs(data - fitted_at_samples)))

    return GammaVariateFitResult(
        fitted_data=fitted_data,
        stretched_time=stretched_time,
        k=float(estimated[0]),
        alpha=float(estimated[1]),
        beta=float(estimated[2]),
        contrast_arrival_idx=closest_idx,
        abs_diff_sum=abs_diff_sum,
        converged=converged,
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