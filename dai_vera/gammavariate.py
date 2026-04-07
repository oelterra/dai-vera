"""
gamma_variate.py
----------------
fitModifiedGammaVariate.
Model:  ct(t) = K * (t - t_AT)^alpha * exp(-(t - t_AT) / beta)
        where t_AT = contrast arrival time, and (t - t_AT) is clamped to 0.

Constraint approach:
  - First, fit (optionally weighted) to get K, alpha, beta.
  - If constraint_points exist, LOCK K and alpha, sweep beta over a
    fine grid, pick the beta that minimizes constraint error while
    keeping the peak within tolerance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

logger = logging.getLogger(__name__)


@dataclass
class GammaVariateFitResult:
    fitted_data:          np.ndarray
    stretched_time:       np.ndarray
    k:                    float
    alpha:                float
    beta:                 float
    contrast_arrival_idx: int
    abs_diff_sum:         float
    converged:            bool = True


def fit_modified_gamma_variate(
    time: np.ndarray,
    data: np.ndarray,
    K_init: float,
    alpha_init: float,
    beta_init: float,
    num_points_to_consider: int,
    contrast_arrival_time: int,
    constraint_points: list = None,
    weights: np.ndarray = None,
) -> GammaVariateFitResult:

    time = np.asarray(time, dtype=float).flatten()
    data = np.asarray(data, dtype=float).flatten()

    if num_points_to_consider > 0:
        times = time[:num_points_to_consider]
        datas = data[:num_points_to_consider]
    else:
        times = time.copy()
        datas = data.copy()

    cat_idx_0 = max(0, min(int(contrast_arrival_time) - 1, len(times) - 1))
    t_at = float(times[cat_idx_0])

    def model(t, K, alpha, beta):
        beta = max(beta, 1e-6)
        tAT = t - t_at
        tAT = np.where(tAT < 0, 0, tAT)
        tAT_safe = np.where(tAT == 0, 1e-12, tAT)
        return K * (tAT_safe ** alpha) * np.exp(np.clip(-tAT / beta, -100, 100))

    p0 = [K_init, alpha_init, beta_init]
    converged = True

    # ── Build sigma (inverse weights) for curve_fit ───────────────────
    #    sigma[i] = 1/w[i]  →  higher weight = lower sigma = tighter fit
    sigma = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).flatten()
        if num_points_to_consider > 0:
            w = w[:num_points_to_consider]
        w = np.maximum(w, 0.1)
        sigma = 1.0 / w

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            fit_kwargs = dict(
                f=model,
                xdata=times,
                ydata=datas,
                p0=p0,
                bounds=([0, 0, 1e-6], [np.inf, 10, np.inf]),
                maxfev=3000,
                ftol=1e-10,
                xtol=1e-10,
            )
            if sigma is not None:
                fit_kwargs["sigma"] = sigma
                fit_kwargs["absolute_sigma"] = False
            estimated, _ = curve_fit(**fit_kwargs)
    except (OptimizeWarning, RuntimeError) as exc:
        logger.warning(f"curve_fit did not converge: {exc}")
        estimated = np.array(p0, dtype=float)
        converged = False

    K_fit     = float(estimated[0])
    alpha_fit = float(estimated[1])
    beta_fit  = float(estimated[2])

    stretched_time = np.linspace(float(time[0]), float(time[-1]), 100)
    closest_idx = int(np.argmin(np.abs(stretched_time - t_at)))
    fitted_data = model(stretched_time, K_fit, alpha_fit, beta_fit)
    unconstrained_peak = float(np.max(fitted_data))

    # ───────────────────────────────────────────────────────────────────
    # Constraint sweep: adjust beta to pass through the washout point
    # while keeping the peak within tolerance.
    #
    # Pass 1: 15% peak tolerance
    # Pass 2: 25% peak tolerance (fallback)
    # ───────────────────────────────────────────────────────────────────
    if constraint_points:
        valid_constraints = [
            (t_c, y_c) for t_c, y_c in constraint_points
            if t_c > t_at and y_c >= 0
        ]

        if valid_constraints:
            beta_original = beta_fit
            n_sweep = 3000
            beta_lo = beta_original * 0.15
            beta_hi = beta_original * 4.0
            betas = np.linspace(beta_lo, beta_hi, n_sweep)

            found = False
            for peak_tol in (0.15, 0.25):
                best_beta = beta_original
                best_constraint_err = float('inf')

                for b_candidate in betas:
                    candidate_curve = model(stretched_time, K_fit, alpha_fit, b_candidate)
                    candidate_peak = float(np.max(candidate_curve))

                    if unconstrained_peak > 0:
                        peak_deviation = abs(candidate_peak - unconstrained_peak) / unconstrained_peak
                        if peak_deviation > peak_tol:
                            continue

                    c_err = 0.0
                    for t_c, y_c in valid_constraints:
                        y_pred = float(model(np.array([t_c]), K_fit, alpha_fit, b_candidate)[0])
                        c_err += (y_pred - y_c) ** 2

                    if c_err < best_constraint_err:
                        best_constraint_err = c_err
                        best_beta = b_candidate

                if best_beta != beta_original:
                    beta_fit = best_beta
                    fitted_data = model(stretched_time, K_fit, alpha_fit, beta_fit)
                    new_peak = float(np.max(fitted_data))
                    print(f"  [CONSTRAINT] Beta: {beta_original:.4f} → {beta_fit:.4f} "
                          f"(peak: {unconstrained_peak:.1f} → {new_peak:.1f}, "
                          f"tol={peak_tol*100:.0f}%)")
                    for t_c, y_c in valid_constraints:
                        y_actual = float(model(np.array([t_c]), K_fit, alpha_fit, beta_fit)[0])
                        print(f"    target y={y_c:.1f} at t={t_c:.2f} → "
                              f"fitted y={y_actual:.1f} (err={abs(y_actual - y_c):.1f})")
                    found = True
                    break
                else:
                    print(f"  [CONSTRAINT] No beta found at {peak_tol*100:.0f}% tolerance, "
                          f"trying wider …")

            if not found:
                print(f"  [CONSTRAINT] Could not satisfy constraint at any tolerance. "
                      f"Keeping unconstrained fit.")

    fitted_at_samples = model(time, K_fit, alpha_fit, beta_fit)
    abs_diff_sum = float(np.sum(np.abs(data - fitted_at_samples)))

    return GammaVariateFitResult(
        fitted_data=fitted_data,
        stretched_time=stretched_time,
        k=K_fit,
        alpha=alpha_fit,
        beta=beta_fit,
        contrast_arrival_idx=closest_idx,
        abs_diff_sum=abs_diff_sum,
        converged=converged,
    )


def compute_auc(fitted_data: np.ndarray, stretched_time: np.ndarray) -> float:
    return float(np.trapezoid(fitted_data, stretched_time))


def get_default_initial_params(
    tdc: np.ndarray,
    time: np.ndarray,
    contrast_arrival_idx: int,
) -> tuple[float, float, float]:
    tdc  = np.asarray(tdc,  dtype=float).flatten()
    time = np.asarray(time, dtype=float).flatten()
    peak_idx = int(np.argmax(tdc))
    peak_hu  = float(tdc[peak_idx])
    t_at     = float(time[max(0, contrast_arrival_idx)])
    t_peak   = float(time[peak_idx])
    K     = max(peak_hu, 1.0)
    alpha = 2.0
    beta  = max((t_peak - t_at) / (alpha + 1.0), 0.1)
    return K, alpha, beta