"""
Debug script for curve fitting issues.

Run this to see what's happening with your data:
    python3 debug_curve_fitting.py
"""

import numpy as np

# Example data that should work
sample_time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])
sample_curve = np.array([50, 52, 54, 165, 285, 245, 190, 145, 115, 105, 100, 98, 96])

print("=" * 60)
print("DEBUGGING CURVE FITTING")
print("=" * 60)

print("\nOriginal Data:")
print(f"Time:   {sample_time}")
print(f"Signal: {sample_curve}")
print(f"Peak signal: {np.max(sample_curve)} at index {np.argmax(sample_curve)}")

# Simulate baseline subtraction
baseline_position = 2  # 1-based, so this means indices 0-1
washout = 8  # 1-based, point 8

print(f"\nBaseline position: {baseline_position} (points 1-{baseline_position})")
print(f"Washout point: {washout}")

# Calculate baseline value (average of first N points)
baseline_idx_0based = baseline_position - 1  # Convert to 0-based
baseline_value = np.mean(sample_curve[:baseline_position])
print(f"\nBaseline value (mean of first {baseline_position} points): {baseline_value:.2f}")

# Subtract baseline
baseline_subtracted = sample_curve - baseline_value
print(f"\nBaseline-subtracted curve: {baseline_subtracted}")
print(f"Peak after subtraction: {np.max(baseline_subtracted):.2f}")

# Points to fit (up to washout)
if washout > 0:
    fit_indices = slice(0, washout)  # 0-based, so washout=8 means indices 0-7
    print(f"\nFitting points 1-{washout} (indices 0-{washout-1})")
else:
    fit_indices = slice(None)
    print(f"\nFitting all points")

time_to_fit = sample_time[fit_indices]
curve_to_fit = baseline_subtracted[fit_indices]

print(f"Time to fit:   {time_to_fit}")
print(f"Curve to fit:  {curve_to_fit}")

# Find contrast arrival (first point after baseline where signal increases significantly)
contrast_arrival_idx = baseline_position  # 1-based
t_at = sample_time[contrast_arrival_idx - 1]  # Convert to 0-based for indexing
print(f"\nContrast arrival time (t_AT): {t_at:.2f}s (at 1-based index {contrast_arrival_idx})")

# Estimate initial parameters
peak_idx = np.argmax(baseline_subtracted)
peak_value = baseline_subtracted[peak_idx]
t_peak = sample_time[peak_idx]

K_init = peak_value
alpha_init = 2.0
beta_init = (t_peak - t_at) / (alpha_init + 1.0)

print(f"\nInitial parameters:")
print(f"  K (amplitude):     {K_init:.2f}")
print(f"  α (alpha):         {alpha_init:.2f}")
print(f"  β (beta):          {beta_init:.2f}")
print(f"  t_AT:              {t_at:.2f}s")

# Test gamma variate function
def gamma_variate(t, K, alpha, beta, t_at):
    """Modified gamma variate function."""
    dt = np.maximum(t - t_at, 0.0)
    result = np.zeros_like(t, dtype=float)
    mask = dt > 0
    if np.any(mask):
        result[mask] = K * (dt[mask] ** alpha) * np.exp(-dt[mask] / beta)
    return result

# Evaluate at sample points
fitted_at_samples = gamma_variate(time_to_fit, K_init, alpha_init, beta_init, t_at)
print(f"\nFitted curve at sample points (initial params):")
print(f"  {fitted_at_samples}")

# Calculate initial error
errors = curve_to_fit - fitted_at_samples
rmse_initial = np.sqrt(np.mean(errors ** 2))
print(f"\nInitial RMSE: {rmse_initial:.2f}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

if K_init < 10:
    print("⚠️  WARNING: K is very small (< 10)")
    print("   → This suggests baseline subtraction removed too much signal")
    print("   → Check that baseline_value is correct")
    
if K_init < 1:
    print("⛔ CRITICAL: K < 1 - curve fitting will fail!")
    print("   → Baseline subtraction is definitely wrong")
    
if rmse_initial > 50:
    print("⚠️  WARNING: High initial RMSE")
    print("   → Initial parameters may need adjustment")
    
if beta_init < 0.1:
    print("⚠️  WARNING: β is very small")
    print("   → Check t_AT calculation")

print("\n" + "=" * 60)
print("RECOMMENDED FIXES:")
print("=" * 60)

print("\n1. Check baseline calculation:")
print(f"   Current: mean of points 1-{baseline_position} = {baseline_value:.2f}")
print(f"   Points used: {sample_curve[:baseline_position]}")

print("\n2. Verify washout point:")
print(f"   Current: point {washout} (value = {sample_curve[washout-1]:.2f})")
print(f"   Points excluded: {sample_curve[washout:]}")

print("\n3. Check if baseline_override is being used correctly:")
print("   It should be the NUMBER of baseline points (1-based)")
print("   NOT the index of the last baseline point")

# Show what the correct implementation should look like
print("\n" + "=" * 60)
print("CORRECT IMPLEMENTATION:")
print("=" * 60)

print("""
def preprocess_curve(raw_tdc, time_points, baseline_override, washout_point):
    # baseline_override is 1-based COUNT of baseline points
    # e.g., baseline_override=2 means use points 1 and 2 (indices 0 and 1)
    
    baseline_value = np.mean(raw_tdc[:baseline_override])
    subtracted = raw_tdc - baseline_value
    
    # washout_point is 1-based index
    # e.g., washout_point=8 means fit up to point 8 (index 7)
    if washout_point and washout_point > 0:
        recirculation_start = washout_point  # This is the count of points to use
    else:
        recirculation_start = len(raw_tdc)
    
    return {
        'subtracted_curve': subtracted,
        'baseline_position': baseline_override - 1,  # Convert to 0-based
        'recirculation_start': recirculation_start
    }

def fit_curve(sample_curve, sample_time, baseline, washout):
    # baseline and washout are 1-based counts/positions from UI
    
    processed = preprocess_curve(sample_curve, sample_time, baseline, washout)
    
    fit_modified_gamma_variate(
        time=sample_time,
        data=processed['subtracted_curve'],
        K_init=np.max(processed['subtracted_curve']),
        alpha_init=2.0,
        beta_init=1.5,
        num_points_to_consider=processed['recirculation_start'],
        contrast_arrival_time=baseline  # This is 1-based position
    )
""")