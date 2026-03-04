#!/usr/bin/env python
"""
Step 2.4: Validate System Parameters - FIXED key names
Verify simulation matches theoretical predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle
import json
from datetime import datetime
from scipy.optimize import curve_fit

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mesolve
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    sys.exit(1)

# Paths
traj_dir = Path('~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories').expanduser()
photo_dir = traj_dir / 'photocurrents'
meta_dir = traj_dir / 'metadata'
wigner_dir = Path('~/projects/qsymphony/results/phase2/wigner_baseline').expanduser()
validation_dir = Path('~/projects/qsymphony/results/phase2/validation').expanduser()
figures_dir = Path('~/projects/qsymphony/results/phase2/figures').expanduser()

# Create directories
validation_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 2.4: Validate System Parameters")
print("="*60)

# Load trajectory summary to get parameters
summary_file = meta_dir / 'trajectory_summary.json'
if not summary_file.exists():
    print(f"❌ Summary file not found: {summary_file}")
    print("Please complete Step 2.2 first.")
    sys.exit(1)

with open(summary_file, 'r') as f:
    summary = json.load(f)

print(f"\n📊 Loaded trajectory summary:")
print(f"  {summary['n_trajectories_successful']} successful trajectories")
params = summary['parameters']

# Print all available keys for debugging
print("\n📋 Available parameters:")
for key in params.keys():
    print(f"  - {key}: {params[key]}")

# Map the actual key names to what we expect
# Based on your summary file, the keys might be different
# Let's try to find the correct keys
T1_q_key = next((k for k in params.keys() if 'T1' in k and 'q' in k.lower()), None)
T2_q_key = next((k for k in params.keys() if 'T2' in k and 'q' in k.lower()), None)
T1_m_key = next((k for k in params.keys() if 'T1' in k and 'm' in k.lower()), None)

if T1_q_key:
    T1_q_us = params[T1_q_key]
    print(f"\n✅ Found T1_q: {T1_q_key} = {T1_q_us} μs")
else:
    T1_q_us = 85.0  # Default from Phase 1
    print(f"⚠️ Using default T1_q = {T1_q_us} μs")

if T2_q_key:
    T2_q_us = params[T2_q_key]
    print(f"✅ Found T2_q: {T2_q_key} = {T2_q_us} μs")
else:
    T2_q_us = 45.0  # Default from Phase 1
    print(f"⚠️ Using default T2_q = {T2_q_us} μs")

if T1_m_key:
    T1_m_us = params[T1_m_key]
    print(f"✅ Found T1_m: {T1_m_key} = {T1_m_us} μs")
else:
    T1_m_us = 1200.0  # Default from Phase 1
    print(f"⚠️ Using default T1_m = {T1_m_us} μs")

n_th = params.get('n_th', 0.443)
print(f"  n_th = {n_th:.3f}")

# Load a representative trajectory for fitting
print("\n🔍 Loading trajectory files for analysis...")
traj_files = sorted(traj_dir.glob('trajectory_*.pkl'))
print(f"  Found {len(traj_files)} trajectory files")

# Select first 10 trajectories for statistics
n_trajs_for_fit = min(10, len(traj_files))
traj_data_list = []

for i in range(n_trajs_for_fit):
    with open(traj_files[i], 'rb') as f:
        traj_data_list.append(pickle.load(f))

print(f"  Loaded {n_trajs_for_fit} trajectories for fitting")

# Extract times in μs
times_us = traj_data_list[0]['times'] * 1e6

# ============================================
# 1. Verify Qubit T₁ (energy relaxation)
# ============================================
print("\n" + "="*60)
print("1. Verifying Qubit T₁ (Energy Relaxation)")
print("="*60)

# Collect qubit populations from all trajectories
n_q_all = np.array([traj['n_q'] for traj in traj_data_list])
n_q_mean = np.mean(n_q_all, axis=0)
n_q_std = np.std(n_q_all, axis=0)

# Fit exponential decay: n_q(t) = n_q(0) * exp(-t/T₁)
def exp_decay(t, n0, T1):
    return n0 * np.exp(-t / T1)

# Fit from t>0 to avoid t=0 point
fit_mask = times_us > 1.0  # Start after 1 μs
try:
    popt, pcov = curve_fit(exp_decay, times_us[fit_mask], n_q_mean[fit_mask], 
                           p0=[n_q_mean[0], T1_q_us])
    T1_fit = popt[1]
    T1_err = np.sqrt(pcov[1,1])
    
    print(f"\n📈 Fitted T₁ = {T1_fit:.2f} ± {T1_err:.2f} μs")
    print(f"   Expected T₁ = {T1_q_us} μs")
    print(f"   Relative error: {abs(T1_fit - T1_q_us)/T1_q_us*100:.2f}%")
    
    T1_match = abs(T1_fit - T1_q_us) / T1_q_us < 0.05
    print(f"   Within 5%: {'✅ YES' if T1_match else '❌ NO'}")
    
except Exception as e:
    print(f"❌ Fit failed: {e}")
    T1_match = False
    T1_fit = None

# Plot qubit decay
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(times_us, n_q_mean, yerr=n_q_std, fmt='o', markersize=3, 
            capsize=2, alpha=0.5, label='Data (mean ± std)')
if T1_fit:
    t_fit = np.linspace(0, times_us[-1], 100)
    ax.plot(t_fit, exp_decay(t_fit, n_q_mean[0], T1_fit), 'r-', linewidth=2, 
            label=f'Fit: T₁ = {T1_fit:.1f} μs')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (μs)', fontsize=12)
ax.set_ylabel('⟨n_q⟩', fontsize=12)
ax.set_title('Qubit Population Decay - T₁ Verification', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(validation_dir / 't1_verification.png', dpi=150, bbox_inches='tight')
plt.savefig(figures_dir / 't1_verification.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ T₁ verification plot saved")

# ============================================
# 2. Verify Qubit T₂* (dephasing)
# ============================================
print("\n" + "="*60)
print("2. Verifying Qubit T₂* (Dephasing)")
print("="*60)

# For T₂*, we need coherence |ρ₀₁|. Since we only have populations,
# we'll estimate from the decay of correlations
corr_all = np.array([traj['correlation'] for traj in traj_data_list])
corr_mean = np.mean(corr_all, axis=0)
corr_std = np.std(corr_all, axis=0)

# Coherence should decay as exp(-t/T₂)
def exp_decay_coherence(t, c0, T2):
    return c0 * np.exp(-t / T2)

try:
    popt_c, pcov_c = curve_fit(exp_decay_coherence, times_us[fit_mask], 
                                np.abs(corr_mean[fit_mask]),
                                p0=[np.abs(corr_mean[0]), T2_q_us])
    T2_fit = popt_c[1]
    T2_err = np.sqrt(pcov_c[1,1])
    
    print(f"\n📈 Fitted T₂* = {T2_fit:.2f} ± {T2_err:.2f} μs")
    print(f"   Expected T₂* = {T2_q_us} μs")
    print(f"   Relative error: {abs(T2_fit - T2_q_us)/T2_q_us*100:.2f}%")
    
    T2_match = abs(T2_fit - T2_q_us) / T2_q_us < 0.05
    print(f"   Within 5%: {'✅ YES' if T2_match else '❌ NO'}")
    
except Exception as e:
    print(f"❌ Fit failed: {e}")
    T2_match = False
    T2_fit = None

# Plot coherence decay
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(times_us, np.abs(corr_mean), yerr=corr_std, fmt='o', markersize=3,
            capsize=2, alpha=0.5, label='Data (mean ± std)')
if T2_fit:
    t_fit = np.linspace(0, times_us[-1], 100)
    ax.plot(t_fit, exp_decay_coherence(t_fit, np.abs(corr_mean[0]), T2_fit), 'r-', linewidth=2,
            label=f'Fit: T₂* = {T2_fit:.1f} μs')
ax.set_xlabel('Time (μs)', fontsize=12)
ax.set_ylabel('|⟨a†b + a b†⟩|', fontsize=12)
ax.set_title('Coherence Decay - T₂* Verification', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(validation_dir / 't2_verification.png', dpi=150, bbox_inches='tight')
plt.savefig(figures_dir / 't2_verification.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ T₂* verification plot saved")

# ============================================
# 3. Verify Mechanical T₁
# ============================================
print("\n" + "="*60)
print("3. Verifying Mechanical T₁")
print("="*60)

# Collect mechanical populations
n_m_all = np.array([traj['n_m'] for traj in traj_data_list])
n_m_mean = np.mean(n_m_all, axis=0)
n_m_std = np.std(n_m_all, axis=0)

# Since mechanical mode starts at 0 and thermalizes to n_th,
# the evolution is: n_m(t) = n_th * (1 - exp(-t/T₁_m))
def mech_thermalization(t, n_th_eq, T1_m):
    return n_th_eq * (1 - np.exp(-t / T1_m))

try:
    popt_m, pcov_m = curve_fit(mech_thermalization, times_us[fit_mask], n_m_mean[fit_mask],
                               p0=[n_th, T1_m_us])
    T1_m_fit = popt_m[1]
    T1_m_err = np.sqrt(pcov_m[1,1])
    n_th_fit = popt_m[0]
    
    print(f"\n📈 Fitted T₁_m = {T1_m_fit:.2f} ± {T1_m_err:.2f} μs")
    print(f"   Expected T₁_m = {T1_m_us} μs")
    print(f"   Relative error: {abs(T1_m_fit - T1_m_us)/T1_m_us*100:.2f}%")
    print(f"   Fitted n_th = {n_th_fit:.3f} (expected {n_th:.3f})")
    
    T1_m_match = abs(T1_m_fit - T1_m_us) / T1_m_us < 0.05
    print(f"   Within 5%: {'✅ YES' if T1_m_match else '❌ NO'}")
    
except Exception as e:
    print(f"❌ Fit failed: {e}")
    T1_m_match = False
    T1_m_fit = None

# Plot mechanical thermalization
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(times_us, n_m_mean, yerr=n_m_std, fmt='o', markersize=3,
            capsize=2, alpha=0.5, label='Data (mean ± std)')
if T1_m_fit:
    t_fit = np.linspace(0, times_us[-1], 100)
    ax.plot(t_fit, mech_thermalization(t_fit, n_th, T1_m_fit), 'r-', linewidth=2,
            label=f'Fit: T₁_m = {T1_m_fit:.1f} μs')
ax.axhline(y=n_th, color='g', linestyle='--', alpha=0.5, 
           label=f'n_th = {n_th:.3f}')
ax.set_xlabel('Time (μs)', fontsize=12)
ax.set_ylabel('⟨n_m⟩', fontsize=12)
ax.set_title('Mechanical Thermalization - T₁ Verification', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(validation_dir / 't1_mech_verification.png', dpi=150, bbox_inches='tight')
plt.savefig(figures_dir / 't1_mech_verification.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Mechanical T₁ verification plot saved")

# ============================================
# 4. Check Numerical Stability
# ============================================
print("\n" + "="*60)
print("4. Checking Numerical Stability")
print("="*60)

# Check for negative probabilities
n_q_min = np.min(n_q_all)
n_m_min = np.min(n_m_all)

print(f"\n📊 Minimum values:")
print(f"  ⟨n_q⟩ min: {n_q_min:.6f}")
print(f"  ⟨n_m⟩ min: {n_m_min:.6f}")

if n_q_min >= -1e-10 and n_m_min >= -1e-10:
    print("  ✅ No negative probabilities")
    stability_pass = True
else:
    print("  ⚠️ Negative values detected")
    stability_pass = False

# Check for rapid oscillations (smoothness)
n_q_diff = np.diff(n_q_mean)
n_m_diff = np.diff(n_m_mean)
max_diff_q = np.max(np.abs(n_q_diff))
max_diff_m = np.max(np.abs(n_m_diff))

print(f"\n📈 Maximum changes between time steps:")
print(f"  Max Δ⟨n_q⟩: {max_diff_q:.6f}")
print(f"  Max Δ⟨n_m⟩: {max_diff_m:.6f}")

if max_diff_q < 0.1 and max_diff_m < 0.1:
    print("  ✅ Smooth evolution (no rapid oscillations)")
else:
    print("  ⚠️ Possible numerical instabilities detected")

# ============================================
# 5. Generate Validation Report
# ============================================
print("\n" + "="*60)
print("5. Generating Validation Report")
print("="*60)

# Compile all results
validation_results = {
    'date': datetime.now().isoformat(),
    'parameters': {
        'T1_q_us': T1_q_us,
        'T2_q_us': T2_q_us,
        'T1_m_us': T1_m_us,
        'n_th': n_th
    },
    'qubit_T1': {
        'expected_us': T1_q_us,
        'fitted_us': float(T1_fit) if T1_fit else None,
        'error_percent': float(abs(T1_fit - T1_q_us)/T1_q_us*100) if T1_fit else None,
        'within_5_percent': bool(T1_match) if 'T1_match' in locals() else False
    },
    'qubit_T2': {
        'expected_us': T2_q_us,
        'fitted_us': float(T2_fit) if T2_fit else None,
        'error_percent': float(abs(T2_fit - T2_q_us)/T2_q_us*100) if T2_fit else None,
        'within_5_percent': bool(T2_match) if 'T2_match' in locals() else False
    },
    'mechanical_T1': {
        'expected_us': T1_m_us,
        'fitted_us': float(T1_m_fit) if T1_m_fit else None,
        'error_percent': float(abs(T1_m_fit - T1_m_us)/T1_m_us*100) if T1_m_fit else None,
        'within_5_percent': bool(T1_m_match) if 'T1_m_match' in locals() else False
    },
    'numerical_stability': {
        'min_n_q': float(n_q_min),
        'min_n_m': float(n_m_min),
        'max_diff_q': float(max_diff_q),
        'max_diff_m': float(max_diff_m),
        'stable': bool(stability_pass and max_diff_q < 0.1 and max_diff_m < 0.1)
    }
}

# Determine overall status
all_passed = (
    validation_results['qubit_T1']['within_5_percent'] and
    validation_results['qubit_T2']['within_5_percent'] and
    validation_results['mechanical_T1']['within_5_percent'] and
    validation_results['numerical_stability']['stable']
)
validation_results['overall_status'] = 'PASS' if all_passed else 'FAIL'

# Save validation report
report_file = validation_dir / 'validation_report.json'
with open(report_file, 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f"\n📊 Validation report saved to: {report_file}")

# Create human-readable report
report_txt = f"""
VALIDATION REPORT - Phase 2.4
Date: {validation_results['date']}
============================================================

1. Qubit T₁ (Energy Relaxation)
   Expected: {T1_q_us} μs
   Fitted:   {validation_results['qubit_T1']['fitted_us']:.2f} μs
   Error:    {validation_results['qubit_T1']['error_percent']:.2f}%
   Status:   {'✅ PASS' if validation_results['qubit_T1']['within_5_percent'] else '❌ FAIL'}

2. Qubit T₂* (Dephasing)
   Expected: {T2_q_us} μs
   Fitted:   {validation_results['qubit_T2']['fitted_us']:.2f} μs
   Error:    {validation_results['qubit_T2']['error_percent']:.2f}%
   Status:   {'✅ PASS' if validation_results['qubit_T2']['within_5_percent'] else '❌ FAIL'}

3. Mechanical T₁
   Expected: {T1_m_us} μs
   Fitted:   {validation_results['mechanical_T1']['fitted_us']:.2f} μs
   Error:    {validation_results['mechanical_T1']['error_percent']:.2f}%
   Status:   {'✅ PASS' if validation_results['mechanical_T1']['within_5_percent'] else '❌ FAIL'}

4. Numerical Stability
   Min ⟨n_q⟩: {validation_results['numerical_stability']['min_n_q']:.6f}
   Min ⟨n_m⟩: {validation_results['numerical_stability']['min_n_m']:.6f}
   Max Δ⟨n_q⟩: {validation_results['numerical_stability']['max_diff_q']:.6f}
   Max Δ⟨n_m⟩: {validation_results['numerical_stability']['max_diff_m']:.6f}
   Status:   {'✅ STABLE' if validation_results['numerical_stability']['stable'] else '⚠️ UNSTABLE'}

============================================================
OVERALL STATUS: {validation_results['overall_status']}
============================================================
"""

report_txt_file = validation_dir / 'validation_report.txt'
with open(report_txt_file, 'w') as f:
    f.write(report_txt)

print(report_txt)
print(f"\n📄 Human-readable report saved to: {report_txt_file}")

# Summary
print("\n" + "="*60)
print("STEP 2.4 COMPLETE")
print("="*60)
print(f"✅ Validation plots generated:")
print(f"  - {validation_dir}/t1_verification.png")
print(f"  - {validation_dir}/t2_verification.png")
print(f"  - {validation_dir}/t1_mech_verification.png")
print(f"✅ Validation reports saved")
print(f"\nNext: Step 2.5 - Prepare RL Environment Interface")
print("="*60)
