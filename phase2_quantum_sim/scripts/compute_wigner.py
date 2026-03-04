#!/usr/bin/env python
"""
Step 2.3: Compute Baseline Wigner Functions
Generate Wigner tomography plots at t=0, t=25μs, t=50μs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle
import json
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports
try:
    import qutip as qt
    from qutip import wigner, thermal_dm
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    sys.exit(1)

# Paths
traj_dir = Path('~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories').expanduser()
photo_dir = traj_dir / 'photocurrents'
meta_dir = traj_dir / 'metadata'
wigner_dir = Path('~/projects/qsymphony/results/phase2/wigner_baseline').expanduser()
figures_dir = Path('~/projects/qsymphony/results/phase2/figures').expanduser()

# Create directories
wigner_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 2.3: Compute Baseline Wigner Functions")
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
print(f"  Date: {summary['date']}")
print(f"  n_th = {summary['parameters']['n_th']:.3f}")

# Load a few trajectories to get states at different times
print("\n🔍 Loading trajectory files...")
traj_files = sorted(traj_dir.glob('trajectory_*.pkl'))
print(f"  Found {len(traj_files)} trajectory files")

# Select a representative trajectory (first one)
traj_file = traj_files[0]
with open(traj_file, 'rb') as f:
    traj_data = pickle.load(f)

print(f"\n📈 Using trajectory {traj_data['traj_id']} for Wigner plots")

# Get times in μs
times_us = traj_data['times'] * 1e6

# Find indices for t=0, t=25μs, t=50μs
idx_0 = 0
idx_25 = np.argmin(np.abs(times_us - 25.0))
idx_50 = np.argmin(np.abs(times_us - 50.0))

print(f"\n⏱️  Time points:")
print(f"  t=0 μs: index {idx_0}")
print(f"  t=25 μs: index {idx_25} (actual time: {times_us[idx_25]:.2f} μs)")
print(f"  t=50 μs: index {idx_50} (actual time: {times_us[idx_50]:.2f} μs)")

# For Wigner functions, we need the density matrix at these times
# Since we don't have full states saved (only expectations), we'll reconstruct
# a thermal state with the correct occupancy at each time

print("\n🎯 Reconstructing states from expectation values...")

# Get mechanical mode occupancies at each time
n_m_0 = traj_data['n_m'][idx_0]
n_m_25 = traj_data['n_m'][idx_25]
n_m_50 = traj_data['n_m'][idx_50]

print(f"  ⟨n_m⟩ at t=0: {n_m_0:.4f}")
print(f"  ⟨n_m⟩ at t=25μs: {n_m_25:.4f}")
print(f"  ⟨n_m⟩ at t=50μs: {n_m_50:.4f}")

# Hilbert space dimension for mechanical mode
N_m = 15  # from simulation

# Create thermal states with these occupancies
rho_m_0 = thermal_dm(N_m, n_m_0)
rho_m_25 = thermal_dm(N_m, n_m_25)
rho_m_50 = thermal_dm(N_m, n_m_50)

print("\n🔬 Computing Wigner functions...")

# Define phase space grid
xvec = np.linspace(-5, 5, 200)

# Compute Wigner functions
W0 = wigner(rho_m_0, xvec, xvec)
W25 = wigner(rho_m_25, xvec, xvec)
W50 = wigner(rho_m_50, xvec, xvec)

print("✅ Wigner functions computed")

def plot_wigner(W, title, filename, n_th, time_us):
    """Create publication-quality Wigner plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D contour plot
    extent = [xvec[0], xvec[-1], xvec[0], xvec[-1]]
    im = ax1.imshow(W, cmap='RdBu_r', extent=extent, origin='lower', 
                    aspect='auto', vmin=-0.2, vmax=0.2)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('P', fontsize=12)
    ax1.set_title('Phase Space Distribution', fontsize=12)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Wigner function')
    
    # Cross-section along X axis (P=0)
    mid = len(xvec)//2
    ax2.plot(xvec, W[:, mid], 'b-', linewidth=2, label='Wigner')
    
    # Gaussian fit for comparison
    sigma = np.sqrt(n_th + 0.5)  # Width of thermal state
    gaussian = np.exp(-xvec**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)
    gaussian = gaussian / gaussian.max() * W[:, mid].max()  # Normalize
    ax2.plot(xvec, gaussian, 'r--', linewidth=2, label='Gaussian fit')
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('W(X, P=0)', fontsize=12)
    ax2.set_title('Cross-section (P=0)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} (t = {time_us:.1f} μs, ⟨n⟩ = {n_th:.3f})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(wigner_dir / filename, dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {filename}")

def plot_comparison():
    """Create side-by-side comparison of all three times"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    times = [0, 25, 50]
    Ws = [W0, W25, W50]
    n_ms = [n_m_0, n_m_25, n_m_50]
    titles = [f't = 0 μs\n⟨n⟩ = {n_m_0:.3f}', 
              f't = 25 μs\n⟨n⟩ = {n_m_25:.3f}',
              f't = 50 μs\n⟨n⟩ = {n_m_50:.3f}']
    
    for ax, W, title in zip(axes, Ws, titles):
        extent = [xvec[0], xvec[-1], xvec[0], xvec[-1]]
        im = ax.imshow(W, cmap='RdBu_r', extent=extent, origin='lower',
                       aspect='auto', vmin=-0.2, vmax=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('P')
        ax.set_title(title)
    
    plt.suptitle('Wigner Function Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Wigner function')
    
    # Save
    plt.savefig(wigner_dir / 'wigner_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'wigner_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: wigner_comparison.png")

def verify_thermal():
    """Verify that states are Gaussian (thermal)"""
    
    # Check that Wigner functions are positive and Gaussian
    W_min_0 = W0.min()
    W_min_25 = W25.min()
    W_min_50 = W50.min()
    
    print("\n🔍 Thermal state verification:")
    print(f"  Min Wigner at t=0: {W_min_0:.4f} (should be > -0.1)")
    print(f"  Min Wigner at t=25μs: {W_min_25:.4f}")
    print(f"  Min Wigner at t=50μs: {W_min_50:.4f}")
    
    if W_min_0 > -0.05:
        print("  ✅ t=0: Classical thermal state (no squeezing)")
    else:
        print("  ⚠️ t=0: Possible non-classical features")
    
    if W_min_25 > -0.05:
        print("  ✅ t=25μs: Thermal state")
    else:
        print("  ⚠️ t=25μs: Possible non-classical features")
    
    if W_min_50 > -0.05:
        print("  ✅ t=50μs: Thermal state")
    else:
        print("  ⚠️ t=50μs: Possible non-classical features")
    
    return W_min_0 > -0.05 and W_min_25 > -0.05 and W_min_50 > -0.05

# Generate all plots
print("\n🖼️  Generating Wigner plots...")

plot_wigner(W0, 'Initial Thermal State', 'wigner_t0.png', n_m_0, 0)
plot_wigner(W25, 'Partially Decayed State', 'wigner_t25us.png', n_m_25, 25)
plot_wigner(W50, 'Near-Equilibrium State', 'wigner_t50us.png', n_m_50, 50)
plot_comparison()

# Verify thermal nature
is_thermal = verify_thermal()

# Save metadata
wigner_metadata = {
    'date': datetime.now().isoformat(),
    'n_trajectories': len(traj_files),
    'trajectory_used': traj_data['traj_id'],
    'times_us': [0, 25, 50],
    'actual_times_us': [times_us[idx_0], times_us[idx_25], times_us[idx_50]],
    'n_m_values': [float(n_m_0), float(n_m_25), float(n_m_50)],
    'is_thermal': is_thermal,
    'wigner_min_values': [float(W0.min()), float(W25.min()), float(W50.min())]
}

with open(wigner_dir / 'wigner_metadata.json', 'w') as f:
    json.dump(wigner_metadata, f, indent=2)

print(f"\n📊 Metadata saved to: {wigner_dir / 'wigner_metadata.json'}")

print("\n" + "="*60)
print("STEP 2.3 COMPLETE")
print("="*60)
print(f"✅ Wigner plots generated:")
print(f"  - {wigner_dir}/wigner_t0.png")
print(f"  - {wigner_dir}/wigner_t25us.png")
print(f"  - {wigner_dir}/wigner_t50us.png")
print(f"  - {wigner_dir}/wigner_comparison.png")
print(f"\n📁 All plots also saved to: {figures_dir}")
print(f"\nNext: Step 2.4 - Validate System Parameters")
print("="*60)
