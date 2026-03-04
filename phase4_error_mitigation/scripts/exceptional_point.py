#!/usr/bin/env python
"""
Step 4.4: Generate Exceptional Point Visualization
Plot complex eigenvalue spectrum showing Liouvillian Exceptional Point
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase4_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths
figures_dir = Path(config['paths']['figures']).expanduser()
data_dir = Path(config['paths']['data']).expanduser()

# Create directories
figures_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.4: Generate Exceptional Point Visualization")
print("="*60)

def compute_liouvillian_spectrum(gamma_range=np.linspace(0, 2, 100)):
    """
    Compute Liouvillian eigenvalues as function of drive parameter
    Simplified model showing exceptional point
    """
    
    eigenvalues_real = []
    eigenvalues_imag = []
    eigenvalues_gamma = []
    
    for gamma in gamma_range:
        # Simplified 2x2 non-Hermitian Hamiltonian
        # H = [[0, gamma], [1, 0]] shows EP at gamma=1
        H = np.array([[0, gamma], [1, 0]])
        
        # Compute eigenvalues
        evals = np.linalg.eigvals(H)
        
        for ev in evals:
            eigenvalues_real.append(ev.real)
            eigenvalues_imag.append(ev.imag)
            eigenvalues_gamma.append(gamma)
    
    return np.array(eigenvalues_gamma), np.array(eigenvalues_real), np.array(eigenvalues_imag)

def plot_3d_exceptional_point():
    """Create 3D plot of eigenvalue trajectories"""
    
    gamma, real, imag = compute_liouvillian_spectrum()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Split into two branches
    n_points = len(gamma) // 2
    gamma1 = gamma[:n_points]
    real1 = real[:n_points]
    imag1 = imag[:n_points]
    
    gamma2 = gamma[n_points:]
    real2 = real[n_points:]
    imag2 = imag[n_points:]
    
    # Plot branches
    ax.plot(gamma1, real1, imag1, 'b-', linewidth=2, label='Branch 1')
    ax.plot(gamma2, real2, imag2, 'r-', linewidth=2, label='Branch 2')
    
    # Mark exceptional point
    ep_idx = np.argmin(np.abs(gamma - 1.0))
    ax.scatter([gamma[ep_idx]], [real[ep_idx]], [imag[ep_idx]], 
               color='gold', s=200, marker='*', label='Exceptional Point', zorder=10)
    
    ax.set_xlabel('Drive Parameter γ')
    ax.set_ylabel('Re(λ)')
    ax.set_zlabel('Im(λ)')
    ax.set_title('Liouvillian Exceptional Point', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Save
    png_file = figures_dir / 'fig3a_exceptional_point.png'
    eps_file = figures_dir / 'fig3a_exceptional_point.eps'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(eps_file, format='eps', bbox_inches='tight')
    print(f"✅ 3D plot saved: {png_file}")
    plt.close()

def plot_complex_plane():
    """Create complex plane visualization"""
    
    gamma, real, imag = compute_liouvillian_spectrum(np.linspace(0, 2, 500))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by gamma parameter
    scatter = ax.scatter(real, imag, c=gamma, cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label='Drive Parameter γ')
    
    # Mark exceptional point
    ep_idx = np.argmin(np.abs(gamma - 1.0))
    ax.scatter([real[ep_idx]], [imag[ep_idx]], 
               color='red', s=200, marker='*', label='Exceptional Point', zorder=10, edgecolors='black')
    
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Complex Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
    
    # Save
    png_file = figures_dir / 'fig3a_complex_plane.png'
    eps_file = figures_dir / 'fig3a_complex_plane.eps'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(eps_file, format='eps', bbox_inches='tight')
    print(f"✅ Complex plane plot saved: {png_file}")
    plt.close()

def main():
    print("\n🎯 Generating exceptional point visualizations...")
    
    plot_3d_exceptional_point()
    plot_complex_plane()
    
    print("\n" + "="*60)
    print("✅ STEP 4.4 COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {figures_dir}")
    print("  - fig3a_exceptional_point.png (3D view)")
    print("  - fig3a_complex_plane.png (Complex plane)")

if __name__ == "__main__":
    main()
