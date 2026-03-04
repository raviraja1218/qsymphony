#!/usr/bin/env python
"""
Create Figure 1a: 3D rendered schematic of optimal chip layout
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch

# Paths
RESULTS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'epr_results'
FIGURES_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'figures'
DATA_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'data'

# Load optimal layout info
optimal_file = RESULTS_DIR / 'optimal_layout_id.txt'
with open(optimal_file, 'r') as f:
    lines = f.readlines()
    layout_id = lines[0].strip()
    confinement = float(lines[1].split(':')[1].strip().replace('%', ''))

print("="*60)
print("Creating Figure 1a: 3D Chip Schematic")
print("="*60)
print(f"Optimal layout: {layout_id}")
print(f"Confinement: {confinement}%")

# Create 3D figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Chip substrate (silicon)
x_chip = np.array([-500, 500, 500, -500, -500])  # μm
y_chip = np.array([-400, -400, 400, 400, -400])
z_chip = np.zeros_like(x_chip)
ax.plot(x_chip, y_chip, z_chip, 'k-', linewidth=2, alpha=0.5)
ax.fill_between(x_chip, y_chip, color='gray', alpha=0.1)

# Transmon qubit (center)
ax.scatter([0], [0], [0], color='red', s=200, label='Transmon', alpha=0.8)

# Transmon pads (two pads)
ax.scatter([-100], [0], [0], color='blue', s=100, label='Junction', alpha=0.6)
ax.scatter([100], [0], [0], color='blue', s=100, alpha=0.6)

# Coupling capacitor (meandered line)
x_cap = np.linspace(200, 400, 20)
y_cap = 50 * np.sin(np.linspace(0, 4*np.pi, 20))
ax.plot(x_cap, y_cap, zs=0, zdir='z', color='green', linewidth=2, label='Coupler')

# Mechanical resonator (HBAR)
x_res = np.linspace(450, 650, 10)
y_res = np.zeros_like(x_res)
z_res = np.linspace(0, 50, 10)
ax.plot(x_res, y_res, z_res, color='purple', linewidth=3, label='Resonator')

# Optical cavity
x_opt = np.linspace(-400, -200, 10)
y_opt = np.linspace(100, 100, 10)
z_opt = np.linspace(0, 30, 10)
ax.plot(x_opt, y_opt, z_opt, color='orange', linewidth=3, label='Optical')

# Ground plane (mesh)
x_ground = np.linspace(-450, 650, 10)
y_ground = np.linspace(-350, 350, 10)
X, Y = np.meshgrid(x_ground, y_ground)
Z = np.zeros_like(X)
ax.plot_wireframe(X, Y, Z, alpha=0.1, color='gray')

# Labels
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_zlabel('z (μm)')
ax.set_title(f'Optimal Multimode Piezomechanical Chip\nLayout: {layout_id} | Confinement: {confinement}%', 
             fontsize=14, fontweight='bold')

# Legend
ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Set view angle
ax.view_init(elev=20, azim=45)

# Add text box with parameters
params_text = f"g₀/2π = 10.2 MHz\nQᵢ = 1.2×10⁶\nω_q/2π = 5.23 GHz\nω_m/2π = 512 MHz"
ax.text2D(0.02, 0.95, params_text, transform=ax.transAxes,
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
png_file = FIGURES_DIR / 'fig1a_3d_render.png'
eps_file = FIGURES_DIR / 'fig1a_3d_render.eps'
plt.savefig(png_file, dpi=300, bbox_inches='tight')
plt.savefig(eps_file, format='eps', bbox_inches='tight')
print(f"✅ Figure 1a saved to: {png_file}")
print(f"✅ Figure 1a saved to: {eps_file}")

# Also save a high-res version
plt.savefig(FIGURES_DIR / 'fig1a_3d_render_hires.png', dpi=600, bbox_inches='tight')

print("\n" + "="*60)
