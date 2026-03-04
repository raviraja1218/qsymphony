#!/usr/bin/env python
"""Create publication-quality 3D render for Figure 1a using PyVista"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create 3D figure using matplotlib (more stable than pyvista for now)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Chip substrate (silicon)
x = np.array([-500, 500, 500, -500, -500])
y = np.array([-400, -400, 400, 400, -400])
z = np.array([-50, -50, -50, -50, -50])
ax.plot(x, y, z, 'k-', linewidth=2, alpha=0.5)

# Transmon pads (two spheres approximated as circles)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_pad = 100 * np.outer(np.cos(u), np.sin(v))
y_pad = 100 * np.outer(np.sin(u), np.sin(v))
z_pad = 100 * np.outer(np.ones(np.size(u)), np.cos(v))

# Left pad
ax.plot_surface(x_pad - 100, y_pad, z_pad + 25, color='red', alpha=0.8)
# Right pad
ax.plot_surface(x_pad + 100, y_pad, z_pad + 25, color='red', alpha=0.8)
# Junction (center)
ax.scatter([0], [0], [25], color='blue', s=100, marker='o')

# Coupling capacitor (meandered line)
x_line = np.linspace(200, 400, 50)
y_line = 50 * np.sin(np.linspace(0, 4*np.pi, 50))
z_line = np.ones_like(x_line) * 25
ax.plot(x_line, y_line, z_line, color='green', linewidth=3, label='Coupler')

# Mechanical resonator (HBAR)
x_res = np.linspace(450, 650, 10)
y_res = np.zeros_like(x_res)
z_res = np.linspace(25, 75, 10)
ax.plot(x_res, y_res, z_res, color='purple', linewidth=4, label='Resonator')

# Optical cavity
x_opt = np.linspace(-400, -200, 10)
y_opt = np.linspace(100, 100, 10)
z_opt = np.linspace(25, 55, 10)
ax.plot(x_opt, y_opt, z_opt, color='orange', linewidth=4, label='Optical')

# Labels and formatting
ax.set_xlabel('x (μm)', fontsize=12)
ax.set_ylabel('y (μm)', fontsize=12)
ax.set_zlabel('z (μm)', fontsize=12)
ax.set_title('Optimal Multimode Piezomechanical Chip\nLayout: layout_004034 | Confinement: 98.94%', 
             fontsize=14, fontweight='bold')

# Legend
ax.legend(loc='upper left', bbox_to_anchor=(0.7, 0.9))

# Set view angle for best visualization
ax.view_init(elev=20, azim=45)

# Add text box with parameters
params_text = f"g₀/2π = 11.19 MHz\nQᵢ = 1.20×10⁶\nω_q/2π = 4.753 GHz\nω_m/2π = 492.4 MHz"
ax.text2D(0.02, 0.95, params_text, transform=ax.transAxes,
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
output_dir = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

png_file = output_dir / 'fig1a_3d_render_final.png'
eps_file = output_dir / 'fig1a_3d_render_final.eps'

plt.savefig(png_file, dpi=300, bbox_inches='tight')
plt.savefig(eps_file, format='eps', bbox_inches='tight')
plt.close()

print(f"✅ Figure 1a 3D render saved to: {png_file}")
print(f"✅ Figure 1a EPS saved to: {eps_file}")
