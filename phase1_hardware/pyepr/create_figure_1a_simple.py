#!/usr/bin/env python
"""Simple 2D version of Figure 1a"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

FIGURES_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'figures'

fig, ax = plt.subplots(figsize=(10, 8))

# Chip substrate
ax.add_patch(patches.Rectangle((-500, -400), 1000, 800, 
                               facecolor='lightgray', edgecolor='black', alpha=0.3))

# Transmon
ax.add_patch(patches.Circle((0, 0), 50, facecolor='red', edgecolor='black', label='Transmon'))
ax.add_patch(patches.Circle((-100, 0), 20, facecolor='blue', edgecolor='black'))
ax.add_patch(patches.Circle((100, 0), 20, facecolor='blue', edgecolor='black'))

# Coupler
x = range(200, 401, 20)
y = [50 * (1 if i%2==0 else -1) for i in range(len(x))]
ax.plot(x, y, 'g-', linewidth=2, label='Coupler')

# Resonator
ax.add_patch(patches.Rectangle((450, -50), 200, 100, 
                               facecolor='purple', edgecolor='black', alpha=0.7, label='Resonator'))

# Optical cavity
ax.add_patch(patches.Rectangle((-400, 50), 150, 30, 
                               facecolor='orange', edgecolor='black', alpha=0.7, label='Optical'))

ax.set_xlim(-600, 700)
ax.set_ylim(-450, 450)
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Optimal Multimode Piezomechanical Chip\nLayout: layout_004034 | Confinement: 98.94%')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1a_chip_layout.png', dpi=300, bbox_inches='tight')
print(f"✅ Figure 1a saved to: {FIGURES_DIR / 'fig1a_chip_layout.png'}")
