#!/usr/bin/env python
"""
Update Phase 4 summary figure with new noise results
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

fig_dir = Path('results/phase4/figures')

# Create 2x2 summary figure with updated plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Check which files exist and load them
plots = {
    'circuit': fig_dir / 'fig3b_circuit_depth.png',
    'exceptional': fig_dir / 'fig3a_exceptional_point.png',
    'noise': fig_dir / 'pinn_noise_final.png',
    'kappa': fig_dir / 'kappa_dependence.png'
}

titles = {
    'circuit': '(a) Circuit Depth Comparison\n43.2% Reduction',
    'exceptional': '(b) Exceptional Point\nγ = 1.0',
    'noise': '(c) Noise Robustness\nFidelity vs Depolarizing Rate',
    'kappa': '(d) Measurement Strength Robustness\nκ Independence'
}

positions = [(0,0), (0,1), (1,0), (1,1)]

for (key, pos) in zip(plots.keys(), positions):
    if plots[key].exists():
        img = mpimg.imread(plots[key])
        axes[pos[0], pos[1]].imshow(img)
        axes[pos[0], pos[1]].axis('off')
        axes[pos[0], pos[1]].set_title(titles[key], fontsize=12, fontweight='bold')

plt.suptitle('Phase 4: Error Mitigation & Readout Classification\n'
             'Circuit Depth: 43.2% Reduction | Readout Errors: 1.04-2.81%', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

# Save updated summary
output_path = fig_dir / 'phase4_summary_updated.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.with_suffix('.eps'), format='eps', bbox_inches='tight')
print(f"✅ Updated summary saved: {output_path}")
plt.close()
