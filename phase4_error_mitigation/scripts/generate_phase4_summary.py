#!/usr/bin/env python
"""
Generate comprehensive Phase 4 summary figure
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

fig_dir = Path('results/phase4/figures')

# Create 2x2 summary figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Load existing figures
try:
    # Top left: Circuit depth comparison
    if (fig_dir / 'fig3b_circuit_depth.png').exists():
        img = mpimg.imread(fig_dir / 'fig3b_circuit_depth.png')
        axes[0, 0].imshow(img)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('(a) Circuit Depth Comparison', fontweight='bold')
    
    # Top right: Exceptional point
    if (fig_dir / 'fig3a_exceptional_point.png').exists():
        img = mpimg.imread(fig_dir / 'fig3a_exceptional_point.png')
        axes[0, 1].imshow(img)
        axes[0, 1].axis('off')
        axes[0, 1].set_title('(b) Exceptional Point', fontweight='bold')
    
    # Bottom left: Fidelity vs noise
    if (fig_dir / 'fidelity_vs_noise.png').exists():
        img = mpimg.imread(fig_dir / 'fidelity_vs_noise.png')
        axes[1, 0].imshow(img)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('(c) Noise Robustness', fontweight='bold')
    
    # Bottom right: Q2 IQ scatter
    if (fig_dir / 'q2_iq_scatter.png').exists():
        img = mpimg.imread(fig_dir / 'q2_iq_scatter.png')
        axes[1, 1].imshow(img)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('(d) IQ Readout (RQC Q2)', fontweight='bold')
    
    plt.suptitle('Phase 4: Error Mitigation & Readout Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'phase4_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig_dir / 'phase4_summary.png'}")
    
except Exception as e:
    print(f"⚠️ Error creating summary: {e}")

print("\n✅ Phase 4 figure generation complete!")
