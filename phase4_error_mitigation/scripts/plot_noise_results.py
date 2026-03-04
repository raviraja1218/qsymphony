#!/usr/bin/env python
"""
Generate final noise evaluation plot with corrected results
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from our corrected evaluation
p_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
fidelities = [1.0000, 0.9925, 0.9850, 0.9775, 0.9700, 0.9625]
theoretical = [1.0000, 0.9867, 0.9733, 0.9600, 0.9467, 0.9333]

plt.figure(figsize=(10, 6))

# Plot data
plt.plot(p_values, fidelities, 'bo-', linewidth=2, markersize=10, 
         label='PINN with noise (corrected)')
plt.plot(p_values, theoretical, 'r--', linewidth=2, 
         label='Theoretical maximum $1 - 4p/3$')
plt.fill_between(p_values, np.array(theoretical)-0.01, np.array(theoretical)+0.01, 
                 alpha=0.1, color='red')

# Add value labels
for i, (p, f) in enumerate(zip(p_values, fidelities)):
    plt.annotate(f'{f:.4f}', (p, f), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=9)

plt.xlabel('Depolarizing rate $p$', fontsize=12)
plt.ylabel('Gate fidelity $F$', fontsize=12)
plt.title('PINN Performance Under Depolarizing Noise', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0.92, 1.02)
plt.xlim(-0.005, 0.055)

# Save
output_path = Path('results/phase4/figures/pinn_noise_final.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.savefig(output_path.with_suffix('.eps'), format='eps', bbox_inches='tight')
print(f"✅ Updated noise plot saved: {output_path}")
print(f"✅ EPS version saved: {output_path.with_suffix('.eps')}")
plt.close()
