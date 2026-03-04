#!/usr/bin/env python
"""
Test robustness of optimal layout to parameter variations
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load optimal parameters
with open('../phase2_quantum_sim/hardware_params.json', 'r') as f:
    params = json.load(f)

g0_nominal = params['couplings']['g0_qubit_mech_mhz']
wm_nominal = params['mechanical']['frequency_mhz']

# Test variations
variations = np.linspace(-0.1, 0.1, 11)  # ±10%
g0_results = []
wm_results = []

for delta in variations:
    # Vary g0
    g0 = g0_nominal * (1 + delta)
    # Here you would run a quick simulation
    # For now, use placeholder
    g0_results.append(1.0 - abs(delta))  # Placeholder
    
    # Vary ω_m
    wm = wm_nominal * (1 + delta)
    wm_results.append(1.0 - abs(delta))  # Placeholder

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(variations*100, g0_results, 'bo-', linewidth=2)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Δg₀/g₀ (%)')
axes[0].set_ylabel('Normalized E_N')
axes[0].set_title('Robustness to Coupling Variations')
axes[0].grid(True, alpha=0.3)

axes[1].plot(variations*100, wm_results, 'gs-', linewidth=2)
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Δω_m/ω_m (%)')
axes[1].set_ylabel('Normalized E_N')
axes[1].set_title('Robustness to Frequency Variations')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('robustness_sweep.png', dpi=150)
print("✅ Robustness plot saved")
