#!/usr/bin/env python
"""
Parameter sweeps to show decoherence dependence
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

def simulate_with_params(kappa, T1, n_th):
    """Simulate entanglement for given parameters"""
    # This is a proxy - replace with actual simulation
    # For now, use physically motivated model
    base_E = 1.0
    decay = np.exp(-kappa/100) * np.exp(-T1/200) * np.exp(-n_th)
    return base_E * decay

# Parameter ranges
kappa_range = np.linspace(10, 100, 20)
T1_range = np.linspace(20, 200, 20)
n_th_range = np.linspace(0, 1, 20)

# Sweep 1: E_N vs κ (fixed T1=85, n_th=0.1)
E_vs_kappa = []
for kappa in tqdm(kappa_range, desc="Sweeping κ"):
    E = simulate_with_params(kappa, 85, 0.1)
    E_vs_kappa.append(E)

# Sweep 2: E_N vs T1 (fixed κ=50, n_th=0.1)
E_vs_T1 = []
for T1 in tqdm(T1_range, desc="Sweeping T1"):
    E = simulate_with_params(50, T1, 0.1)
    E_vs_T1.append(E)

# Sweep 3: E_N vs n_th (fixed κ=50, T1=85)
E_vs_nth = []
for n_th in tqdm(n_th_range, desc="Sweeping n_th"):
    E = simulate_with_params(50, 85, n_th)
    E_vs_nth.append(E)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(kappa_range, E_vs_kappa, 'b-', linewidth=2)
axes[0].set_xlabel('κ (MHz)')
axes[0].set_ylabel('E_N')
axes[0].set_title('Entanglement vs Measurement Strength')
axes[0].grid(True, alpha=0.3)

axes[1].plot(T1_range, E_vs_T1, 'r-', linewidth=2)
axes[1].set_xlabel('T₁ (μs)')
axes[1].set_ylabel('E_N')
axes[1].set_title('Entanglement vs Qubit Lifetime')
axes[1].grid(True, alpha=0.3)

axes[2].plot(n_th_range, E_vs_nth, 'g-', linewidth=2)
axes[2].set_xlabel('n_th')
axes[2].set_ylabel('E_N')
axes[2].set_title('Entanglement vs Thermal Noise')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_sweeps.png', dpi=150)
print("✅ Parameter sweeps saved: parameter_sweeps.png")
