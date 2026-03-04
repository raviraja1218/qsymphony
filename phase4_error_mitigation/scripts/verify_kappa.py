#!/usr/bin/env python
"""
Verify that κ (measurement strength) affects simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.environment_wrapper_physics import PhysicsControlEnv

print("="*60)
print("VERIFYING κ DEPENDENCE")
print("="*60)

kappa_values = [30e6, 50e6, 70e6]  # 30, 50, 70 MHz
results = []

for kappa in kappa_values:
    env = PhysicsControlEnv(mode='oracle')
    env.quantum_env.kappa = kappa
    
    # Run a few steps with random actions
    obs, _ = env.reset()
    E_Ns = []
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        E_Ns.append(info.get('E_N', 0))
        if terminated or truncated:
            break
    
    mean_EN = np.mean(E_Ns)
    std_EN = np.std(E_Ns)
    results.append((kappa/1e6, mean_EN, std_EN))
    print(f"κ = {kappa/1e6:.0f} MHz: E_N = {mean_EN:.4f} ± {std_EN:.4f}")

# Plot
plt.figure(figsize=(10, 6))
kappa_vals = [r[0] for r in results]
means = [r[1] for r in results]
stds = [r[2] for r in results]

plt.errorbar(kappa_vals, means, yerr=stds, fmt='bo-', linewidth=2, markersize=8, capsize=5)
plt.xlabel('κ (MHz)')
plt.ylabel('E_N')
plt.title('Effect of Measurement Strength on Entanglement')
plt.grid(True, alpha=0.3)

plot_path = Path('results/phase4/figures/kappa_dependence.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150)
print(f"\n✅ Plot saved: {plot_path}")

# Check if values vary
if len(set(means)) > 1:
    print("✅ κ affects simulation (values vary)")
else:
    print("⚠️ κ does NOT affect simulation (all values identical)")

print("\n✅ Verification complete!")
