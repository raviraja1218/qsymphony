#!/usr/bin/env python
"""
Optimize two-mode squeezing parameters for max entanglement
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_log_negativity(rho):
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    trace_norm = np.sum(np.abs(evals))
    return np.log2(trace_norm)

print("="*60)
print("OPTIMIZING TWO-MODE SQUEEZING")
print("="*60)

# Parameters
N_q, N_m = 2, 15
wq = 2 * np.pi * 4.753e9
wm = 2 * np.pi * 492.4e6
g0 = 2 * np.pi * 11.19e6

# Operators
a = qt.tensor(qt.destroy(N_q), qt.qeye(N_m))
b = qt.tensor(qt.qeye(N_q), qt.destroy(N_m))

# Initial state
psi0 = qt.tensor(qt.basis(N_q, 0), qt.basis(N_m, 0))

# Scan g_tms
g_tms_values = np.linspace(0, 50e6, 10)  # 0-50 MHz
t_final = 500e-9  # 500 ns
tlist = np.linspace(0, t_final, 50)

results = []

for g_tms in tqdm(g_tms_values, desc="Scanning g_tms"):
    H = (wq * a.dag() * a + wm * b.dag() * b + 
         g0 * (a.dag() * b + a * b.dag()) + 
         g_tms * (a.dag() * b.dag() + a * b))
    
    result = qt.mesolve(H, psi0, tlist, e_ops=[])
    rho_final = qt.ket2dm(result.states[-1])
    E_N = compute_log_negativity(rho_final)
    results.append(E_N)
    
    if g_tms > 0:
        print(f"g_tms = {g_tms/1e6:.1f} MHz: E_N = {E_N:.6f}")

# Find optimal
best_idx = np.argmax(results)
best_g_tms = g_tms_values[best_idx]
best_E_N = results[best_idx]

print(f"\n✅ Optimal: g_tms = {best_g_tms/1e6:.1f} MHz → E_N = {best_E_N:.6f}")

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(g_tms_values/1e6, results, 'bo-')
plt.xlabel('g_tms (MHz)')
plt.ylabel('E_N')
plt.title('Entanglement vs Squeezing Strength')
plt.grid(True, alpha=0.3)

# Time evolution at optimal
H_opt = (wq * a.dag() * a + wm * b.dag() * b + 
         g0 * (a.dag() * b + a * b.dag()) + 
         best_g_tms * (a.dag() * b.dag() + a * b))

result = qt.mesolve(H_opt, psi0, tlist, e_ops=[])
E_N_t = []
for i, state in enumerate(result.states):
    rho = qt.ket2dm(state)
    E_N_t.append(compute_log_negativity(rho))

plt.subplot(1, 2, 2)
plt.plot(tlist*1e9, E_N_t, 'r-', linewidth=2)
plt.xlabel('Time (ns)')
plt.ylabel('E_N')
plt.title(f'Evolution at g_tms = {best_g_tms/1e6:.1f} MHz')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tms_optimization.png', dpi=150)
print("✅ Plot saved: tms_optimization.png")

# Save best parameters
print("\n📝 Best parameters for Phase 3:")
print(f"g_tms/2π = {best_g_tms/1e6:.1f} MHz")
print(f"Expected E_N = {best_E_N:.4f}")
