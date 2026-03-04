#!/usr/bin/env python
"""
Optimize two-mode squeezing parameters for max entanglement
Fixed ODE solver options
"""

import numpy as np
import qutip as qt
from qutip import Options
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

# Increase ODE solver options
options = Options(nsteps=10000, atol=1e-8, rtol=1e-6)

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

# Scan g_tms - smaller range first
g_tms_values = np.linspace(0, 20e6, 8)  # 0-20 MHz
t_final = 200e-9  # 200 ns
tlist = np.linspace(0, t_final, 30)

results = []

for g_tms in tqdm(g_tms_values, desc="Scanning g_tms"):
    H = (wq * a.dag() * a + wm * b.dag() * b + 
         g0 * (a.dag() * b + a * b.dag()) + 
         g_tms * (a.dag() * b.dag() + a * b))
    
    try:
        result = qt.mesolve(H, psi0, tlist, e_ops=[], options=options)
        rho_final = qt.ket2dm(result.states[-1])
        E_N = compute_log_negativity(rho_final)
        results.append(E_N)
        print(f"\n  g_tms = {g_tms/1e6:.1f} MHz: E_N = {E_N:.6f}")
    except Exception as e:
        print(f"\n  g_tms = {g_tms/1e6:.1f} MHz: FAILED")
        results.append(0)

# Find optimal
if results:
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
    
    tlist_fine = np.linspace(0, t_final, 100)
    result = qt.mesolve(H_opt, psi0, tlist_fine, e_ops=[], options=options)
    E_N_t = []
    for i, state in enumerate(result.states):
        rho = qt.ket2dm(state)
        E_N_t.append(compute_log_negativity(rho))
    
    plt.subplot(1, 2, 2)
    plt.plot(tlist_fine*1e9, E_N_t, 'r-', linewidth=2)
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
else:
    print("❌ No valid results")

