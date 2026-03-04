#!/usr/bin/env python
"""
Optimize two-mode squeezing - MAXIMUM
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
print("OPTIMIZING TWO-MODE SQUEEZING - MAXIMUM")
print("="*60)

options = Options(nsteps=50000, atol=1e-8, rtol=1e-6)

# Parameters
N_q, N_m = 2, 15
wq = 2 * np.pi * 4.753e9
wm = 2 * np.pi * 492.4e6
g0 = 2 * np.pi * 11.19e6

a = qt.tensor(qt.destroy(N_q), qt.qeye(N_m))
b = qt.tensor(qt.qeye(N_q), qt.destroy(N_m))
psi0 = qt.tensor(qt.basis(N_q, 0), qt.basis(N_m, 0))

# Try very strong coupling (up to 200 MHz)
g_tms_values = np.linspace(0, 200e6, 15)
t_final = 200e-9  # 200 ns (shorter time for stability)
tlist = np.linspace(0, t_final, 30)

results = []
stable_g_tms = []

for g_tms in tqdm(g_tms_values, desc="Scanning g_tms"):
    H = (wq * a.dag() * a + wm * b.dag() * b + 
         g0 * (a.dag() * b + a * b.dag()) + 
         g_tms * (a.dag() * b.dag() + a * b))
    
    try:
        result = qt.mesolve(H, psi0, tlist, e_ops=[], options=options)
        rho_final = qt.ket2dm(result.states[-1])
        E_N = compute_log_negativity(rho_final)
        results.append(E_N)
        stable_g_tms.append(g_tms)
        print(f"\n  g_tms = {g_tms/1e6:.1f} MHz: E_N = {E_N:.6f}")
    except Exception as e:
        print(f"\n  g_tms = {g_tms/1e6:.1f} MHz: UNSTABLE")
        results.append(0)
        stable_g_tms.append(g_tms)

if results:
    best_idx = np.argmax(results)
    best_g_tms = stable_g_tms[best_idx]
    best_E_N = results[best_idx]
    
    print(f"\n✅ Optimal: g_tms = {best_g_tms/1e6:.1f} MHz → E_N = {best_E_N:.4f}")
    
    # Time evolution at optimal
    H_opt = (wq * a.dag() * a + wm * b.dag() * b + 
             g0 * (a.dag() * b + a * b.dag()) + 
             best_g_tms * (a.dag() * b.dag() + a * b))
    
    tlist_fine = np.linspace(0, t_final, 200)
    result = qt.mesolve(H_opt, psi0, tlist_fine, e_ops=[], options=options)
    E_N_t = []
    for state in result.states:
        rho = qt.ket2dm(state)
        E_N_t.append(compute_log_negativity(rho))
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(np.array(stable_g_tms)/1e6, results, 'bo-')
    plt.xlabel('g_tms (MHz)')
    plt.ylabel('E_N')
    plt.title('Entanglement vs Squeezing Strength')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(tlist_fine*1e9, E_N_t, 'r-', linewidth=2)
    plt.xlabel('Time (ns)')
    plt.ylabel('E_N')
    plt.title(f'Evolution at optimal')
    plt.grid(True, alpha=0.3)
    
    # Photon numbers
    n_q_t = []
    n_m_t = []
    for state in result.states:
        n_q_t.append(qt.expect(a.dag() * a, state))
        n_m_t.append(qt.expect(b.dag() * b, state))
    
    plt.subplot(1, 3, 3)
    plt.plot(tlist_fine*1e9, n_q_t, 'b-', label='⟨n_q⟩')
    plt.plot(tlist_fine*1e9, n_m_t, 'r-', label='⟨n_m⟩')
    plt.xlabel('Time (ns)')
    plt.ylabel('Photon number')
    plt.title('Population evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tms_max.png', dpi=150)
    print("✅ Plot saved: tms_max.png")
    
    print("\n📝 Best parameters for Phase 3:")
    print(f"g_tms/2π = {best_g_tms/1e6:.1f} MHz")
    print(f"Expected E_N = {best_E_N:.4f}")
    print(f"Max ⟨n_q⟩ = {max(n_q_t):.4f}")
    print(f"Max ⟨n_m⟩ = {max(n_m_t):.4f}")
