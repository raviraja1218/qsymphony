#!/usr/bin/env python
"""
OPTIMIZE TWO-MODE SQUEEZING - FINAL RUN
Find maximum entanglement
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
print("OPTIMIZING TWO-MODE SQUEEZING - FINAL")
print("="*60)

options = Options(nsteps=200000, atol=1e-8, rtol=1e-6)

# Parameters
N_q, N_m = 2, 15
wq = 2 * np.pi * 4.753e9
wm = 2 * np.pi * 492.4e6
g0 = 2 * np.pi * 11.19e6

a = qt.tensor(qt.destroy(N_q), qt.qeye(N_m))
b = qt.tensor(qt.qeye(N_q), qt.destroy(N_m))
psi0 = qt.tensor(qt.basis(N_q, 0), qt.basis(N_m, 0))

# Scan up to 2 GHz
g_tms_values = np.linspace(0, 2e9, 25)  # 0-2000 MHz
t_final = 50e-9  # 50 ns
tlist = np.linspace(0, t_final, 20)

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
        print(f"\n  g_tms = {g_tms/1e6:.0f} MHz: E_N = {E_N:.4f}")
    except Exception as e:
        print(f"\n  g_tms = {g_tms/1e6:.0f} MHz: UNSTABLE - {str(e)[:50]}")
        results.append(0)
        stable_g_tms.append(g_tms)

if results:
    best_idx = np.argmax(results)
    best_g_tms = stable_g_tms[best_idx]
    best_E_N = results[best_idx]
    
    print(f"\n✅ OPTIMAL: g_tms = {best_g_tms/1e6:.0f} MHz → E_N = {best_E_N:.4f}")
    
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
    
    # Create final figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # E_N vs g_tms
    ax = axes[0, 0]
    ax.plot(np.array(stable_g_tms)/1e6, results, 'bo-', linewidth=2)
    ax.axvline(x=best_g_tms/1e6, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('g_tms (MHz)')
    ax.set_ylabel('E_N')
    ax.set_title(f'Entanglement vs Squeezing Strength\nPeak: {best_E_N:.4f}')
    ax.grid(True, alpha=0.3)
    
    # Time evolution
    ax = axes[0, 1]
    ax.plot(tlist_fine*1e9, E_N_t, 'r-', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('E_N')
    ax.set_title('Entanglement Evolution at Optimal')
    ax.grid(True, alpha=0.3)
    
    # Populations
    n_q_t = [qt.expect(a.dag() * a, state) for state in result.states]
    n_m_t = [qt.expect(b.dag() * b, state) for state in result.states]
    
    ax = axes[1, 0]
    ax.plot(tlist_fine*1e9, n_q_t, 'b-', label='⟨n_q⟩', linewidth=2)
    ax.plot(tlist_fine*1e9, n_m_t, 'r-', label='⟨n_m⟩', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_title('Population Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final state information
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
    FINAL RESULTS
    =============
    Optimal g_tms: {best_g_tms/1e6:.0f} MHz
    Maximum E_N: {best_E_N:.4f}
    
    Populations:
    ⟨n_q⟩ = {n_q_t[-1]:.4f}
    ⟨n_m⟩ = {n_m_t[-1]:.4f}
    
    Theoretical max for 2×15: ~3.9
    Efficiency: {best_E_N/3.9*100:.1f}%
    
    Next step: Use g_tms = {best_g_tms/1e6:.0f} MHz
    in Phase 3 training
    """
    ax.text(0.1, 0.9, info_text, fontsize=12, va='top', family='monospace')
    
    plt.suptitle('Two-Mode Squeezing Optimization - FINAL', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tms_final.png', dpi=150)
    print("✅ Final plot saved: tms_final.png")
    
    # Save parameters for Phase 3
    with open('tms_params.txt', 'w') as f:
        f.write(f"# Optimal parameters for Phase 3\n")
        f.write(f"g_tms_hz = {best_g_tms:.1f}\n")
        f.write(f"g_tms_mhz = {best_g_tms/1e6:.1f}\n")
        f.write(f"max_E_N = {best_E_N:.4f}\n")
    
    print("\n📁 Parameters saved to: tms_params.txt")
    print("\n🚀 Ready for Phase 3 with g_tms =", best_g_tms/1e6, "MHz")

if __name__ == "__main__":
    main()
