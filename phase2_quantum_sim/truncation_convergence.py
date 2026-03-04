#!/usr/bin/env python
"""
Test Hilbert space truncation convergence
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def compute_entanglement_for_N(N_m, seed=42):
    """Compute E_N for given mechanical truncation"""
    np.random.seed(seed)
    
    # System parameters from Phase 1
    wq = 2 * np.pi * 4.753e9
    wm = 2 * np.pi * 492.4e6
    g0 = 2 * np.pi * 11.19e6
    
    # Operators
    a = qt.tensor(qt.destroy(2), qt.qeye(N_m))
    b = qt.tensor(qt.qeye(2), qt.destroy(N_m))
    
    # Hamiltonian
    H = wq * a.dag() * a + wm * b.dag() * b + g0 * (a.dag() * b + a * b.dag())
    
    # Initial state |0,0⟩
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(N_m, 0))
    
    # Evolve for short time
    tlist = np.linspace(0, 50e-9, 10)
    result = qt.mesolve(H, psi0, tlist, e_ops=[])
    
    # Compute entanglement at final time
    rho = qt.ket2dm(result.states[-1])
    
    # CORRECTED entanglement calculation
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    trace_norm = np.sum(np.abs(evals))
    E_N = np.log2(trace_norm)
    
    return E_N

def convergence_test():
    """Run convergence test for different truncations"""
    N_values = [10, 15, 20, 25, 30]
    seeds = [42, 123, 456, 789, 101112]
    
    results = {N: [] for N in N_values}
    
    print("="*60)
    print("TRUNCATION CONVERGENCE TEST")
    print("="*60)
    
    for N in N_values:
        print(f"\nTesting N_m = {N}...")
        for seed in tqdm(seeds, desc=f"Seeds for N={N}"):
            E_N = compute_entanglement_for_N(N, seed)
            results[N].append(E_N)
    
    # Compute statistics
    means = []
    stds = []
    
    print("\n" + "="*60)
    print("CONVERGENCE RESULTS")
    print("="*60)
    print(f"{'N_m':<6} {'Mean E_N':<12} {'Std':<12} {'Change from N=15':<20}")
    print("-"*60)
    
    ref_mean = np.mean(results[15])
    
    for N in N_values:
        mean_val = np.mean(results[N])
        std_val = np.std(results[N])
        means.append(mean_val)
        stds.append(std_val)
        
        change = (mean_val - ref_mean) / ref_mean * 100 if ref_mean != 0 else 0
        print(f"{N:<6} {mean_val:<12.6f} {std_val:<12.6f} {change:>+18.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(N_values, means, yerr=stds, fmt='bo-', capsize=5, linewidth=2)
    plt.axhline(y=ref_mean, color='r', linestyle='--', alpha=0.5, label=f'N=15 reference')
    plt.xlabel('Mechanical Hilbert space dimension N_m')
    plt.ylabel('E_N')
    plt.title('Convergence of Entanglement with Truncation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save
    plot_file = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase2' / 'figures' / 'truncation_convergence.png'
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=150)
    print(f"\n✅ Plot saved: {plot_file}")
    
    # Save table for supplementary
    table_file = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase2' / 'data' / 'convergence_table.csv'
    with open(table_file, 'w') as f:
        f.write("N_m,Mean_E_N,Std_E_N,Percent_Change\n")
        for i, N in enumerate(N_values):
            change = (means[i] - ref_mean) / ref_mean * 100
            f.write(f"{N},{means[i]:.6f},{stds[i]:.6f},{change:.2f}\n")
    
    print(f"✅ Table saved: {table_file}")
    
    return results

if __name__ == "__main__":
    results = convergence_test()
