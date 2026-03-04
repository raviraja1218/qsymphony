#!/usr/bin/env python
"""
Test Hilbert space truncation convergence
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_entanglement_for_N(N, seed=42):
    """Compute E_N for given mechanical truncation"""
    np.random.seed(seed)
    
    # System parameters (from Phase 1)
    wq = 2 * np.pi * 4.753e9
    wm = 2 * np.pi * 492.4e6
    g0 = 2 * np.pi * 11.19e6
    
    # Hilbert space
    N_q = 2
    
    # Operators
    a = qt.tensor(qt.destroy(N_q), qt.qeye(N))
    a_dag = a.dag()
    b = qt.tensor(qt.qeye(N_q), qt.destroy(N))
    b_dag = b.dag()
    
    # Hamiltonian
    H = wq * a_dag * a + wm * b_dag * b + g0 * (a_dag * b + a * b_dag)
    
    # Initial state: |0,0⟩
    psi0 = qt.tensor(qt.basis(N_q, 0), qt.basis(N, 0))
    
    # Time evolution (short time for test)
    tlist = np.linspace(0, 1e-6, 10)
    result = qt.mesolve(H, psi0, tlist, e_ops=[])
    
    # Compute entanglement at final time
    rho = qt.ket2dm(result.states[-1])
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    negativity = (np.sum(np.abs(evals[evals < 0])) + 1) / 2
    E_N = np.log2(2 * negativity + 1)
    
    return E_N

def convergence_test():
    """Test E_N for different truncations"""
    N_values = [10, 15, 20, 25, 30, 35, 40]
    seeds = [42, 123, 456, 789, 101112]
    
    results = {N: [] for N in N_values}
    
    for N in N_values:
        print(f"\nTesting N={N}...")
        for seed in tqdm(seeds):
            E_N = compute_entanglement_for_N(N, seed)
            results[N].append(E_N)
    
    # Compute statistics
    means = [np.mean(results[N]) for N in N_values]
    stds = [np.std(results[N]) for N in N_values]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(N_values, means, yerr=stds, fmt='bo-', capsize=5)
    plt.axhline(y=means[2], color='r', linestyle='--', label=f'N=15 reference')
    plt.xlabel('Mechanical truncation N')
    plt.ylabel('E_N')
    plt.title('Convergence of Entanglement with Hilbert Space Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_test.png', dpi=150)
    
    # Print results
    print("\n" + "="*60)
    print("CONVERGENCE TEST RESULTS")
    print("="*60)
    print(f"{'N':<5} {'Mean E_N':<12} {'Std':<10} {'Change from N=15':<20}")
    print("-"*60)
    
    ref = means[2]  # N=15
    for N, mean, std in zip(N_values, means, stds):
        change = (mean - ref) / ref * 100
        print(f"{N:<5} {mean:<12.6f} {std:<10.6f} {change:>+15.2f}%")
    
    return results

if __name__ == "__main__":
    convergence_test()
