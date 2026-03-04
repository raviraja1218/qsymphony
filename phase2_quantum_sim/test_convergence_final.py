#!/usr/bin/env python
"""
Final convergence test for Hilbert space truncation
"""
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm

print("="*60)
print("Convergence Test for Hilbert Space Truncation")
print("="*60)

N_values = [10, 15, 20, 25, 30]
seeds = [42, 123, 456, 789, 101112]

results = {N: [] for N in N_values}

for N in N_values:
    print(f"\nTesting N={N}...")
    for seed in tqdm(seeds):
        np.random.seed(seed)
        
        # System parameters
        wq = 2 * np.pi * 4.753e9
        wm = 2 * np.pi * 492.4e6
        g0 = 2 * np.pi * 11.19e6
        
        # Operators
        a = qt.tensor(qt.destroy(2), qt.qeye(N))
        b = qt.tensor(qt.qeye(2), qt.destroy(N))
        
        # Hamiltonian
        H = wq * a.dag() * a + wm * b.dag() * b + g0 * (a.dag() * b + a * b.dag())
        
        # Initial state
        psi0 = qt.tensor(qt.basis(2, 0), qt.basis(N, 0))
        
        # Evolve
        tlist = np.linspace(0, 50e-9, 10)
        result = qt.mesolve(H, psi0, tlist, e_ops=[])
        
        # Compute E_N
        rho = qt.ket2dm(result.states[-1])
        rho_pt = qt.partial_transpose(rho, [1, 0])
        evals = rho_pt.eigenenergies()
        negativity = (np.sum(np.abs(evals[evals < 0])) + 1) / 2
        E_N = np.log2(2 * negativity + 1)
        
        results[N].append(E_N)

# Plot results
plt.figure(figsize=(10, 6))
means = [np.mean(results[N]) for N in N_values]
stds = [np.std(results[N]) for N in N_values]

plt.errorbar(N_values, means, yerr=stds, fmt='bo-', capsize=5, linewidth=2)
plt.xlabel('Mechanical Hilbert Space Dimension N')
plt.ylabel('E_N')
plt.title('Convergence of Entanglement with Truncation')
plt.grid(True, alpha=0.3)
plt.savefig('convergence_test_final.png', dpi=150)
print("\n✅ Convergence plot saved: convergence_test_final.png")

# Print table
print("\n" + "="*60)
print("Convergence Results:")
print("-"*60)
print(f"{'N':<5} {'Mean E_N':<12} {'Std':<10} {'Change from N=15':<20}")
print("-"*60)
ref = means[N_values.index(15)]
for N, mean, std in zip(N_values, means, stds):
    change = (mean - ref)/ref * 100 if ref != 0 else 0
    print(f"{N:<5} {mean:<12.6f} {std:<10.6f} {change:>+15.2f}%")
print("="*60)
