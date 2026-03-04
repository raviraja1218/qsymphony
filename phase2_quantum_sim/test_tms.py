#!/usr/bin/env python
"""
Test Two-Mode Squeezing for entanglement generation
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def compute_log_negativity(rho):
    """Correct logarithmic negativity"""
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    trace_norm = np.sum(np.abs(evals))
    return np.log2(trace_norm)

print("="*60)
print("TESTING TWO-MODE SQUEEZING")
print("="*60)

# Parameters
N_q, N_m = 2, 15
wq = 2 * np.pi * 4.753e9
wm = 2 * np.pi * 492.4e6
g0 = 2 * np.pi * 11.19e6  # beam splitter
g_tms = 2 * np.pi * 5e6     # two-mode squeezing

# Operators
a = qt.tensor(qt.destroy(N_q), qt.qeye(N_m))
b = qt.tensor(qt.qeye(N_q), qt.destroy(N_m))

# Initial state |0,0⟩
psi0 = qt.tensor(qt.basis(N_q, 0), qt.basis(N_m, 0))

# Test 1: Beam splitter only (NO entanglement)
print("\n📊 TEST 1: Beam splitter only (a†b + ab†)")
H_bs = wq * a.dag() * a + wm * b.dag() * b + g0 * (a.dag() * b + a * b.dag())

tlist = np.linspace(0, 100e-9, 100)
result = qt.mesolve(H_bs, psi0, tlist, e_ops=[])
rho_final = qt.ket2dm(result.states[-1])
E_N = compute_log_negativity(rho_final)
print(f"Final E_N = {E_N:.6f} (should be 0)")

# Test 2: Two-mode squeezing only (a†b† + ab)
print("\n📊 TEST 2: Two-mode squeezing only (a†b† + ab)")
H_tms = wq * a.dag() * a + wm * b.dag() * b + g_tms * (a.dag() * b.dag() + a * b)

result = qt.mesolve(H_tms, psi0, tlist, e_ops=[])
rho_final = qt.ket2dm(result.states[-1])
E_N = compute_log_negativity(rho_final)
print(f"Final E_N = {E_N:.6f} (should be >0)")

# Test 3: Both together
print("\n📊 TEST 3: Both interactions")
H_both = wq * a.dag() * a + wm * b.dag() * b + g0 * (a.dag() * b + a * b.dag()) + g_tms * (a.dag() * b.dag() + a * b)

result = qt.mesolve(H_both, psi0, tlist, e_ops=[])
rho_final = qt.ket2dm(result.states[-1])
E_N = compute_log_negativity(rho_final)
print(f"Final E_N = {E_N:.6f}")

# Plot time evolution
print("\n📈 Computing time evolution...")
E_N_t = []
for i, t in enumerate(tlist):
    rho = qt.ket2dm(result.states[i])
    E_N_t.append(compute_log_negativity(rho))

plt.figure(figsize=(10, 6))
plt.plot(tlist*1e9, E_N_t, 'b-', linewidth=2)
plt.xlabel('Time (ns)')
plt.ylabel('E_N')
plt.title('Entanglement Generation via Two-Mode Squeezing')
plt.grid(True, alpha=0.3)
plt.savefig('tms_test.png', dpi=150)
print("✅ Plot saved: tms_test.png")
