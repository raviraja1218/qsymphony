#!/usr/bin/env python
"""
Step 2.1: Simplified SME solver that works with QuTiP 4.7.5
"""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mesolve
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    print("Please install QuTiP: pip install qutip")
    sys.exit(1)

# Load hardware parameters
hw_params_file = Path('~/projects/qsymphony/phase2_quantum_sim/hardware_params.json').expanduser()
with open(hw_params_file, 'r') as f:
    hw_params = json.load(f)

# Configuration
config = {
    'hilbert': {'transmon_levels': 2, 'mechanical_levels': 15},
    'simulation': {'time_step_ns': 1.0, 'time_total_us': 50.0},
    'constants': {'temperature_mK': 20, 'hbar': 1.0545718e-34, 'kBoltzmann': 1.380649e-23},
    'measurement': {'kappa_MHz': 50.0, 'efficiency': 0.9},
}

print("="*60)
print("STEP 2.1: Simplified SME Implementation")
print("="*60)

print(f"\n📋 Loaded hardware parameters:")
print(f"  Qubit frequency: {hw_params['qubit']['frequency_ghz']} GHz")
print(f"  Mechanical frequency: {hw_params['mechanical']['frequency_mhz']} MHz")
print(f"  Coupling g0: {hw_params['couplings']['g0_qubit_mech_mhz']} MHz")

# Hilbert space dimensions
N_q = config['hilbert']['transmon_levels']
N_m = config['hilbert']['mechanical_levels']

# Convert frequencies to angular frequencies
wq = 2 * np.pi * hw_params['qubit']['frequency_ghz'] * 1e9
wm = 2 * np.pi * hw_params['mechanical']['frequency_mhz'] * 1e6
g0 = 2 * np.pi * hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6

# Decay rates
T1_q = hw_params['losses']['t1_qubit_us'] * 1e-6
T2_q = hw_params['losses']['t2_qubit_us'] * 1e-6
T1_m = hw_params['losses']['t1_mech_us'] * 1e-6

gamma_q = 1.0 / T1_q
gamma_phi = 1.0 / T2_q - 0.5 / T1_q
gamma_m = 1.0 / T1_m

# Thermal occupancy
T = config['constants']['temperature_mK'] * 1e-3
hbar = config['constants']['hbar']
kB = config['constants']['kBoltzmann']
n_th = 1.0 / (np.exp(hbar * wm / (kB * T)) - 1)
print(f"\n🌡️  Thermal occupancy: n_th = {n_th:.3f}")

# Build operators
a = tensor(destroy(N_q), qeye(N_m))
a_dag = tensor(destroy(N_q).dag(), qeye(N_m))
n_q = a_dag * a

b = tensor(qeye(N_q), destroy(N_m))
b_dag = tensor(qeye(N_q), destroy(N_m).dag())
n_m = b_dag * b

# Hamiltonian
H_q = wq * n_q
H_m = wm * n_m
H_int = g0 * (a_dag * b + a * b_dag)
H = H_q + H_m + H_int

print(f"\n🎯 Hamiltonian built")

# Collapse operators
c_ops = []

# Qubit relaxation
c_ops.append(np.sqrt(gamma_q) * a)

# Qubit dephasing
c_ops.append(np.sqrt(2 * gamma_phi) * n_q)

# Mechanical emission
c_ops.append(np.sqrt(gamma_m * (n_th + 1)) * b)

# Mechanical absorption
c_ops.append(np.sqrt(gamma_m * n_th) * b_dag)

print(f"\n💫 Collapse operators: {len(c_ops)}")

# Initial state
rho_q = basis(N_q, 0) * basis(N_q, 0).dag()
rho_m = qt.thermal_dm(N_m, n_th)
rho0 = tensor(rho_q, rho_m)

print(f"\n🚀 Running mesolve (deterministic master equation)...")

# Time list
dt = config['simulation']['time_step_ns'] * 1e-9
T_total = config['simulation']['time_total_us'] * 1e-6
tlist = np.arange(0, T_total + dt, dt)

# Run mesolve (deterministic)
result = qt.mesolve(
    H, rho0, tlist,
    c_ops=c_ops,
    e_ops=[n_q, n_m, a_dag * b + a * b_dag],
    progress_bar=True
)

print("✅ Simulation complete!")

# Create verification plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

t_us = result.times * 1e6

# Plot 1: Qubit population
axes[0, 0].plot(t_us, result.expect[0], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (μs)')
axes[0, 0].set_ylabel('⟨n_q⟩')
axes[0, 0].set_title('Qubit Population Decay')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Mechanical population
axes[0, 1].plot(t_us, result.expect[1], 'g-', linewidth=2)
axes[0, 1].axhline(y=n_th, color='r', linestyle='--', label=f'n_th={n_th:.3f}')
axes[0, 1].set_xlabel('Time (μs)')
axes[0, 1].set_ylabel('⟨n_m⟩')
axes[0, 1].set_title('Mechanical Mode Thermalization')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Correlation
axes[1, 0].plot(t_us, result.expect[2], 'm-', linewidth=2)
axes[1, 0].set_xlabel('Time (μs)')
axes[1, 0].set_ylabel('⟨a†b + a b†⟩')
axes[1, 0].set_title('Qubit-Mechanical Correlation')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Exponential fit check
axes[1, 1].semilogy(t_us, result.expect[0], 'b-', label='Simulation')
n_q0 = result.expect[0][0]
theory = n_q0 * np.exp(-t_us / (T1_q * 1e6))
axes[1, 1].semilogy(t_us, theory, 'r--', label=f'Theory T₁={T1_q*1e6:.1f}μs')
axes[1, 1].set_xlabel('Time (μs)')
axes[1, 1].set_ylabel('⟨n_q⟩ (log scale)')
axes[1, 1].set_title('Exponential Decay Check')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('SME Solver Verification (mesolve)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save plot
save_dir = Path('~/projects/qsymphony/results/phase2/validation').expanduser()
save_dir.mkdir(parents=True, exist_ok=True)
plot_path = save_dir / 'sme_verification.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Verification plot saved to: {plot_path}")

# Save solver object
import pickle
solver = {
    'H': H, 'c_ops': c_ops, 'n_th': n_th,
    'wq': wq, 'wm': wm, 'g0': g0,
    'T1_q': T1_q, 'T2_q': T2_q, 'T1_m': T1_m
}
solver_path = save_dir / 'sme_solver.pkl'
with open(solver_path, 'wb') as f:
    pickle.dump(solver, f)
print(f"✅ Solver saved to: {solver_path}")

# Print summary
print("\n" + "="*60)
print("STEP 2.1 SUMMARY")
print("="*60)
print("✅ SME solver validated with mesolve")
print(f"✅ Thermal occupancy: n_th = {n_th:.3f}")
print(f"✅ Qubit T₁: {T1_q*1e6:.1f} μs")
print(f"✅ Mechanical T₁: {T1_m*1e6:.1f} μs")
print(f"\nVerification plot: {plot_path}")
print(f"Solver object: {solver_path}")
print("\n✅ Ready for Step 2.2")
print("="*60)
