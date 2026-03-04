#!/usr/bin/env python
"""
Debug why agent isn't learning - with correct E_N calculation
"""

import numpy as np
import torch
from pathlib import Path
import sys
import qutip as qt

sys.path.append(str(Path(__file__).parent))

from utils.environment_wrapper_quantum import QuantumControlEnv

# CORRECT entanglement calculation
def compute_log_negativity_correct(psi):
    if psi.isket:
        rho = qt.ket2dm(psi)
    else:
        rho = psi
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    trace_norm = np.sum(np.abs(evals))
    return np.log2(trace_norm)

print("="*60)
print("DEBUG: Testing with CORRECT E_N calculation")
print("="*60)

env = QuantumControlEnv(mode='oracle')

# Test 1: Get E_N directly from quantum state
print("\n📊 TEST 1: E_N from quantum state vs info dict")
print("-"*50)

obs, _ = env.reset()
if hasattr(env.unwrapped, 'quantum_env'):
    qenv = env.unwrapped.quantum_env
    if hasattr(qenv, 'psi'):
        correct_E_N = compute_log_negativity_correct(qenv.psi)
        info_E_N = 0  # We don't have info dict here
        
        print(f"Initial state:")
        print(f"  Correct E_N: {correct_E_N:.6f}")
        print(f"  Info dict E_N: {info_E_N}")

# Test 2: Take a few steps and track both
print("\n📊 TEST 2: Evolution of correct E_N")
print("-"*50)

obs, _ = env.reset()
E_N_correct = []

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if hasattr(env.unwrapped, 'quantum_env'):
        qenv = env.unwrapped.quantum_env
        if hasattr(qenv, 'psi'):
            correct_E_N = compute_log_negativity_correct(qenv.psi)
            E_N_correct.append(correct_E_N)
            print(f"Step {step+1}: Correct E_N = {correct_E_N:.6f}")
    
    if terminated or truncated:
        break

print(f"\nCorrect E_N statistics:")
print(f"  Mean: {np.mean(E_N_correct):.6f}")
print(f"  Std: {np.std(E_N_correct):.6f}")
print(f"  Min: {np.min(E_N_correct):.6f}")
print(f"  Max: {np.max(E_N_correct):.6f}")

# Test 3: Try to find actions that increase E_N
print("\n📊 TEST 3: Searching for actions that increase correct E_N")
print("-"*50)

best_E_N = 0
best_action = None

for delta in np.linspace(-2, 2, 5):
    for alpha in np.linspace(0, 1, 5):
        obs, _ = env.reset()
        action = np.array([delta, alpha])
        
        E_Ns = []
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if hasattr(env.unwrapped, 'quantum_env'):
                qenv = env.unwrapped.quantum_env
                if hasattr(qenv, 'psi'):
                    correct_E_N = compute_log_negativity_correct(qenv.psi)
                    E_Ns.append(correct_E_N)
            
            if terminated or truncated:
                break
        
        if E_Ns:
            mean_EN = np.mean(E_Ns)
            if mean_EN > best_E_N:
                best_E_N = mean_EN
                best_action = action
                print(f"  Δ={delta:.2f}, α={alpha:.2f} → E_N={mean_EN:.4f}")

print(f"\n✅ Best found: Δ={best_action[0]:.2f}, α={best_action[1]:.2f} → E_N={best_E_N:.4f}")
