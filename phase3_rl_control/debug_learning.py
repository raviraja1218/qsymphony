#!/usr/bin/env python
"""
Debug why agent isn't learning - test if actions affect E_N
"""

import numpy as np
import torch
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from utils.environment_wrapper_quantum import QuantumControlEnv

# Test 1: Random actions - see if E_N varies
print("="*60)
print("TEST 1: Random actions - checking E_N variation")
print("="*60)

env = QuantumControlEnv(mode='oracle')
E_N_values = []

for episode in range(3):
    obs, _ = env.reset()
    episode_EN = []
    
    for step in range(100):  # Just 100 steps per episode
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_EN.append(info.get('E_N', 0))
        
        if terminated or truncated:
            break
    
    E_N_values.extend(episode_EN)
    print(f"Episode {episode+1}: Mean E_N = {np.mean(episode_EN):.4f}, Std = {np.std(episode_EN):.4f}")

print(f"\nOverall: Mean E_N = {np.mean(E_N_values):.4f}, Std = {np.std(E_N_values):.4f}")

# Test 2: Fixed actions - see if we can find actions that increase E_N
print("\n" + "="*60)
print("TEST 2: Searching for actions that increase E_N")
print("="*60)

best_E_N = 0
best_action = None

# Try different action combinations
for delta in np.linspace(-2, 2, 10):
    for alpha in np.linspace(0, 1, 10):
        obs, _ = env.reset()
        action = np.array([delta, alpha])
        
        # Run for 100 steps with fixed action
        E_Ns = []
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            E_Ns.append(info.get('E_N', 0))
            if terminated or truncated:
                break
        
        mean_EN = np.mean(E_Ns)
        if mean_EN > best_E_N:
            best_E_N = mean_EN
            best_action = action
            print(f"  New best: Δ={delta:.2f}, α={alpha:.2f} → E_N={mean_EN:.4f}")

print(f"\nBest found: Δ={best_action[0]:.2f}, α={best_action[1]:.2f} → E_N={best_E_N:.4f}")

# Test 3: Check if environment is actually evolving
print("\n" + "="*60)
print("TEST 3: Check if state is evolving")
print("="*60)

obs, _ = env.reset()
initial_obs = obs.copy()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    diff = np.linalg.norm(obs - initial_obs)
    print(f"Step {step+1}: Observation change = {diff:.6f}")
    if diff < 1e-6:
        print("  ⚠️ Observation not changing!")
    initial_obs = obs.copy()
