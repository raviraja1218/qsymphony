#!/usr/bin/env python
"""
Robustness sweeps - OPTIMIZED for speed
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_physics import PhysicsControlEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {device}")

def load_best_policy(seed=1001):
    """Load the best trained policy"""
    model_path = Path(f"models/ppo_physics_seed_{seed}.pt")
    checkpoint = torch.load(model_path, map_location=device)
    
    env = PhysicsControlEnv(mode='oracle')
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = env.action_space.shape[0]
    
    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    return policy

def quick_evaluate(policy, env, max_steps=5000):
    """Fast evaluation - only 5000 steps instead of 50000"""
    obs, _ = env.reset()
    hidden_state = None
    E_Ns = []
    
    for step in range(max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, hidden_state = policy.select_action(
                obs_tensor, hidden_state, deterministic=True
            )
        
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
        E_Ns.append(info.get('E_N', 0))
        
        if terminated or truncated:
            break
    
    return np.mean(E_Ns)

def sweep_kappa(policy):
    """Test different measurement strengths"""
    print("\n" + "="*60)
    print("SWEEP 1: Measurement Strength (κ)")
    print("="*60)
    
    kappa_values = [30, 50, 70, 90, 110]  # Fewer points
    results = []
    
    for kappa in kappa_values:
        print(f"  Testing κ = {kappa} MHz...", end='', flush=True)
        start = time.time()
        
        env = PhysicsControlEnv(mode='oracle')
        env.quantum_env.kappa = 2 * np.pi * kappa * 1e6
        mean_EN = quick_evaluate(policy, env)
        
        elapsed = time.time() - start
        results.append(mean_EN)
        print(f" E_N = {mean_EN:.4f} (took {elapsed:.1f}s)")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(kappa_values, results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('κ (MHz)')
    plt.ylabel('E_N')
    plt.title('Robustness to Measurement Strength')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_kappa.png', dpi=150)
    print(f"\n✅ Plot saved: results/phase3/figures/robustness_kappa.png")
    
    return kappa_values, results

def sweep_T1(policy):
    """Test different qubit lifetimes"""
    print("\n" + "="*60)
    print("SWEEP 2: Qubit Lifetime (T₁)")
    print("="*60)
    
    T1_nominal = 85e-6
    factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Fewer points
    results = []
    
    for factor in factors:
        T1 = T1_nominal * factor
        print(f"  Testing T₁ = {T1*1e6:.0f} μs (x{factor:.2f})...", end='', flush=True)
        start = time.time()
        
        env = PhysicsControlEnv(mode='oracle')
        env.quantum_env.T1_q = T1
        mean_EN = quick_evaluate(policy, env)
        
        elapsed = time.time() - start
        results.append(mean_EN)
        print(f" E_N = {mean_EN:.4f} (took {elapsed:.1f}s)")
    
    plt.figure(figsize=(10, 6))
    plt.plot(factors, results, 'gs-', linewidth=2, markersize=8)
    plt.xlabel('T₁ multiplier')
    plt.ylabel('E_N')
    plt.title('Robustness to Qubit Lifetime')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_T1.png', dpi=150)
    print(f"\n✅ Plot saved: results/phase3/figures/robustness_T1.png")
    
    return factors, results

def sweep_nth(policy):
    """Test different thermal occupancies"""
    print("\n" + "="*60)
    print("SWEEP 3: Thermal Occupancy (n_th)")
    print("="*60)
    
    nth_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # Fewer points
    results = []
    
    for nth in nth_values:
        print(f"  Testing n_th = {nth:.1f}...", end='', flush=True)
        start = time.time()
        
        env = PhysicsControlEnv(mode='oracle')
        env.quantum_env.n_th = nth
        mean_EN = quick_evaluate(policy, env)
        
        elapsed = time.time() - start
        results.append(mean_EN)
        print(f" E_N = {mean_EN:.4f} (took {elapsed:.1f}s)")
    
    plt.figure(figsize=(10, 6))
    plt.plot(nth_values, results, 'rs-', linewidth=2, markersize=8)
    plt.xlabel('Thermal occupancy n_th')
    plt.ylabel('E_N')
    plt.title('Robustness to Thermal Noise')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_nth.png', dpi=150)
    print(f"\n✅ Plot saved: results/phase3/figures/robustness_nth.png")
    
    return nth_values, results

def main():
    print("="*60)
    print("ROBUSTNESS SWEEPS - OPTIMIZED")
    print("="*60)
    
    start_time = time.time()
    
    policy = load_best_policy(seed=1001)
    
    # Run sweeps
    sweep_kappa(policy)
    sweep_T1(policy)
    sweep_nth(policy)
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ ALL SWEEPS COMPLETE in {elapsed/60:.1f} minutes!")
    print("📁 Figures saved in results/phase3/figures/")
    print("="*60)

if __name__ == "__main__":
    main()
