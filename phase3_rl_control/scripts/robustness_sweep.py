#!/usr/bin/env python
"""
Robustness sweeps for trained policies
Tests performance against parameter variations
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_physics import PhysicsControlEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {device}")

def load_best_policy(seed=1001):
    """Load the best trained policy (seed 1001 had highest E_N)"""
    model_path = Path(f"models/ppo_physics_seed_{seed}.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create environment to get dimensions
    env = PhysicsControlEnv(mode='oracle')
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = env.action_space.shape[0]
    
    policy = PPOActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    return policy

def evaluate_policy(policy, env, n_episodes=5):
    """Evaluate policy over multiple episodes"""
    E_Ns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden_state = None
        episode_EN = []
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, hidden_state = policy.select_action(
                    obs_tensor, hidden_state, deterministic=True
                )
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            episode_EN.append(info.get('E_N', 0))
            
            if terminated or truncated:
                break
        
        E_Ns.append(np.mean(episode_EN))
    
    return np.mean(E_Ns), np.std(E_Ns)

def sweep_kappa(policy):
    """Test different measurement strengths"""
    print("\n" + "="*60)
    print("SWEEP 1: Measurement Strength (κ)")
    print("="*60)
    
    kappa_values = [30, 40, 50, 60, 70, 80, 90, 100]  # MHz
    E_N_means = []
    E_N_stds = []
    
    for kappa in kappa_values:
        env = PhysicsControlEnv(mode='oracle')
        env.quantum_env.kappa = 2 * np.pi * kappa * 1e6
        
        mean_EN, std_EN = evaluate_policy(policy, env)
        E_N_means.append(mean_EN)
        E_N_stds.append(std_EN)
        print(f"  κ = {kappa:3d} MHz: E_N = {mean_EN:.4f} ± {std_EN:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(kappa_values, E_N_means, yerr=E_N_stds, fmt='bo-', capsize=5, linewidth=2)
    plt.xlabel('κ (MHz)')
    plt.ylabel('E_N')
    plt.title('Robustness to Measurement Strength')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_kappa.png', dpi=150)
    print("\n✅ Plot saved: results/phase3/figures/robustness_kappa.png")
    
    return kappa_values, E_N_means, E_N_stds

def sweep_T1(policy):
    """Test different qubit lifetimes"""
    print("\n" + "="*60)
    print("SWEEP 2: Qubit Lifetime (T₁)")
    print("="*60)
    
    T1_nominal = 85e-6
    factors = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    E_N_means = []
    E_N_stds = []
    
    for factor in factors:
        env = PhysicsControlEnv(mode='oracle')
        env.quantum_env.T1_q = T1_nominal * factor
        
        mean_EN, std_EN = evaluate_policy(policy, env)
        E_N_means.append(mean_EN)
        E_N_stds.append(std_EN)
        print(f"  T₁ = {T1_nominal*factor*1e6:.0f} μs (x{factor:.1f}): E_N = {mean_EN:.4f} ± {std_EN:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(factors, E_N_means, yerr=E_N_stds, fmt='gs-', capsize=5, linewidth=2)
    plt.xlabel('T₁ multiplier')
    plt.ylabel('E_N')
    plt.title('Robustness to Qubit Lifetime')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_T1.png', dpi=150)
    print("\n✅ Plot saved: results/phase3/figures/robustness_T1.png")
    
    return factors, E_N_means, E_N_stds

def sweep_nth(policy):
    """Test different thermal occupancies"""
    print("\n" + "="*60)
    print("SWEEP 3: Thermal Occupancy (n_th)")
    print("="*60)
    
    nth_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    E_N_means = []
    E_N_stds = []
    
    for nth in nth_values:
        env = PhysicsControlEnv(mode='oracle')
        env.quantum_env.n_th = nth
        
        mean_EN, std_EN = evaluate_policy(policy, env)
        E_N_means.append(mean_EN)
        E_N_stds.append(std_EN)
        print(f"  n_th = {nth:.1f}: E_N = {mean_EN:.4f} ± {std_EN:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(nth_values, E_N_means, yerr=E_N_stds, fmt='rs-', capsize=5, linewidth=2)
    plt.xlabel('Thermal occupancy n_th')
    plt.ylabel('E_N')
    plt.title('Robustness to Thermal Noise')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_nth.png', dpi=150)
    print("\n✅ Plot saved: results/phase3/figures/robustness_nth.png")
    
    return nth_values, E_N_means, E_N_stds

def main():
    print("="*60)
    print("ROBUSTNESS SWEEPS - PHASE 3")
    print("="*60)
    
    policy = load_best_policy(seed=1001)  # Best performing seed
    if policy is None:
        return
    
    # Run all sweeps
    sweep_kappa(policy)
    sweep_T1(policy)
    sweep_nth(policy)
    
    print("\n" + "="*60)
    print("✅ All robustness sweeps complete!")
    print("📁 Figures saved in results/phase3/figures/")
    print("="*60)

if __name__ == "__main__":
    main()
