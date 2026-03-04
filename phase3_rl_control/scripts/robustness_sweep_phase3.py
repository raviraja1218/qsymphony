#!/usr/bin/env python
"""
Test policy robustness against parameter variations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_quantum import QuantumControlEnv
from utils.entanglement_fixed import compute_log_negativity_correct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_best_policy():
    """Load the best trained policy"""
    model_path = Path('models/ppo_oracle_seed_1042.pt')  # Use best seed
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get observation dimension
    env = QuantumControlEnv(mode='oracle')
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
        episode_E_N = []
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, hidden_state = policy.select_action(
                    obs_tensor, hidden_state, deterministic=True
                )
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            
            if 'psi' in info:
                E_N = compute_log_negativity_correct(info['psi'])
                episode_E_N.append(E_N)
            
            if terminated or truncated:
                break
        
        E_Ns.append(np.mean(episode_E_N))
    
    return np.mean(E_Ns), np.std(E_Ns)

def sweep_kappa(policy):
    """Test different measurement strengths"""
    print("\n🔬 Sweeping κ (measurement strength)...")
    
    kappa_values = [30, 40, 50, 60, 70]  # MHz
    E_N_means = []
    E_N_stds = []
    
    for kappa in kappa_values:
        # Create environment with modified kappa
        env = QuantumControlEnv(mode='oracle')
        env.kappa = 2 * np.pi * kappa * 1e6
        
        mean_EN, std_EN = evaluate_policy(policy, env)
        E_N_means.append(mean_EN)
        E_N_stds.append(std_EN)
        
        print(f"  κ={kappa}MHz: E_N = {mean_EN:.4f} ± {std_EN:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(kappa_values, E_N_means, yerr=E_N_stds, fmt='bo-', capsize=5)
    plt.xlabel('κ (MHz)')
    plt.ylabel('E_N')
    plt.title('Robustness to Measurement Strength')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_kappa.png', dpi=150)
    print("✅ Plot saved: robustness_kappa.png")

def sweep_T1(policy):
    """Test different qubit lifetimes"""
    print("\n🔬 Sweeping T₁ (qubit lifetime)...")
    
    T1_factors = [0.6, 0.8, 1.0, 1.2, 1.4]  # multipliers
    T1_nominal = 85e-6
    E_N_means = []
    E_N_stds = []
    
    for factor in T1_factors:
        # Create environment with modified T1
        env = QuantumControlEnv(mode='oracle')
        env.T1_q = T1_nominal * factor
        
        mean_EN, std_EN = evaluate_policy(policy, env)
        E_N_means.append(mean_EN)
        E_N_stds.append(std_EN)
        
        print(f"  T₁={T1_nominal*factor*1e6:.0f}μs (x{factor}): E_N = {mean_EN:.4f} ± {std_EN:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(T1_factors, E_N_means, yerr=E_N_stds, fmt='gs-', capsize=5)
    plt.xlabel('T₁ multiplier')
    plt.ylabel('E_N')
    plt.title('Robustness to Qubit Lifetime')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/robustness_T1.png', dpi=150)
    print("✅ Plot saved: robustness_T1.png")

def main():
    print("="*60)
    print("ROBUSTNESS SWEEP - PHASE 3")
    print("="*60)
    
    policy = load_best_policy()
    if policy is None:
        return
    
    sweep_kappa(policy)
    sweep_T1(policy)
    
    print("\n" + "="*60)
    print("✅ Robustness sweep complete")
    print("="*60)

if __name__ == "__main__":
    main()
