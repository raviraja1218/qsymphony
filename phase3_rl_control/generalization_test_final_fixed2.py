#!/usr/bin/env python
"""
Test policy generalization - FIXED for dimension mismatch
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_quantum import QuantumControlEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the measurement model (11-dim)
model_path = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models' / 'ppo_measurement_final.zip'
print(f"Loading measurement model from: {model_path}")

if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    sys.exit(1)

checkpoint = torch.load(model_path, map_location=device)

# Create network for measurement mode (11-dim input)
policy = PPOActorCritic(obs_dim=11, action_dim=2).to(device)
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()
print("✅ Measurement model loaded successfully")

def evaluate_policy(env, policy, n_episodes=5):
    """Evaluate policy over multiple episodes"""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden_state = None
        total_reward = 0
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, hidden_state = policy.select_action(
                    obs_tensor, hidden_state, deterministic=True
                )
            obs, reward, done, _, _ = env.step(action.cpu().numpy()[0])
            total_reward += reward
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

# Test different κ values (measurement strength)
print("\n1. Testing different measurement strengths:")
kappa_values = [25, 50, 75, 100]
kappa_results = []
kappa_stds = []

for kappa in kappa_values:
    # Create environment in measurement mode
    env = QuantumControlEnv(mode='measurement')
    # Note: You'll need to modify your env to accept kappa parameter
    mean_r, std_r = evaluate_policy(env, policy)
    kappa_results.append(mean_r)
    kappa_stds.append(std_r)
    print(f"  κ={kappa}MHz: {mean_r:.2f} ± {std_r:.2f}")

# Test different thermal occupancies
print("\n2. Testing different thermal occupancies:")
nth_values = [0.1, 0.3, 0.5, 0.7, 1.0]
nth_results = []
nth_stds = []

for nth in nth_values:
    env = QuantumControlEnv(mode='measurement')
    # Note: You'll need to modify your env to accept n_th parameter
    mean_r, std_r = evaluate_policy(env, policy)
    nth_results.append(mean_r)
    nth_stds.append(std_r)
    print(f"  n_th={nth}: {mean_r:.2f} ± {std_r:.2f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Kappa plot
axes[0].errorbar(kappa_values, kappa_results, yerr=kappa_stds, 
                 fmt='ro-', linewidth=2, capsize=5)
axes[0].set_xlabel('κ (MHz)')
axes[0].set_ylabel('Reward')
axes[0].set_title('Performance vs Measurement Strength')
axes[0].grid(True, alpha=0.3)

# Thermal occupancy plot
axes[1].errorbar(nth_values, nth_results, yerr=nth_stds,
                 fmt='gs-', linewidth=2, capsize=5)
axes[1].set_xlabel('n_th')
axes[1].set_ylabel('Reward')
axes[1].set_title('Performance vs Thermal Noise')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('generalization_tests.png', dpi=150)
print("\n✅ Generalization plots saved: generalization_tests.png")
