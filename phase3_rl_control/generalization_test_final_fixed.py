#!/usr/bin/env python
"""
Test policy generalization across different conditions
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

# Load trained policy - FIXED PATH
model_path = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models' / 'ppo_measurement_final.zip'
print(f"Loading model from: {model_path}")

if not model_path.exists():
    print(f"❌ Model not found at {model_path}")
    print("Please complete Phase 3 first.")
    sys.exit(1)

checkpoint = torch.load(model_path, map_location=device)

policy = PPOActorCritic(obs_dim=17, action_dim=2).to(device)
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()
print("✅ Policy loaded successfully")

def evaluate_policy(env, policy, n_episodes=10):
    """Evaluate policy over multiple episodes"""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action, _, _, _ = policy.select_action(
                    torch.FloatTensor(obs).unsqueeze(0).to(device),
                    deterministic=True
                )
            obs, reward, done, _, _ = env.step(action.cpu().numpy()[0])
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# Test different initial states
print("\n1. Testing different initial states:")
states = ['ground', 'excited', 'superposition', 'thermal']
state_results = []
for state in states:
    # Create environment with specific initial state
    env = QuantumControlEnv(mode='measurement')
    # Set initial state (you'll need to implement this in your env)
    mean_r, std_r = evaluate_policy(env, policy)
    state_results.append(mean_r)
    print(f"  {state}: {mean_r:.2f} ± {std_r:.2f}")

# Test different κ values
print("\n2. Testing different measurement strengths:")
kappa_values = [25, 50, 75, 100]
kappa_results = []
for kappa in kappa_values:
    env = QuantumControlEnv(mode='measurement')
    # Set kappa (you'll need to implement this)
    mean_r, std_r = evaluate_policy(env, policy)
    kappa_results.append(mean_r)
    print(f"  κ={kappa}MHz: {mean_r:.2f} ± {std_r:.2f}")

# Test different temperatures
print("\n3. Testing different thermal occupancies:")
nth_values = [0.1, 0.3, 0.5, 0.7, 1.0]
nth_results = []
for nth in nth_values:
    env = QuantumControlEnv(mode='measurement')
    # Set n_th (you'll need to implement this)
    mean_r, std_r = evaluate_policy(env, policy)
    nth_results.append(mean_r)
    print(f"  n_th={nth}: {mean_r:.2f} ± {std_r:.2f}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

if state_results:
    axes[0].bar(states, state_results, color='blue', alpha=0.7)
    axes[0].set_xlabel('Initial State')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Performance vs Initial State')
    axes[0].grid(True, alpha=0.3, axis='y')

if kappa_results:
    axes[1].plot(kappa_values, kappa_results, 'ro-', linewidth=2)
    axes[1].set_xlabel('κ (MHz)')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Performance vs Measurement Strength')
    axes[1].grid(True, alpha=0.3)

if nth_results:
    axes[2].plot(nth_values, nth_results, 'gs-', linewidth=2)
    axes[2].set_xlabel('n_th')
    axes[2].set_ylabel('Reward')
    axes[2].set_title('Performance vs Thermal Noise')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('generalization_tests.png', dpi=150)
print("\n✅ Generalization plots saved: generalization_tests.png")
