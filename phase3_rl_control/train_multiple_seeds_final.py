#!/usr/bin/env python
"""
Train with multiple random seeds for statistical significance
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from scripts.train_ppo_quantum import PPOTrainer
from utils.environment_wrapper_quantum import QuantumControlEnv
import yaml

# Load config
with open('config/phase3_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

n_seeds = 5
all_rewards = []
all_fidelities = []

for seed in range(n_seeds):
    print(f"\n{'='*60}")
    print(f"Training seed {seed+1}/{n_seeds}")
    print(f"{'='*60}")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment and trainer
    env = QuantumControlEnv(mode='oracle')
    trainer = PPOTrainer(env, config, mode='oracle')
    
    # Train
    results = trainer.train(100000)  # 100k timesteps
    
    all_rewards.append(results['rewards'])
    
    # Save individual seed results
    torch.save(trainer.actor_critic.state_dict(), f'model_seed_{seed}.pt')

# Plot learning curves with variance
plt.figure(figsize=(12, 6))

# Align lengths
min_len = min(len(r) for r in all_rewards)
rewards_array = np.array([r[:min_len] for r in all_rewards])

mean_rewards = np.mean(rewards_array, axis=0)
std_rewards = np.std(rewards_array, axis=0)
episodes = np.arange(min_len)

plt.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean')
plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                 alpha=0.3, color='b', label='±1 std')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'PPO Learning Curve over {n_seeds} seeds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve_multiple_seeds.png', dpi=150)

print(f"\n✅ Results: Mean final reward = {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
print("✅ Plot saved: learning_curve_multiple_seeds.png")
