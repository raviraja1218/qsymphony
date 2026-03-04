#!/usr/bin/env python
"""
Train PPO with multiple random seeds for statistical significance
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from scripts.train_ppo_quantum import PPOTrainer
from utils.environment_wrapper_quantum import QuantumControlEnv

def train_with_seeds(n_seeds=5, timesteps=100000):
    """Train with multiple seeds"""
    
    all_rewards = []
    all_lengths = []
    
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
        results = trainer.train(timesteps)
        
        all_rewards.append(results['rewards'])
        all_lengths.append(results['lengths'])
    
    # Convert to arrays
    all_rewards = np.array(all_rewards)
    all_lengths = np.array(all_lengths)
    
    # Compute statistics
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    # Plot learning curve with variance
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(mean_rewards))
    
    plt.plot(episodes, mean_rewards, 'b-', label='Mean reward')
    plt.fill_between(episodes, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     alpha=0.3, color='b', label='±1 std')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'PPO Learning Curve over {n_seeds} seeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curve_multiple_seeds.png', dpi=150)
    
    # Print statistics
    print("\n" + "="*60)
    print("MULTIPLE SEED STATISTICS")
    print("="*60)
    print(f"Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"Peak mean reward: {np.max(mean_rewards):.2f}")
    print(f"Average episode length: {np.mean(all_lengths):.0f}")
    
    return all_rewards, all_lengths

if __name__ == "__main__":
    train_with_seeds(n_seeds=5, timesteps=100000)
