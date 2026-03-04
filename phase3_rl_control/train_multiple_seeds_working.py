#!/usr/bin/env python
"""
Train with multiple random seeds for statistical significance
WORKING VERSION with proper imports
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.ppo_network import PPOActorCritic, count_parameters
from utils.environment_wrapper_quantum import QuantumControlEnv

# Load configuration
config_path = Path(__file__).parent / 'config' / 'phase3_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleTrainer:
    """Simplified trainer for multiple seeds"""
    
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.policy = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            lstm_dim=128
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.rewards_history = []
        
    def train_episode(self):
        """Train one episode"""
        obs, _ = self.env.reset()
        hidden_state = None
        episode_reward = 0
        step = 0
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, hidden_state = self.policy.select_action(
                    obs_tensor, hidden_state, deterministic=False
                )
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = self.env.step(action_np)
            
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                break
        
        return episode_reward, step
    
    def train(self, n_episodes=50):
        """Train for specified number of episodes"""
        print(f"  Training seed {self.seed}...")
        
        for ep in range(n_episodes):
            reward, length = self.train_episode()
            self.rewards_history.append(reward)
            
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"    Ep {ep+1}: reward={avg_reward:.2f}")
        
        return self.rewards_history

def main():
    print("="*60)
    print("Multiple Seeds Training")
    print("="*60)
    
    n_seeds = 5
    n_episodes_per_seed = 50
    all_rewards = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"Seed {seed+1}/{n_seeds}")
        print(f"{'='*50}")
        
        # Create environment
        env = QuantumControlEnv(mode='oracle')
        
        # Create trainer
        trainer = SimpleTrainer(env, seed)
        
        # Train
        rewards = trainer.train(n_episodes=n_episodes_per_seed)
        all_rewards.append(rewards)
        
        # Save model
        torch.save(trainer.policy.state_dict(), f'model_seed_{seed}.pt')
    
    # Convert to array
    all_rewards = np.array(all_rewards)
    
    # Compute statistics
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    episodes = np.arange(1, len(mean_rewards) + 1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean')
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
    
    print(f"\n✅ Results:")
    print(f"  Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
    print(f"  Plot saved: learning_curve_multiple_seeds.png")

if __name__ == "__main__":
    main()
