#!/usr/bin/env python
"""
PPO Training with Quantum Environment - FINAL FIX
Automatically uses the actual observation dimension from environment
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import json
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from utils.ppo_network import PPOActorCritic, count_parameters
from utils.environment_wrapper_quantum import QuantumControlEnv
from utils.reward_functions import OracleReward

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase3_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PPOTrainer:
    def __init__(self, env, config, mode='oracle'):
        self.env = env
        self.config = config['ppo']
        self.mode = mode
        
        # Get dimensions from environment - THIS IS THE KEY FIX
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        print(f"\n📊 Environment dimensions:")
        print(f"  Observation dim: {self.obs_dim}")  # Will be 17
        print(f"  Action dim: {self.action_dim}")
        
        # Initialize networks with CORRECT observation dimension
        self.actor_critic = PPOActorCritic(
            obs_dim=self.obs_dim,  # Use actual dimension (17)
            action_dim=self.action_dim,
            hidden_dim=self.config['policy_hidden_dim'],
            lstm_dim=self.config['lstm_hidden_dim']
        ).to(device)
        
        print(f"Model parameters: {count_parameters(self.actor_critic):,}")
        
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=float(self.config['learning_rate'])
        )
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.hidden_states = []
    
    def compute_gae(self, rewards, values, dones):
        gamma = float(self.config['gamma'])
        lam = float(self.config['gae_lambda'])
        advantages = []
        gae = 0
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def collect_episode(self, max_steps=50000):
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        
        hidden_state = None
        total_reward = 0
        
        for step in range(max_steps):
            episode_states.append(obs.cpu())
            
            with torch.no_grad():
                action, log_prob, value, hidden_state = self.actor_critic.select_action(
                    obs, hidden_state
                )
            
            episode_actions.append(action.cpu())
            episode_log_probs.append(log_prob.cpu())
            episode_values.append(value.cpu())
            
            action_np = action.squeeze().cpu().numpy()
            obs_np, reward, terminated, truncated, _ = self.env.step(action_np)
            
            obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_dones.append(done)
            total_reward += reward
            
            if done:
                break
        
        advantages, returns = self.compute_gae(episode_rewards, episode_values, episode_dones)
        
        self.states.extend(episode_states)
        self.actions.extend(episode_actions)
        self.log_probs.extend(episode_log_probs)
        self.rewards.extend(episode_rewards)
        self.dones.extend(episode_dones)
        self.values.extend(episode_values)
        self.hidden_states.extend([hidden_state] * len(episode_states))
        
        return {
            'length': len(episode_rewards),
            'reward': total_reward,
            'advantages': advantages,
            'returns': returns
        }
    
    def update_policy(self, advantages, returns):
        states = torch.cat(self.states).to(device)
        actions = torch.cat(self.actions).to(device)
        old_log_probs = torch.cat(self.log_probs).to(device)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = len(states)
        mini_batch_size = int(self.config['mini_batch_size'])
        n_mini_batches = max(1, batch_size // mini_batch_size)
        
        for epoch in range(int(self.config['epochs_per_update'])):
            indices = np.random.permutation(batch_size)
            
            for i in range(n_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                idx = indices[start:end]
                
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                new_values, new_log_probs, entropy = self.actor_critic.evaluate_actions(
                    mb_states, mb_actions
                )
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - float(self.config['clip_param']), 
                                    1 + float(self.config['clip_param'])) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(new_values.squeeze(), mb_returns)
                
                loss = policy_loss + float(self.config['value_coef']) * value_loss - \
                       float(self.config['entropy_coef']) * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                               float(self.config['max_grad_norm']))
                self.optimizer.step()
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.hidden_states = []
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }
    
    def train(self, total_timesteps):
        print(f"\n🚀 Starting training for {total_timesteps} timesteps...")
        
        timesteps_so_far = 0
        episode_num = 0
        
        while timesteps_so_far < total_timesteps:
            episode_stats = self.collect_episode()
            timesteps_so_far += episode_stats['length']
            episode_num += 1
            
            update_stats = self.update_policy(episode_stats['advantages'], episode_stats['returns'])
            
            if episode_num % 10 == 0:
                print(f"\n📊 Episode {episode_num} [{timesteps_so_far}/{total_timesteps}]")
                print(f"  Reward: {episode_stats['reward']:.4f}")
                print(f"  Length: {episode_stats['length']}")
                print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
        
        return {'total_time': time.time() - start_time}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100)
    args = parser.parse_args()
    
    print("="*60)
    print("PPO Training with Quantum Environment")
    print("="*60)
    
    env = QuantumControlEnv(mode='oracle')
    trainer = PPOTrainer(env, config)
    trainer.train(args.timesteps)
    
    # Save model
    model_path = Path(config['paths']['oracle_model']).expanduser()
    torch.save(trainer.actor_critic.state_dict(), model_path)
    print(f"\n✅ Model saved: {model_path}")

if __name__ == "__main__":
    main()
