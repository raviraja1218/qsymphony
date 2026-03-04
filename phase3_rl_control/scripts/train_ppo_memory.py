#!/usr/bin/env python
"""
PPO Training with Quantum Environment - MEMORY OPTIMIZED
Reduces batch size and clears memory frequently
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
import gc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from utils.ppo_network import PPOActorCritic, count_parameters
from utils.environment_wrapper_quantum import QuantumControlEnv

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase3_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class PPOTrainer:
    def __init__(self, env, config):
        self.env = env
        self.config = config['ppo']
        
        # Get observation dimension
        obs, _ = env.reset()
        self.obs_dim = len(obs)
        self.action_dim = env.action_space.shape[0]
        
        print(f"\n📊 Environment dimensions:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        
        # Initialize network with correct dimensions
        self.actor_critic = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config['policy_hidden_dim'],
            lstm_dim=self.config['lstm_hidden_dim']
        ).to(device)
        
        print(f"Model parameters: {count_parameters(self.actor_critic):,}")
        
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=float(self.config['learning_rate'])
        )
        
        # Clear buffers
        self.clear_buffers()
    
    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def select_action(self, obs, hidden_state=None):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, value, hidden_state = self.actor_critic.select_action(
                obs_tensor, hidden_state
            )
        return action.cpu().numpy()[0], log_prob.cpu(), value.cpu(), hidden_state
    
    def collect_episode(self):
        obs, _ = self.env.reset()
        hidden_state = None
        
        episode_obs = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        
        total_reward = 0
        
        for step in range(50000):
            episode_obs.append(obs)
            
            action, log_prob, value, hidden_state = self.select_action(obs, hidden_state)
            
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_values.append(value.item() if hasattr(value, 'item') else value)
            
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_dones.append(done)
            total_reward += reward
            
            if done:
                break
        
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        gamma = float(self.config['gamma'])
        lam = float(self.config['gae_lambda'])
        
        values = episode_values + [0]
        
        for t in reversed(range(len(episode_rewards))):
            delta = episode_rewards[t] + gamma * values[t+1] * (1 - episode_dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - episode_dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Store in buffers (as numpy arrays to save memory)
        self.states.extend(episode_obs)
        self.actions.extend(episode_actions)
        self.log_probs.extend([lp.numpy() for lp in episode_log_probs])
        self.rewards.extend(episode_rewards)
        self.dones.extend(episode_dones)
        self.values.extend(episode_values)
        
        return {
            'length': len(episode_rewards),
            'reward': total_reward,
            'advantages': advantages,
            'returns': returns
        }
    
    def update_policy(self, advantages, returns):
        # Convert to tensors in batches to save memory
        batch_size = 32  # Small batch size
        n_batches = (len(self.states) + batch_size - 1) // batch_size
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.states))
            
            # Get batch
            batch_states = torch.FloatTensor(np.array(self.states[start_idx:end_idx])).to(device)
            batch_actions = torch.FloatTensor(np.array(self.actions[start_idx:end_idx])).to(device)
            batch_old_log_probs = torch.cat(self.log_probs[start_idx:end_idx]).to(device)
            batch_advantages = torch.FloatTensor(advantages[start_idx:end_idx]).to(device)
            batch_returns = torch.FloatTensor(returns[start_idx:end_idx]).to(device)
            
            # Normalize advantages
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
            
            # Get new log probs and values
            new_values, new_log_probs, entropy = self.actor_critic.evaluate_actions(
                batch_states, batch_actions
            )
            
            # Compute ratios
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO clip loss
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - float(self.config['clip_param']), 
                                1 + float(self.config['clip_param'])) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
            
            # Total loss
            loss = policy_loss + float(self.config['value_coef']) * value_loss - \
                   float(self.config['entropy_coef']) * entropy.mean()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 
                                           float(self.config['max_grad_norm']))
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            
            # Clear batch tensors
            del batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        self.clear_buffers()
        
        return {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches
        }
    
    def train(self, total_timesteps):
        print(f"\n🚀 Starting training for {total_timesteps} timesteps...")
        
        timesteps_so_far = 0
        episode_num = 0
        start_time = time.time()
        
        while timesteps_so_far < total_timesteps:
            # Clear memory before each episode
            self.clear_buffers()
            
            episode_stats = self.collect_episode()
            timesteps_so_far += episode_stats['length']
            episode_num += 1
            
            update_stats = self.update_policy(episode_stats['advantages'], episode_stats['returns'])
            
            if episode_num % 5 == 0:
                elapsed = time.time() - start_time
                print(f"\n📊 Episode {episode_num} [{timesteps_so_far}/{total_timesteps}]")
                print(f"  Reward: {episode_stats['reward']:.4f}")
                print(f"  Length: {episode_stats['length']}")
                print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                print(f"  Time: {elapsed:.1f}s")
                
                # Save checkpoint
                if episode_num % 10 == 0:
                    self.save_checkpoint(f"model_ep{episode_num}.pt")
        
        total_time = time.time() - start_time
        print(f"\n✅ Training complete in {total_time:.2f}s")
        
        return total_time
    
    def save_checkpoint(self, filename):
        path = Path(config['paths']['rl_checkpoints']).expanduser() / filename
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"  ✅ Checkpoint saved: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100)
    args = parser.parse_args()
    
    print("="*60)
    print("PPO Training with Quantum Environment - MEMORY OPTIMIZED")
    print("="*60)
    
    # Clear GPU memory at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    env = QuantumControlEnv(mode='oracle')
    trainer = PPOTrainer(env, config)
    trainer.train(args.timesteps)
    
    # Save final model
    model_path = Path(config['paths']['oracle_model']).expanduser()
    torch.save(trainer.actor_critic.state_dict(), model_path)
    print(f"\n✅ Final model saved: {model_path}")

if __name__ == "__main__":
    main()
