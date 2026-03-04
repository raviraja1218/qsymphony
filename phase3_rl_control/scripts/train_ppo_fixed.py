#!/usr/bin/env python
"""
Step 3.1 & 3.2: PPO Training with LSTM - Oracle Phase - FIXED
Main training script for Proximal Policy Optimization
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
from datetime import datetime
import json
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from utils.ppo_network import PPOActorCritic, count_parameters
from utils.environment_wrapper_quantum import QuantumControlEnv
from utils.reward_functions import OracleReward

# QuTiP for entanglement calculation (if available)
try:
    import qutip as qt
except ImportError:
    print("Warning: QuTiP not available - using mock entanglement")
    qt = None

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase3_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Convert string numeric values to float - FIXED
def convert_numeric_strings(d):
    """Recursively convert string numbers to float"""
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str):
                try:
                    # Try to convert to float if it looks like a number
                    if v.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                        d[k] = float(v)
                except:
                    pass
            elif isinstance(v, dict):
                convert_numeric_strings(v)
    return d

config = convert_numeric_strings(config)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class PPOTrainer:
    """PPO Trainer with LSTM for quantum control"""
    
    def __init__(self, env, config, mode='oracle'):
        self.env = env
        self.config = config['ppo']
        self.mode = mode
        
        # Get dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        print(f"\n📊 Environment dimensions:")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        
        # Initialize networks
        self.actor_critic = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config['policy_hidden_dim'],
            lstm_dim=self.config['lstm_hidden_dim']
        ).to(device)
        
        print(f"Model parameters: {count_parameters(self.actor_critic):,}")
        
        # Ensure learning rate is float
        lr = float(self.config['learning_rate'])
        print(f"Learning rate: {lr}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=lr
        )
        
        # Reward function
        if mode == 'oracle':
            self.reward_fn = OracleReward(
                lambda_photon=float(config['reward']['lambda_photon']),
                mu_thermal=float(config['reward']['mu_thermal']),
                target_ent=float(config['reward']['entanglement_target'])
            )
        else:
            # Will be set later with golden path
            self.reward_fn = None
        
        # Training buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.hidden_states = []
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_entanglements = []
        
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = values + [0]  # Append 0 for final state
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def collect_episode(self, max_steps=50000):
        """Collect one episode of experience"""
        
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_dones = []
        episode_values = []
        episode_hidden = []
        
        hidden_state = None
        total_reward = 0
        
        for step in range(max_steps):
            # Store state
            episode_states.append(obs.cpu())
            
            # Select action
            with torch.no_grad():
                action, log_prob, value, hidden_state = self.actor_critic.select_action(
                    obs, hidden_state
                )
            
            # Store
            episode_actions.append(action.cpu())
            episode_log_probs.append(log_prob.cpu())
            episode_values.append(value.cpu())
            episode_hidden.append(hidden_state)
            
            # Step environment
            action_np = action.squeeze().cpu().numpy()
            obs_np, reward, terminated, truncated, info = self.env.step(action_np)
            
            obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_dones.append(done)
            total_reward += reward
            
            if done:
                break
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            episode_rewards, episode_values, episode_dones,
            gamma=float(self.config['gamma']),
            lam=float(self.config['gae_lambda'])
        )
        
        # Store in main buffers
        self.states.extend(episode_states)
        self.actions.extend(episode_actions)
        self.log_probs.extend(episode_log_probs)
        self.rewards.extend(episode_rewards)
        self.dones.extend(episode_dones)
        self.values.extend(episode_values)
        self.hidden_states.extend(episode_hidden)
        
        return {
            'length': len(episode_rewards),
            'reward': total_reward,
            'advantages': advantages,
            'returns': returns
        }
    
    def update_policy(self, advantages, returns):
        """Update policy using PPO clipped objective"""
        
        # Convert buffers to tensors
        states = torch.cat(self.states).to(device)
        actions = torch.cat(self.actions).to(device)
        old_log_probs = torch.cat(self.log_probs).to(device)
        
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch updates
        batch_size = len(states)
        mini_batch_size = int(self.config['mini_batch_size'])
        n_mini_batches = batch_size // mini_batch_size
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(int(self.config['epochs_per_update'])):
            # Shuffle indices
            indices = np.random.permutation(batch_size)
            
            for i in range(n_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = start_idx + mini_batch_size
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Evaluate actions
                new_values, new_log_probs, entropy = self.actor_critic.evaluate_actions(
                    mb_states, mb_actions
                )
                
                # Probability ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - float(self.config['clip_param']), 
                                    1 + float(self.config['clip_param'])) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(new_values.squeeze(), mb_returns)
                
                # Total loss
                loss = policy_loss + float(self.config['value_coef']) * value_loss - \
                       float(self.config['entropy_coef']) * entropy.mean()
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 
                    float(self.config['max_grad_norm'])
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.hidden_states = []
        
        n_updates = max(1, n_mini_batches * int(self.config['epochs_per_update']))
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
    
    def train(self, total_timesteps, log_frequency=50):
        """Main training loop"""
        
        print(f"\n🚀 Starting {self.mode} training for {total_timesteps} timesteps...")
        
        timesteps_so_far = 0
        episode_num = 0
        best_reward = -np.inf
        
        # Training stats
        all_rewards = []
        all_lengths = []
        all_losses = []
        
        start_time = time.time()
        
        while timesteps_so_far < total_timesteps:
            # Collect episode
            episode_stats = self.collect_episode()
            
            timesteps_so_far += episode_stats['length']
            episode_num += 1
            
            all_rewards.append(episode_stats['reward'])
            all_lengths.append(episode_stats['length'])
            
            # Update policy
            update_stats = self.update_policy(
                episode_stats['advantages'],
                episode_stats['returns']
            )
            all_losses.append(update_stats)
            
            # Log progress
            if episode_num % log_frequency == 0:
                avg_reward = np.mean(all_rewards[-log_frequency:])
                avg_length = np.mean(all_lengths[-log_frequency:])
                elapsed = time.time() - start_time
                
                print(f"\n📊 Episode {episode_num} [{timesteps_so_far}/{total_timesteps}]")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Avg Length: {avg_length:.0f}")
                print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                print(f"  Entropy: {update_stats['entropy']:.4f}")
                print(f"  Time: {elapsed/60:.1f} min")
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_checkpoint(f"best_model_ep{episode_num}.pt")
                    print(f"  ✅ New best model saved!")
        
        total_time = time.time() - start_time
        print(f"\n✅ Training complete in {total_time/60:.2f} minutes")
        
        return {
            'rewards': all_rewards,
            'lengths': all_lengths,
            'losses': all_losses,
            'total_time': total_time
        }
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = Path(config['paths']['rl_checkpoints']).expanduser() / filename
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        print(f"✅ Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = Path(filename).expanduser()
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Model loaded from {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='oracle', 
                        choices=['oracle', 'measurement'])
    parser.add_argument('--load', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps to train')
    args = parser.parse_args()
    
    print("="*60)
    print(f"PPO Training - {args.mode.upper()} Phase")
    print("="*60)
    
    # Create environment
    env = QuantumControlEnv(
        mode=args.mode,
        golden_path_file=config['paths']['golden_path'] if args.mode == 'measurement' else None
    )
    
    # Create trainer
    trainer = PPOTrainer(env, config, mode=args.mode)
    
    # Load checkpoint if specified
    if args.load:
        trainer.load_checkpoint(args.load)
    
    # Set timesteps
    if args.timesteps is None:
        timesteps = config[args.mode]['total_timesteps']
    else:
        timesteps = args.timesteps
    
    # Train
    results = trainer.train(timesteps)
    
    # Save final model
    if args.mode == 'oracle':
        model_path = Path(config['paths']['oracle_model']).expanduser()
    else:
        model_path = Path(config['paths']['measurement_model']).expanduser()
    
    trainer.save_checkpoint(model_path)
    print(f"\n✅ Final model saved: {model_path}")
    
    # Save training results
    results_path = Path(config['paths']['data']).expanduser() / f'training_results_{args.mode}.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists
        results_serializable = {
            'rewards': [float(r) for r in results['rewards']],
            'lengths': [int(l) for l in results['lengths']],
            'total_time': results['total_time']
        }
        json.dump(results_serializable, f, indent=2)
    print(f"✅ Training results saved: {results_path}")

if __name__ == "__main__":
    main()
