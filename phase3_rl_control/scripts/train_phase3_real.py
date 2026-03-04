#!/usr/bin/env python
"""
PHASE 3 TRAINING - REAL PHYSICS with two-mode squeezing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_physics import PhysicsControlEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🎮 Using device: {device}")

class PhysicsPPOTrainer:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        obs, _ = env.reset()
        self.obs_dim = len(obs)
        self.action_dim = env.action_space.shape[0]
        
        print(f"   Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        
        self.policy = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            lstm_dim=128
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # PPO hyperparameters
        self.clip_param = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        self.results = []
    
    def collect_episode(self):
        """Collect one episode of experience"""
        obs, _ = self.env.reset()
        hidden_state = None
        episode_reward = 0
        step = 0
        E_Ns = []
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        while True:
            states.append(obs)
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, log_prob, value, hidden_state = self.policy.select_action(
                    obs_tensor, hidden_state, deterministic=False
                )
            
            action_np = action.cpu().numpy()[0]
            
            actions.append(action_np)
            log_probs.append(log_prob)
            values.append(value.item())
            
            obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            E_N = info.get('E_N', 0.0)
            
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            episode_reward += reward
            E_Ns.append(E_N)
            step += 1
            
            if terminated or truncated:
                break
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'advantages': advantages,
            'returns': returns,
            'episode_reward': episode_reward,
            'mean_E_N': np.mean(E_Ns),
            'max_E_N': np.max(E_Ns),
            'steps': step
        }
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        values = values + [0]  # Append 0 for final state
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def update_policy(self, episode_data):
        """Update policy using PPO"""
        states = torch.FloatTensor(np.array(episode_data['states'])).to(device)
        actions = torch.FloatTensor(np.array(episode_data['actions'])).to(device)
        old_log_probs = torch.stack(episode_data['log_probs']).to(device)
        advantages = torch.FloatTensor(episode_data['advantages']).to(device)
        returns = torch.FloatTensor(episode_data['returns']).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mini-batch updates
        batch_size = len(states)
        mini_batch_size = 64
        n_mini_batches = max(1, batch_size // mini_batch_size)
        
        for _ in range(10):  # 10 epochs
            indices = np.random.permutation(batch_size)
            
            for i in range(n_mini_batches):
                start = i * mini_batch_size
                end = min(start + mini_batch_size, batch_size)
                idx = indices[start:end]
                
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # Evaluate actions
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(
                    mb_states, mb_actions
                )
                
                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(new_values.squeeze(), mb_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, n_episodes=10):
        """Train for specified episodes"""
        print(f"\n   📈 Training seed {self.seed}...")
        
        for ep in range(n_episodes):
            episode_data = self.collect_episode()
            self.update_policy(episode_data)
            
            self.results.append({
                'episode': ep + 1,
                'reward': episode_data['episode_reward'],
                'mean_E_N': episode_data['mean_E_N'],
                'max_E_N': episode_data['max_E_N'],
                'steps': episode_data['steps']
            })
            
            print(f"   Episode {ep+1:2d}: Reward={episode_data['episode_reward']:7.2f}, "
                  f"Mean E_N={episode_data['mean_E_N']:.4f}, Max E_N={episode_data['max_E_N']:.4f}")
        
        return self.results

def main():
    print("="*70)
    print("PHASE 3 TRAINING - REAL PHYSICS")
    print("="*70)
    
    seeds = [1000, 1001, 1002]
    all_results = []
    
    start_time = time.time()
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'─'*50}")
        print(f"SEED {seed_idx+1}/{len(seeds)} (value: {seed})")
        print(f"{'─'*50}")
        
        env = PhysicsControlEnv(mode='oracle', seed=seed)
        trainer = PhysicsPPOTrainer(env, seed)
        results = trainer.train(n_episodes=5)
        
        # Save model
        model_path = f"models/ppo_physics_seed_{seed}.pt"
        torch.save({
            'model_state_dict': trainer.policy.state_dict(),
            'results': results,
            'seed': seed
        }, model_path)
        print(f"   ✅ Model saved: {model_path}")
        
        all_results.append({
            'seed': seed,
            'results': results
        })
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"{'Seed':<8} {'Final Mean E_N':<15} {'Final Max E_N':<15}")
    print("-"*50)
    
    for r in all_results:
        final = r['results'][-1]
        print(f"{r['seed']:<8} {final['mean_E_N']:<15.4f} {final['max_E_N']:<15.4f}")
    
    print("="*70)
    print(f"\n✅ Total training time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
