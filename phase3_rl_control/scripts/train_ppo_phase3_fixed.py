#!/usr/bin/env python
"""
PPO Training with CORRECT entanglement calculation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_quantum import QuantumControlEnv
from utils.entanglement_fixed import compute_log_negativity_correct

# Load config
with open('config/phase3_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FixedPPOTrainer:
    def __init__(self, env, seed=42):
        self.env = env
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get dimensions
        obs, _ = env.reset()
        self.obs_dim = len(obs)
        self.action_dim = env.action_space.shape[0]
        
        print(f"\n📊 Environment dims: obs={self.obs_dim}, action={self.action_dim}")
        
        # Network
        self.policy = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            lstm_dim=128
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Training stats
        self.episode_rewards = []
        self.episode_E_N = []
        
    def compute_reward(self, psi, n_q, n_m):
        """CORRECT reward function - no scaling, no clipping"""
        # Convert to density matrix if needed
        if psi.isket:
            rho = qt.ket2dm(psi)
        else:
            rho = psi
        
        # Compute entanglement
        E_N = compute_log_negativity_correct(rho)
        
        # Reward: maximize entanglement, minimize photons, stay near thermal
        n_th = 0.443
        reward = E_N - 0.1 * n_q - 0.05 * abs(n_m - n_th)
        
        return reward, E_N
    
    def train_episode(self):
        """Train one episode"""
        obs, _ = self.env.reset()
        hidden_state = None
        episode_reward = 0
        step = 0
        E_N_values = []
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, hidden_state = self.policy.select_action(
                    obs_tensor, hidden_state, deterministic=False
                )
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Get quantum state from environment (need to modify env to return psi)
            # For now, use info dict
            if 'psi' in info:
                psi = info['psi']
                n_q = info.get('n_q', 0)
                n_m = info.get('n_m', 0)
                corrected_reward, E_N = self.compute_reward(psi, n_q, n_m)
            else:
                corrected_reward = reward
                E_N = 0
            
            episode_reward += corrected_reward
            E_N_values.append(E_N)
            step += 1
            
            if terminated or truncated:
                break
        
        self.episode_rewards.append(episode_reward)
        self.episode_E_N.append(np.mean(E_N_values))
        
        return episode_reward, np.mean(E_N_values)
    
    def train(self, n_episodes=3):
        """Train for specified episodes"""
        print(f"\n🚀 Training seed {self.seed} for {n_episodes} episodes...")
        
        for ep in range(n_episodes):
            reward, E_N = self.train_episode()
            print(f"  Episode {ep+1}: Reward={reward:.2f}, Mean E_N={E_N:.4f}")
        
        return self.episode_rewards, self.episode_E_N
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_E_N': self.episode_E_N,
            'seed': self.seed
        }, path)
        print(f"✅ Model saved: {path}")

def main():
    print("="*60)
    print("PHASE 3 RETRAINING - CORRECT ENTANGLEMENT")
    print("="*60)
    
    seeds = [42, 123, 456]  # 3 seeds
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Training seed {seed}")
        print(f"{'='*50}")
        
        env = QuantumControlEnv(mode='oracle')
        trainer = FixedPPOTrainer(env, seed=seed)
        rewards, E_Ns = trainer.train(n_episodes=3)
        
        # Save seed model
        model_path = f"models/ppo_oracle_seed_{seed}.pt"
        trainer.save_model(model_path)
        
        all_results.append({
            'seed': seed,
            'rewards': rewards,
            'E_Ns': E_Ns
        })
    
    # Print summary table
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Seed':<6} {'Ep1 E_N':<10} {'Ep2 E_N':<10} {'Ep3 E_N':<10} {'Mean':<10}")
    print("-"*60)
    
    for r in all_results:
        mean_EN = np.mean(r['E_Ns'])
        print(f"{r['seed']:<6} {r['E_Ns'][0]:<10.4f} {r['E_Ns'][1]:<10.4f} "
              f"{r['E_Ns'][2]:<10.4f} {mean_EN:<10.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
