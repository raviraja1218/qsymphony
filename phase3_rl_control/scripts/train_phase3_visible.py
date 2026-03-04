#!/usr/bin/env python
"""
PHASE 3 TRAINING - VISIBLE PROGRESS
Shows real-time output for each seed and episode
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import time
from datetime import datetime
import qutip as qt

sys.path.append(str(Path(__file__).parent.parent))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_quantum import QuantumControlEnv

# CORRECT entanglement calculation
def compute_log_negativity(rho):
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    trace_norm = np.sum(np.abs(evals))
    return np.log2(trace_norm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🎮 Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

class PPOTrainer:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        obs, _ = env.reset()
        self.obs_dim = len(obs)
        self.action_dim = env.action_space.shape[0]
        
        self.policy = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            lstm_dim=128
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.results = []
    
    def train_episode(self, ep_num):
        """Train one episode with visible progress"""
        obs, _ = self.env.reset()
        hidden_state = None
        episode_reward = 0
        step = 0
        E_Ns = []
        
        print(f"      Step: ", end="")
        
        while True:
            if step % 5000 == 0:
                print(f"{step/1000:.0f}k ", end="", flush=True)
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, hidden_state = self.policy.select_action(
                    obs_tensor, hidden_state, deterministic=False
                )
            
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = self.env.step(action_np)
            
            # Get quantum state from environment
            if hasattr(self.env.unwrapped, 'psi'):
                psi = self.env.unwrapped.psi
                rho = qt.ket2dm(psi)
                E_N = compute_log_negativity(rho)
            else:
                E_N = 0
            
            # Correct reward
            n_q = obs[10] if len(obs) > 10 else 0
            n_m = obs[11] if len(obs) > 11 else 0
            n_th = 0.443
            
            corrected_reward = E_N - 0.1 * n_q - 0.05 * abs(n_m - n_th)
            
            episode_reward += corrected_reward
            E_Ns.append(E_N)
            step += 1
            
            if terminated or truncated:
                print(f" done")
                break
        
        mean_E_N = np.mean(E_Ns)
        final_E_N = E_Ns[-1] if E_Ns else 0
        
        self.results.append({
            'episode': ep_num,
            'reward': episode_reward,
            'mean_E_N': mean_E_N,
            'final_E_N': final_E_N
        })
        
        return episode_reward, mean_E_N, final_E_N
    
    def train(self, n_episodes=3):
        print(f"\n   📈 Training seed {self.seed}...")
        
        for ep in range(n_episodes):
            print(f"   Episode {ep+1}/{n_episodes}:")
            reward, mean_EN, final_EN = self.train_episode(ep+1)
            print(f"      → Reward: {reward:.2f}, Mean E_N: {mean_EN:.4f}, Final E_N: {final_EN:.4f}")
        
        return self.results

def main():
    print("\n" + "="*70)
    print("PHASE 3 TRAINING - REAL-TIME PROGRESS")
    print("="*70)
    
    seeds = [1000, 1001, 1002]  # 3 seeds for faster demo
    all_results = []
    
    start_time = time.time()
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'─'*50}")
        print(f"SEED {seed_idx+1}/{len(seeds)} (value: {seed})")
        print(f"{'─'*50}")
        
        env = QuantumControlEnv(mode='oracle')
        trainer = PPOTrainer(env, seed)
        results = trainer.train(n_episodes=3)
        
        # Save model
        model_path = f"models/ppo_oracle_seed_{seed}.pt"
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
    
    # Summary table
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"{'Seed':<8} {'Ep1 Final':<12} {'Ep2 Final':<12} {'Ep3 Final':<12} {'Mean E_N':<12}")
    print("-"*70)
    
    all_finals = []
    for r in all_results:
        finals = [r['results'][i]['final_E_N'] for i in range(3)]
        mean_final = np.mean(finals)
        all_finals.extend(finals)
        print(f"{r['seed']:<8} {finals[0]:<12.4f} {finals[1]:<12.4f} {finals[2]:<12.4f} {mean_final:<12.4f}")
    
    print("-"*70)
    print(f"{'ALL':<8} {'':<12} {'':<12} {'':<12} {np.mean(all_finals):<12.4f} ± {np.std(all_finals):<12.4f}")
    print("="*70)
    print(f"\n✅ Total training time: {total_time/60:.1f} minutes")
    print("="*70)

if __name__ == "__main__":
    main()
