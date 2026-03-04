#!/usr/bin/env python
"""
Train multiple seeds for statistical significance
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import json

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_ppo_phase3_fixed import FixedPPOTrainer
from utils.environment_wrapper_quantum import QuantumControlEnv

def train_all_seeds(n_seeds=5, n_episodes=3):
    """Train multiple seeds and collect statistics"""
    
    results = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*60}")
        print(f"Training seed {seed+1}/{n_seeds} (value: {seed+1000})")
        print(f"{'='*60}")
        
        env = QuantumControlEnv(mode='oracle')
        trainer = FixedPPOTrainer(env, seed=seed+1000)
        rewards, E_Ns = trainer.train(n_episodes=n_episodes)
        
        # Save model
        model_path = f"models/ppo_oracle_seed_{seed+1000}.pt"
        trainer.save_model(model_path)
        
        results.append({
            'seed': seed+1000,
            'rewards': rewards,
            'E_Ns': E_Ns,
            'final_E_N': E_Ns[-1],
            'mean_E_N': np.mean(E_Ns),
            'std_E_N': np.std(E_Ns)
        })
    
    return results

def create_summary_table(results):
    """Create markdown table for paper"""
    
    print("\n" + "="*70)
    print("📊 MULTIPLE SEEDS RESULTS - FOR PAPER")
    print("="*70)
    print(f"{'Seed':<8} {'Ep1 E_N':<12} {'Ep2 E_N':<12} {'Ep3 E_N':<12} {'Final':<10} {'Mean±Std':<20}")
    print("-"*70)
    
    all_final = []
    all_means = []
    
    for r in results:
        print(f"{r['seed']:<8} {r['E_Ns'][0]:<12.4f} {r['E_Ns'][1]:<12.4f} "
              f"{r['E_Ns'][2]:<12.4f} {r['final_E_N']:<10.4f} "
              f"{r['mean_E_N']:.4f}±{r['std_E_N']:.4f}")
        all_final.append(r['final_E_N'])
        all_means.append(r['mean_E_N'])
    
    print("-"*70)
    print(f"{'AVG':<8} {'':<12} {'':<12} {'':<12} "
          f"{np.mean(all_final):<10.4f} {np.mean(all_means):.4f}±{np.std(all_means):.4f}")
    print("="*70)
    
    # Save to CSV
    import csv
    with open('results/phase3/multiple_seeds_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Seed', 'Ep1_E_N', 'Ep2_E_N', 'Ep3_E_N', 'Final_E_N', 'Mean_E_N', 'Std_E_N'])
        for r in results:
            writer.writerow([r['seed'], r['E_Ns'][0], r['E_Ns'][1], r['E_Ns'][2], 
                           r['final_E_N'], r['mean_E_N'], r['std_E_N']])
    
    return np.mean(all_final), np.std(all_final)

def plot_learning_curve(results):
    """Plot learning curves with variance"""
    
    plt.figure(figsize=(10, 6))
    
    # Get max episodes
    max_ep = max(len(r['E_Ns']) for r in results)
    episodes = range(1, max_ep+1)
    
    # Collect data
    all_E_N = np.array([r['E_Ns'] for r in results])
    
    mean_E_N = np.mean(all_E_N, axis=0)
    std_E_N = np.std(all_E_N, axis=0)
    
    plt.plot(episodes, mean_E_N, 'b-', linewidth=2, label='Mean E_N')
    plt.fill_between(episodes, mean_E_N - std_E_N, mean_E_N + std_E_N, 
                     alpha=0.3, color='b', label='±1 std')
    
    plt.xlabel('Episode')
    plt.ylabel('E_N')
    plt.title('Learning Curve (5 seeds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/phase3/figures/learning_curve_multiple_seeds.png', dpi=150)
    print("\n✅ Learning curve saved")

def main():
    print("="*60)
    print("MULTIPLE SEEDS TRAINING - PHASE 3")
    print("="*60)
    
    # Train seeds
    results = train_all_seeds(n_seeds=5, n_episodes=3)
    
    # Create summary table
    mean_final, std_final = create_summary_table(results)
    
    # Plot learning curve
    plot_learning_curve(results)
    
    # Save results
    with open('results/phase3/data/multiple_seeds_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Average final E_N: {mean_final:.4f} ± {std_final:.4f}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
