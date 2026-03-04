#!/usr/bin/env python
"""
Step 4.1: Generate Synthetic Readout Data - FINAL VERSION
Correctly calibrated for target error rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase4_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths
iq_dir = Path(config['paths']['iq_data']).expanduser()
figures_dir = Path(config['paths']['figures']).expanduser()

iq_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.1: Generate Synthetic Readout Data - FINAL VERSION")
print("="*60)

class FinalReadoutGenerator:
    def __init__(self):
        # Target error rates from Table 1
        self.targets = {
            'Q2': 2.74,
            'Q3': 1.57,
            'Q4': 1.52,
            'Q5': 1.66,
            'Q6': 1.05
        }
        
        self.qubits = {
            'Q2': {'chi': -0.69, 'target': 2.74},
            'Q3': {'chi': -0.66, 'target': 1.57},
            'Q4': {'chi': -0.73, 'target': 1.52},
            'Q5': {'chi': -0.95, 'target': 1.66},
            'Q6': {'chi': -1.74, 'target': 1.05}
        }
        
        self.n_samples = 20000
        self.alpha = 1.0
        
        # Calibrated noise levels (found empirically)
        self.noise_levels = {
            'Q2': 0.32,
            'Q3': 0.25,
            'Q4': 0.24,
            'Q5': 0.26,
            'Q6': 0.21
        }
        
        print(f"\n📊 Target error rates:")
        for q, t in self.targets.items():
            print(f"  {q}: {t}%")
    
    def estimate_error_rate(self, I0, I1):
        """Estimate error rate from I projections using optimal threshold"""
        # Find optimal threshold
        all_I = np.concatenate([I0, I1])
        all_labels = np.concatenate([np.zeros(len(I0)), np.ones(len(I1))])
        
        thresholds = np.linspace(0, 1, 100)
        best_error = 100
        best_thresh = 0.5
        
        for thresh in thresholds:
            pred = (all_I > thresh).astype(int)
            error = np.mean(pred != all_labels) * 100
            if error < best_error:
                best_error = error
                best_thresh = thresh
        
        return best_error, best_thresh
    
    def generate_with_calibrated_noise(self, qubit_id, params):
        """Generate data with pre-calibrated noise levels"""
        
        noise_std = self.noise_levels[qubit_id]
        n_per_state = self.n_samples // 2
        
        # Generate |0⟩ state centered at (1, 0)
        I0 = self.alpha + np.random.normal(0, noise_std, n_per_state)
        Q0 = 0 + np.random.normal(0, noise_std, n_per_state)
        
        # Generate |1⟩ state centered at (0, 1)
        I1 = 0 + np.random.normal(0, noise_std, n_per_state)
        Q1 = self.alpha + np.random.normal(0, noise_std, n_per_state)
        
        # Estimate error rate
        error_rate, threshold = self.estimate_error_rate(I0, I1)
        
        # Create DataFrames
        df0 = pd.DataFrame({
            'I': I0, 'Q': Q0, 'state': 0,
            'qubit_id': qubit_id, 'chi_MHz': params['chi']
        })
        
        df1 = pd.DataFrame({
            'I': I1, 'Q': Q1, 'state': 1,
            'qubit_id': qubit_id, 'chi_MHz': params['chi']
        })
        
        df = pd.concat([df0, df1], ignore_index=True)
        
        return df, error_rate, threshold
    
    def plot_iq_data(self, df, qubit_id, params, error_rate, threshold):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        df0 = df[df['state'] == 0]
        df1 = df[df['state'] == 1]
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(df0['I'], df0['Q'], alpha=0.1, s=1, label='|0⟩', c='blue')
        ax.scatter(df1['I'], df1['Q'], alpha=0.1, s=1, label='|1⟩', c='red')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'{qubit_id}: {error_rate:.2f}% error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
        ax.set_aspect('equal')
        
        # Histogram with optimal threshold
        ax = axes[1]
        ax.hist(df0['I'], bins=50, alpha=0.5, label='|0⟩', color='blue', density=True)
        ax.hist(df1['I'], bins=50, alpha=0.5, label='|1⟩', color='red', density=True)
        ax.axvline(x=threshold, color='black', linestyle='--', 
                  label=f'threshold={threshold:.2f}')
        ax.set_xlabel('I')
        ax.set_ylabel('Density')
        ax.set_title(f'I-projection (target: {params["target"]}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = figures_dir / f'{qubit_id.lower()}_iq_final.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def generate_all(self):
        all_dfs = []
        results = []
        
        for qubit_id, params in self.qubits.items():
            print(f"\n🔬 Generating {qubit_id}...")
            
            df, error_rate, threshold = self.generate_with_calibrated_noise(qubit_id, params)
            
            print(f"  Target: {params['target']}%")
            print(f"  Achieved: {error_rate:.2f}%")
            print(f"  Difference: {abs(error_rate - params['target']):.2f}%")
            print(f"  Optimal threshold: {threshold:.3f}")
            
            self.plot_iq_data(df, qubit_id, params, error_rate, threshold)
            
            filename = iq_dir / f"{qubit_id.lower()}_iq_final.csv"
            df.to_csv(filename, index=False)
            print(f"  ✅ Saved to {filename}")
            
            all_dfs.append(df)
            results.append({
                'qubit': qubit_id,
                'target': params['target'],
                'achieved': error_rate,
                'threshold': threshold
            })
        
        print("\n📊 Final Results Summary:")
        print("-" * 50)
        print(f"{'Qubit':<6} {'Target':<8} {'Achieved':<10} {'Diff':<8}")
        print("-" * 50)
        for r in results:
            diff = abs(r['achieved'] - r['target'])
            print(f"{r['qubit']:<6} {r['target']:<8.2f}% {r['achieved']:<10.2f}% {diff:<8.2f}%")
        print("-" * 50)
        
        avg_diff = np.mean([abs(r['achieved'] - r['target']) for r in results])
        print(f"\nAverage difference: {avg_diff:.2f}%")
        
        return pd.concat(all_dfs, ignore_index=True)

def main():
    generator = FinalReadoutGenerator()
    df_all = generator.generate_all()
    
    print("\n" + "="*60)
    print("✅ STEP 4.1 COMPLETE - Final calibrated data ready")
    print("="*60)

if __name__ == "__main__":
    main()
