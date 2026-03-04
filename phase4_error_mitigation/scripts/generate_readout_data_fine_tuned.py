#!/usr/bin/env python
"""
Step 4.1: Generate Synthetic Readout Data - FINE TUNED for exact target errors
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
print("STEP 4.1: Generate Synthetic Readout Data - FINE TUNED")
print("="*60)

class FineTunedReadoutGenerator:
    def __init__(self):
        # Target error rates and empirically calibrated noise levels
        self.qubits = {
            'Q2': {'chi': -0.69, 'target': 2.74, 'noise_std': 0.35},
            'Q3': {'chi': -0.66, 'target': 1.57, 'noise_std': 0.28},
            'Q4': {'chi': -0.73, 'target': 1.52, 'noise_std': 0.27},
            'Q5': {'chi': -0.95, 'target': 1.66, 'noise_std': 0.29},
            'Q6': {'chi': -1.74, 'target': 1.05, 'noise_std': 0.23}
        }
        
        self.n_samples = 20000
        self.alpha = 1.0
        
        print(f"\n📊 Fine-tuned noise levels:")
        for q, p in self.qubits.items():
            print(f"  {q}: target {p['target']}% (noise_std={p['noise_std']})")
    
    def estimate_error_rate(self, I0, I1):
        """Estimate error rate from I projections"""
        # Simple threshold at 0.5
        errors0 = np.sum(I0 > 0.5) / len(I0) * 100
        errors1 = np.sum(I1 < 0.5) / len(I1) * 100
        return (errors0 + errors1) / 2
    
    def generate_with_feedback(self, qubit_id, params, max_iterations=5):
        """Generate data with iterative noise adjustment"""
        
        noise_std = params['noise_std']
        target = params['target']
        n_per_state = self.n_samples // 2
        
        for iteration in range(max_iterations):
            # Generate data
            I0 = self.alpha + np.random.normal(0, noise_std, n_per_state)
            Q0 = 0 + np.random.normal(0, noise_std, n_per_state)
            
            I1 = 0 + np.random.normal(0, noise_std, n_per_state)
            Q1 = self.alpha + np.random.normal(0, noise_std, n_per_state)
            
            # Estimate error
            error = self.estimate_error_rate(I0, I1)
            
            print(f"    Iter {iteration+1}: noise={noise_std:.3f}, error={error:.3f}%")
            
            # Adjust noise based on error
            if abs(error - target) < 0.1:
                break
            elif error > target:
                noise_std *= 1.1  # Increase noise to increase error
            else:
                noise_std *= 0.9   # Decrease noise to decrease error
        
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
        
        return df, error
    
    def plot_iq_data(self, df, qubit_id, params, actual_error):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        df0 = df[df['state'] == 0]
        df1 = df[df['state'] == 1]
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(df0['I'], df0['Q'], alpha=0.1, s=1, label='|0⟩', c='blue')
        ax.scatter(df1['I'], df1['Q'], alpha=0.1, s=1, label='|1⟩', c='red')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'{qubit_id}: {actual_error:.2f}% error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
        ax.set_aspect('equal')
        
        # Histogram
        ax = axes[1]
        ax.hist(df0['I'], bins=50, alpha=0.5, label='|0⟩', color='blue', density=True)
        ax.hist(df1['I'], bins=50, alpha=0.5, label='|1⟩', color='red', density=True)
        ax.axvline(x=0.5, color='black', linestyle='--', label='decision boundary')
        ax.set_xlabel('I')
        ax.set_ylabel('Density')
        ax.set_title(f'I-projection (target: {params["target"]}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = figures_dir / f'{qubit_id.lower()}_iq_fine_tuned.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def generate_all(self):
        all_dfs = []
        final_errors = {}
        
        for qubit_id, params in self.qubits.items():
            print(f"\n🔬 Generating {qubit_id}...")
            
            df, actual_error = self.generate_with_feedback(qubit_id, params)
            final_errors[qubit_id] = actual_error
            
            print(f"  Final: target={params['target']}%, actual={actual_error:.2f}%")
            
            self.plot_iq_data(df, qubit_id, params, actual_error)
            
            filename = iq_dir / f"{qubit_id.lower()}_iq_fine_tuned.csv"
            df.to_csv(filename, index=False)
            print(f"  ✅ Saved to {filename}")
            
            all_dfs.append(df)
        
        print("\n📊 Final Error Rates:")
        print("-" * 40)
        for qubit_id, params in self.qubits.items():
            print(f"  {qubit_id}: target {params['target']}% → achieved {final_errors[qubit_id]:.2f}%")
        print("-" * 40)
        
        return pd.concat(all_dfs, ignore_index=True)

def main():
    generator = FineTunedReadoutGenerator()
    df_all = generator.generate_all()
    
    print("\n" + "="*60)
    print("✅ STEP 4.1 COMPLETE - Fine-tuned data ready")
    print("="*60)

if __name__ == "__main__":
    main()
