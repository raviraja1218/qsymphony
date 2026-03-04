#!/usr/bin/env python
"""
Step 4.1: Generate Synthetic Readout Data - CORRECTED VERSION
Proper IQ separation for low error rates
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
print("STEP 4.1: Generate Synthetic Readout Data - CORRECTED")
print("="*60)

class CorrectedReadoutGenerator:
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
            'Q2': {'chi': -0.69, 'target': 2.74, 'noise': 0.25},
            'Q3': {'chi': -0.66, 'target': 1.57, 'noise': 0.19},
            'Q4': {'chi': -0.73, 'target': 1.52, 'noise': 0.18},
            'Q5': {'chi': -0.95, 'target': 1.66, 'noise': 0.20},
            'Q6': {'chi': -1.74, 'target': 1.05, 'noise': 0.15}
        }
        
        self.n_samples = 20000
        self.signal = 2.0  # Signal amplitude
        
        print(f"\n📊 Target error rates:")
        for q, t in self.targets.items():
            print(f"  {q}: {t}%")
    
    def calculate_error_rate(self, I0, I1):
        """Calculate error rate with optimal threshold"""
        # Find best threshold
        thresholds = np.linspace(-2, 2, 200)
        best_error = 100
        best_thresh = 0
        
        for thresh in thresholds:
            errors0 = np.sum(I0 > thresh) / len(I0) * 100
            errors1 = np.sum(I1 < thresh) / len(I1) * 100
            error = (errors0 + errors1) / 2
            
            if error < best_error:
                best_error = error
                best_thresh = thresh
        
        return best_error, best_thresh
    
    def generate_qubit_data(self, qubit_id, params):
        """Generate data for a single qubit"""
        
        noise = params['noise']
        n_per_state = self.n_samples // 2
        
        # |0⟩ state at (+signal, 0)
        I0 = self.signal + np.random.normal(0, noise, n_per_state)
        Q0 = 0 + np.random.normal(0, noise, n_per_state)
        
        # |1⟩ state at (-signal, 0) - opposite side for maximum separation
        I1 = -self.signal + np.random.normal(0, noise, n_per_state)
        Q1 = 0 + np.random.normal(0, noise, n_per_state)
        
        # Calculate error rate
        error_rate, threshold = self.calculate_error_rate(I0, I1)
        
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
    
    def plot_data(self, df, qubit_id, params, error_rate, threshold):
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
        ax.set_xlim([-4, 4])
        ax.set_ylim([-2, 2])
        ax.set_aspect('equal')
        ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.5)
        
        # Histogram
        ax = axes[1]
        ax.hist(df0['I'], bins=50, alpha=0.5, label='|0⟩', color='blue', density=True)
        ax.hist(df1['I'], bins=50, alpha=0.5, label='|1⟩', color='red', density=True)
        ax.axvline(x=threshold, color='black', linestyle='--', label=f'thresh={threshold:.2f}')
        ax.set_xlabel('I')
        ax.set_ylabel('Density')
        ax.set_title(f'I-projection (target: {params["target"]}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = figures_dir / f'{qubit_id.lower()}_iq_corrected.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_all(self):
        all_dfs = []
        results = []
        
        for qubit_id, params in self.qubits.items():
            print(f"\n🔬 Generating {qubit_id}...")
            
            df, error_rate, threshold = self.generate_qubit_data(qubit_id, params)
            
            print(f"  Target: {params['target']}%")
            print(f"  Achieved: {error_rate:.2f}%")
            print(f"  Difference: {abs(error_rate - params['target']):.2f}%")
            print(f"  Threshold: {threshold:.2f}")
            
            self.plot_data(df, qubit_id, params, error_rate, threshold)
            
            filename = iq_dir / f"{qubit_id.lower()}_iq_corrected.csv"
            df.to_csv(filename, index=False)
            print(f"  ✅ Saved to {filename}")
            
            all_dfs.append(df)
            results.append({
                'qubit': qubit_id,
                'target': params['target'],
                'achieved': error_rate,
                'diff': abs(error_rate - params['target'])
            })
        
        # Print summary
        print("\n" + "="*60)
        print("📊 FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Qubit':<6} {'Target':<10} {'Achieved':<10} {'Diff':<10}")
        print("-"*40)
        for r in results:
            print(f"{r['qubit']:<6} {r['target']:<10.2f}% {r['achieved']:<10.2f}% {r['diff']:<10.2f}%")
        print("-"*40)
        
        avg_diff = np.mean([r['diff'] for r in results])
        print(f"\nAverage difference: {avg_diff:.2f}%")
        
        return pd.concat(all_dfs, ignore_index=True)

def main():
    generator = CorrectedReadoutGenerator()
    df_all = generator.generate_all()
    
    print("\n" + "="*60)
    print("✅ STEP 4.1 COMPLETE - Corrected data ready")
    print("="*60)

if __name__ == "__main__":
    main()
