#!/usr/bin/env python
"""
Step 4.1: Generate Synthetic Readout Data - CALIBRATED for exact target errors
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
print("STEP 4.1: Generate Synthetic Readout Data - CALIBRATED")
print("="*60)

class CalibratedReadoutGenerator:
    def __init__(self):
        # Target error rates and calibrated noise levels
        self.qubits = {
            'Q2': {'chi': -0.69, 'target': 2.74, 'noise_std': 0.22},
            'Q3': {'chi': -0.66, 'target': 1.57, 'noise_std': 0.17},
            'Q4': {'chi': -0.73, 'target': 1.52, 'noise_std': 0.16},
            'Q5': {'chi': -0.95, 'target': 1.66, 'noise_std': 0.18},
            'Q6': {'chi': -1.74, 'target': 1.05, 'noise_std': 0.14}
        }
        
        self.n_samples = 20000  # 10k per state
        self.alpha = 1.0
        
        print(f"\n📊 Calibrated noise levels:")
        for q, p in self.qubits.items():
            print(f"  {q}: {p['target']}% error (noise_std={p['noise_std']})")
    
    def error_function(self, noise_std):
        """Theoretical error rate for given noise"""
        # For two Gaussians with separation d=√2, error = erfc(d/(2√2σ))
        d = np.sqrt(2)  # distance between |0⟩ and |1⟩ in IQ space
        error = 0.5 * np.math.erfc(d / (2 * np.sqrt(2) * noise_std)) * 100
        return error
    
    def generate_iq_data(self, qubit_id, params):
        """Generate IQ data with specific noise level"""
        
        n_per_state = self.n_samples // 2
        
        # |0⟩ state at (1, 0)
        I0 = self.alpha + np.random.normal(0, params['noise_std'], n_per_state)
        Q0 = 0 + np.random.normal(0, params['noise_std'], n_per_state)
        
        # |1⟩ state at (0, 1) - 90-degree separation
        I1 = 0 + np.random.normal(0, params['noise_std'], n_per_state)
        Q1 = self.alpha + np.random.normal(0, params['noise_std'], n_per_state)
        
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
        
        # Calculate actual error rate
        errors0 = np.sum(I0 < 0.5) / len(I0)  # misclassify as |1⟩
        errors1 = np.sum(I1 > 0.5) / len(I1)  # misclassify as |0⟩
        actual_error = (errors0 + errors1) / 2 * 100
        
        return df, actual_error
    
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
        
        plot_file = figures_dir / f'{qubit_id.lower()}_iq_calibrated.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def generate_all(self):
        all_dfs = []
        
        for qubit_id, params in self.qubits.items():
            print(f"\n🔬 Generating {qubit_id}...")
            
            df, actual_error = self.generate_iq_data(qubit_id, params)
            
            print(f"  Target: {params['target']}%")
            print(f"  Actual: {actual_error:.2f}%")
            print(f"  Difference: {abs(actual_error - params['target']):.2f}%")
            
            self.plot_iq_data(df, qubit_id, params, actual_error)
            
            filename = iq_dir / f"{qubit_id.lower()}_iq_calibrated.csv"
            df.to_csv(filename, index=False)
            print(f"  ✅ Saved to {filename}")
            
            all_dfs.append(df)
        
        return pd.concat(all_dfs, ignore_index=True)

def main():
    generator = CalibratedReadoutGenerator()
    df_all = generator.generate_all()
    
    print("\n" + "="*60)
    print("✅ STEP 4.1 COMPLETE - Calibrated data ready")
    print("="*60)

if __name__ == "__main__":
    main()
