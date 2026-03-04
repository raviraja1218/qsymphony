#!/usr/bin/env python
"""
STEP 4.1: Generate Synthetic Readout Data - ADVANCED CALIBRATION
Automatically finds noise levels to match target error rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
from scipy.optimize import minimize_scalar
from scipy.special import erfc

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
print("STEP 4.1: Generate Synthetic Readout Data - ADVANCED CALIBRATION")
print("="*60)

class AdvancedReadoutGenerator:
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
        
        self.n_samples = 50000  # More samples for better statistics
        self.signal = 3.0  # Larger signal for better separation
        
        print(f"\n📊 Target error rates:")
        for q, t in self.targets.items():
            print(f"  {q}: {t}%")
    
    def theoretical_error(self, noise, signal=3.0):
        """Theoretical error rate for given noise level"""
        # For two Gaussians centered at ±signal, error = erfc(signal/(√2*noise))
        return 50 * erfc(signal / (np.sqrt(2) * noise))
    
    def find_noise_for_target(self, target_error, signal=3.0):
        """Find noise level that gives target error rate"""
        def error_diff(noise):
            theo_error = self.theoretical_error(noise, signal)
            return abs(theo_error - target_error)
        
        result = minimize_scalar(error_diff, bounds=(0.1, 5.0), method='bounded')
        return result.x
    
    def calculate_error_rate(self, I0, I1):
        """Calculate error rate with optimal threshold"""
        # Combine data
        all_I = np.concatenate([I0, I1])
        all_labels = np.concatenate([np.zeros(len(I0)), np.ones(len(I1))])
        
        # Try many thresholds
        thresholds = np.linspace(-signal, signal, 1000)
        best_error = 100
        best_thresh = 0
        
        for thresh in thresholds:
            pred = (all_I > thresh).astype(int)
            error = np.mean(pred != all_labels) * 100
            if error < best_error:
                best_error = error
                best_thresh = thresh
        
        return best_error, best_thresh
    
    def calibrate_qubit(self, qubit_id, target_error, max_iterations=10):
        """Calibrate noise level for a specific qubit"""
        
        print(f"\n🔬 Calibrating {qubit_id} for {target_error}% error...")
        
        # Start with theoretical estimate
        noise = self.find_noise_for_target(target_error, self.signal)
        print(f"  Theoretical noise estimate: {noise:.4f}")
        
        history = []
        
        for iteration in range(max_iterations):
            # Generate test data
            n_test = 20000
            I0 = self.signal + np.random.normal(0, noise, n_test)
            I1 = -self.signal + np.random.normal(0, noise, n_test)
            
            # Calculate actual error
            error, _ = self.calculate_error_rate(I0, I1)
            history.append((noise, error))
            
            print(f"  Iter {iteration+1}: noise={noise:.4f}, error={error:.3f}%")
            
            if abs(error - target_error) < 0.1:
                print(f"  ✅ Calibrated at noise={noise:.4f}")
                break
            
            # Adjust noise based on error difference
            if error > target_error:
                noise *= 1.1  # Increase noise to increase error
            else:
                noise *= 0.9  # Decrease noise to decrease error
        
        return noise, history
    
    def generate_qubit_data(self, qubit_id, params, noise):
        """Generate final data for qubit with calibrated noise"""
        
        n_per_state = self.n_samples // 2
        
        # Generate I only (simplified 1D readout)
        I0 = self.signal + np.random.normal(0, noise, n_per_state)
        I1 = -self.signal + np.random.normal(0, noise, n_per_state)
        
        # Add small Q noise for 2D visualization
        Q0 = np.random.normal(0, noise/2, n_per_state)
        Q1 = np.random.normal(0, noise/2, n_per_state)
        
        # Calculate final error rate
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
    
    def plot_results(self, df, qubit_id, params, error_rate, threshold, history):
        """Plot calibration history and IQ data"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Calibration history
        ax = axes[0, 0]
        noises, errors = zip(*history)
        ax.plot(noises, errors, 'bo-', label='Actual error')
        ax.axhline(y=params['target'], color='r', linestyle='--', label='Target')
        ax.set_xlabel('Noise level')
        ax.set_ylabel('Error rate (%)')
        ax.set_title(f'{qubit_id} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # IQ Scatter
        ax = axes[0, 1]
        df0 = df[df['state'] == 0]
        df1 = df[df['state'] == 1]
        ax.scatter(df0['I'], df0['Q'], alpha=0.1, s=1, label='|0⟩', c='blue')
        ax.scatter(df1['I'], df1['Q'], alpha=0.1, s=1, label='|1⟩', c='red')
        ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'{qubit_id} IQ Scatter')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # I histogram
        ax = axes[1, 0]
        ax.hist(df0['I'], bins=50, alpha=0.5, label='|0⟩', color='blue', density=True)
        ax.hist(df1['I'], bins=50, alpha=0.5, label='|1⟩', color='red', density=True)
        ax.axvline(x=threshold, color='black', linestyle='--', label=f'thresh={threshold:.2f}')
        ax.set_xlabel('I')
        ax.set_ylabel('Density')
        ax.set_title(f'I-projection (error: {error_rate:.2f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Confusion matrix text
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate confusion matrix
        all_I = np.concatenate([df0['I'].values, df1['I'].values])
        all_labels = np.concatenate([np.zeros(len(df0)), np.ones(len(df1))])
        pred = (all_I > threshold).astype(int)
        
        cm = np.zeros((2, 2))
        for t, p in zip(all_labels, pred):
            cm[int(t), int(p)] += 1
        
        cm = cm.astype(int)
        
        # Display confusion matrix
        table_data = [
            ['', 'Pred 0', 'Pred 1'],
            ['True 0', cm[0,0], cm[0,1]],
            ['True 1', cm[1,0], cm[1,1]]
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        ax.set_title('Confusion Matrix', pad=20)
        
        plt.suptitle(f'{qubit_id} Readout Characterization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = figures_dir / f'{qubit_id.lower()}_advanced.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_all(self):
        """Generate all qubit data"""
        
        all_dfs = []
        results = []
        calibrated_noises = {}
        
        for qubit_id, params in self.qubits.items():
            # Calibrate noise for this qubit
            noise, history = self.calibrate_qubit(qubit_id, params['target'])
            calibrated_noises[qubit_id] = noise
            
            # Generate final data
            df, error_rate, threshold = self.generate_qubit_data(qubit_id, params, noise)
            
            # Plot results
            self.plot_results(df, qubit_id, params, error_rate, threshold, history)
            
            # Save data
            filename = iq_dir / f"{qubit_id.lower()}_iq_advanced.csv"
            df.to_csv(filename, index=False)
            print(f"  ✅ Saved to {filename}")
            
            all_dfs.append(df)
            results.append({
                'qubit': qubit_id,
                'target': params['target'],
                'achieved': error_rate,
                'noise': noise,
                'threshold': threshold
            })
        
        # Print summary
        print("\n" + "="*70)
        print("📊 FINAL CALIBRATION RESULTS")
        print("="*70)
        print(f"{'Qubit':<6} {'Target':<10} {'Achieved':<10} {'Noise':<10} {'Diff':<10}")
        print("-"*50)
        for r in results:
            diff = abs(r['achieved'] - r['target'])
            print(f"{r['qubit']:<6} {r['target']:<10.2f}% {r['achieved']:<10.2f}% "
                  f"{r['noise']:<10.4f} {diff:<10.2f}%")
        print("-"*50)
        
        avg_diff = np.mean([abs(r['achieved'] - r['target']) for r in results])
        print(f"\n✅ Average difference: {avg_diff:.2f}%")
        
        # Save calibration results
        cal_file = iq_dir.parent / 'metadata' / 'calibration_results.json'
        import json
        with open(cal_file, 'w') as f:
            json.dump({
                'noise_levels': calibrated_noises,
                'results': results
            }, f, indent=2)
        
        return pd.concat(all_dfs, ignore_index=True)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    generator = AdvancedReadoutGenerator()
    df_all = generator.generate_all()
    
    print("\n" + "="*60)
    print("✅ STEP 4.1 COMPLETE - Advanced calibrated data ready")
    print("="*60)

if __name__ == "__main__":
    main()
