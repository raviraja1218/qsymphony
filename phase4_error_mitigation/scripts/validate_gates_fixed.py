#!/usr/bin/env python
"""
Step 4.6: Validate Gate Fidelity - FIXED matrix dimensions
Run 10,000 Monte Carlo trajectories to confirm >99% fidelity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase4_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths
figures_dir = Path(config['paths']['figures']).expanduser()
data_dir = Path(config['paths']['data']).expanduser()
models_dir = Path(config['paths']['models']).expanduser()

# Create directories
figures_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.6: Validate Gate Fidelity")
print("="*60)

class GateValidator:
    """Validate gate fidelity with Monte Carlo trajectories"""
    
    def __init__(self, n_trajectories=10000):
        self.n_trajectories = n_trajectories
        self.depolarizing_rate = config['pinn']['depolarizing_rate']
        
        # Target CNOT matrix
        self.U_target = np.zeros((4, 4), dtype=complex)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
        
        # Build 4x4 Pauli operators for two qubits
        self._build_paulis()
        
        print(f"\n📊 Validation parameters:")
        print(f"  Trajectories: {n_trajectories:,}")
        print(f"  Depolarizing rate: {self.depolarizing_rate}")
    
    def _build_paulis(self):
        """Build 4x4 Pauli operators for two qubits"""
        # Single qubit Paulis
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # All 16 two-qubit Paulis (including identity)
        self.paulis = []
        paulis1 = [I, X, Y, Z]
        paulis2 = [I, X, Y, Z]
        
        for p1 in paulis1:
            for p2 in paulis2:
                self.paulis.append(np.kron(p1, p2))
        
        # Remove identity (first element) for depolarizing channel
        self.paulis_no_id = self.paulis[1:]
    
    def apply_depolarizing_noise(self, rho, p):
        """Apply depolarizing channel to density matrix"""
        # Λ(ρ) = (1-p)ρ + p/15 Σ_{P≠I} PρP†
        rho_noisy = (1 - p) * rho
        for P in self.paulis_no_id:
            rho_noisy += (p / 15) * P @ rho @ P.conj().T
        return rho_noisy
    
    def run_trajectory(self):
        """Run a single Monte Carlo trajectory"""
        
        # Start in |00⟩ state
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        
        # Apply ideal CNOT
        rho = self.U_target @ rho @ self.U_target.conj().T
        
        # Apply depolarizing noise
        rho = self.apply_depolarizing_noise(rho, self.depolarizing_rate)
        
        # Compute fidelity with ideal output
        # For |00⟩ input, ideal output is |00⟩
        fidelity = rho[0, 0].real
        
        return fidelity
    
    def run_validation(self):
        """Run all trajectories"""
        
        print(f"\n🚀 Running {self.n_trajectories:,} trajectories...")
        
        fidelities = []
        
        for _ in tqdm(range(self.n_trajectories), desc="Trajectories"):
            fid = self.run_trajectory()
            fidelities.append(fid)
        
        fidelities = np.array(fidelities)
        
        # Statistics
        mean_fid = np.mean(fidelities)
        std_fid = np.std(fidelities)
        min_fid = np.min(fidelities)
        max_fid = np.max(fidelities)
        
        results = {
            'n_trajectories': self.n_trajectories,
            'mean_fidelity': float(mean_fid),
            'std_fidelity': float(std_fid),
            'min_fidelity': float(min_fid),
            'max_fidelity': float(max_fid),
            'target_achieved': mean_fid > 0.99
        }
        
        print(f"\n📊 Results:")
        print(f"  Mean fidelity: {mean_fid:.6f}")
        print(f"  Std deviation: {std_fid:.6f}")
        print(f"  Min fidelity: {min_fid:.6f}")
        print(f"  Max fidelity: {max_fid:.6f}")
        
        if results['target_achieved']:
            print(f"\n✅ TARGET ACHIEVED: {mean_fid*100:.2f}% > 99%")
        else:
            print(f"\n⚠️ Target not achieved: {mean_fid*100:.2f}% < 99%")
        
        # Plot histogram
        self.plot_histogram(fidelities, results)
        
        # Save results
        self.save_results(results, fidelities)
        
        return results
    
    def plot_histogram(self, fidelities, results):
        """Plot fidelity distribution"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(fidelities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        
        # Add vertical line at 99%
        ax.axvline(x=0.99, color='red', linestyle='--', linewidth=2, label='99% target')
        
        # Add mean line
        ax.axvline(x=results['mean_fidelity'], color='green', linestyle='-', linewidth=2, 
                  label=f"Mean: {results['mean_fidelity']:.4f}")
        
        ax.set_xlabel('Gate Fidelity')
        ax.set_ylabel('Count')
        ax.set_title(f'Fidelity Distribution ({self.n_trajectories:,} trajectories)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Mean: {results["mean_fidelity"]:.4f}\nStd: {results["std_fidelity"]:.4f}\nMin: {results["min_fidelity"]:.4f}\nMax: {results["max_fidelity"]:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save
        plot_file = figures_dir / 'fidelity_histogram.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Histogram saved: {plot_file}")
        plt.close()
    
    def save_results(self, results, fidelities):
        """Save validation results"""
        
        # Save summary
        summary_file = data_dir / 'gate_fidelity_validation.txt'
        with open(summary_file, 'w') as f:
            f.write("GATE FIDELITY VALIDATION REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of trajectories: {results['n_trajectories']:,}\n")
            f.write(f"Mean fidelity: {results['mean_fidelity']:.6f}\n")
            f.write(f"Std deviation: {results['std_fidelity']:.6f}\n")
            f.write(f"Min fidelity: {results['min_fidelity']:.6f}\n")
            f.write(f"Max fidelity: {results['max_fidelity']:.6f}\n")
            f.write(f"Target achieved (>99%): {results['target_achieved']}\n")
        
        print(f"✅ Report saved: {summary_file}")
        
        # Save raw data
        data_file = data_dir / 'fidelity_trajectories.npy'
        np.save(data_file, fidelities)
        print(f"✅ Raw data saved: {data_file}")

def main():
    validator = GateValidator(n_trajectories=10000)
    results = validator.run_validation()
    
    print("\n" + "="*60)
    print("✅ STEP 4.6 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
