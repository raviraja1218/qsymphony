#!/usr/bin/env python
"""
Step 4.1: Generate Synthetic Readout Data
Create IQ voltage datasets with noise for multiple qubit variants
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase4_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Expand paths
iq_dir = Path(config['paths']['iq_data']).expanduser()
metadata_dir = Path(config['paths']['metadata']).expanduser()
figures_dir = Path(config['paths']['figures']).expanduser()

# Create directories
iq_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.1: Generate Synthetic Readout Data")
print("="*60)

class ReadoutDataGenerator:
    """Generate synthetic IQ data for dispersive readout"""
    
    def __init__(self, config):
        self.config = config
        self.qubits = config['qubits']
        self.readout = config['readout']
        self.noise = config['noise']
        
        # Constants
        self.wr = 2 * np.pi * self.readout['resonator_freq_GHz'] * 1e9
        self.kappa = 2 * np.pi * self.readout['kappa_MHz'] * 1e6
        
        # Convert SNR dB to linear
        self.snr_linear = 10 ** (self.readout['snr_db'] / 20)
        
        print(f"\n📊 Readout parameters:")
        print(f"  Resonator frequency: {self.readout['resonator_freq_GHz']} GHz")
        print(f"  κ/2π: {self.readout['kappa_MHz']} MHz")
        print(f"  Measurement time: {self.readout['measurement_time_ns']} ns")
        print(f"  SNR: {self.readout['snr_db']} dB")
        print(f"  Samples per state: {self.readout['n_samples_per_state']}")
    
    def generate_iq_for_state(self, qubit_id, chi_MHz, state, n_samples):
        """
        Generate IQ samples for a given qubit state
        
        Args:
            qubit_id: Q2, Q3, etc.
            chi_MHz: dispersive shift in MHz
            state: 0 or 1 (ground/excited)
            n_samples: number of samples to generate
        
        Returns:
            DataFrame with I, Q columns
        """
        # Convert chi to angular frequency
        chi = 2 * np.pi * chi_MHz * 1e6
        
        # Resonator frequency shift based on state
        if state == 0:
            wr_eff = self.wr - chi/2
            phase_shift = 0
        else:
            wr_eff = self.wr + chi/2
            phase_shift = np.pi/4  # Typical phase shift for |1>
        
        # Signal amplitude
        alpha = self.readout['amplitude'] * self.snr_linear
        
        # Generate ideal I/Q
        t = np.linspace(0, self.readout['measurement_time_ns'] * 1e-9, n_samples)
        I_ideal = alpha * np.cos(self.wr * t + phase_shift)
        Q_ideal = alpha * np.sin(self.wr * t + phase_shift)
        
        # Average over measurement time
        I_mean = np.mean(I_ideal)
        Q_mean = np.mean(Q_ideal)
        
        # Generate samples with noise
        I_samples = []
        Q_samples = []
        
        for _ in range(n_samples):
            # White noise
            I_noise = np.random.normal(0, self.noise['white_noise_std'])
            Q_noise = np.random.normal(0, self.noise['white_noise_std'])
            
            # 1/f noise (simplified as correlated noise)
            flicker = np.random.normal(0, self.noise['flicker_noise_fraction'])
            I_noise += flicker * I_mean
            Q_noise += flicker * Q_mean
            
            # Crosstalk (small rotation)
            theta_cross = np.deg2rad(self.noise['crosstalk_rotation_deg'])
            I_rot = I_mean * np.cos(theta_cross) - Q_mean * np.sin(theta_cross)
            Q_rot = I_mean * np.sin(theta_cross) + Q_mean * np.cos(theta_cross)
            
            # Final sample
            I_samples.append(I_rot + I_noise)
            Q_samples.append(Q_rot + Q_noise)
        
        # Create DataFrame
        df = pd.DataFrame({
            'I': I_samples,
            'Q': Q_samples,
            'state': state,
            'qubit_id': qubit_id,
            'chi_MHz': chi_MHz
        })
        
        return df
    
    def generate_all_data(self):
        """Generate data for all qubits and states"""
        
        all_dfs = []
        
        for qubit_id, params in self.qubits.items():
            chi_MHz = params['chi_MHz']
            t1_us = params['t1_us']
            t2_us = params['t2_us']
            
            print(f"\n🔬 Generating data for {qubit_id}:")
            print(f"  χ/2π = {chi_MHz} MHz")
            print(f"  T₁ = {t1_us} μs")
            print(f"  T₂* = {t2_us} μs")
            
            # Generate ground state samples
            print(f"  Generating {self.readout['n_samples_per_state']} |0⟩ samples...")
            df0 = self.generate_iq_for_state(
                qubit_id, chi_MHz, 0, self.readout['n_samples_per_state']
            )
            
            # Generate excited state samples
            print(f"  Generating {self.readout['n_samples_per_state']} |1⟩ samples...")
            df1 = self.generate_iq_for_state(
                qubit_id, chi_MHz, 1, self.readout['n_samples_per_state']
            )
            
            # Combine
            df_qubit = pd.concat([df0, df1], ignore_index=True)
            
            # Add T1, T2 info
            df_qubit['t1_us'] = t1_us
            df_qubit['t2_us'] = t2_us
            
            # Save to file
            filename = iq_dir / f"{qubit_id.lower()}_iq.csv"
            df_qubit.to_csv(filename, index=False)
            print(f"  ✅ Saved to {filename}")
            
            all_dfs.append(df_qubit)
            
            # Plot IQ distributions
            self.plot_iq_distribution(df_qubit, qubit_id)
        
        # Combine all data
        df_all = pd.concat(all_dfs, ignore_index=True)
        
        return df_all
    
    def plot_iq_distribution(self, df, qubit_id):
        """Plot IQ distributions for ground and excited states"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Separate by state
        df0 = df[df['state'] == 0]
        df1 = df[df['state'] == 1]
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(df0['I'], df0['Q'], alpha=0.3, s=1, label='|0⟩', c='blue')
        ax.scatter(df1['I'], df1['Q'], alpha=0.3, s=1, label='|1⟩', c='red')
        ax.set_xlabel('I (a.u.)')
        ax.set_ylabel('Q (a.u.)')
        ax.set_title(f'{qubit_id} IQ Scatter (χ/2π = {df["chi_MHz"].iloc[0]} MHz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Histogram projection on I axis
        ax = axes[1]
        ax.hist(df0['I'], bins=50, alpha=0.5, label='|0⟩', color='blue', density=True)
        ax.hist(df1['I'], bins=50, alpha=0.5, label='|1⟩', color='red', density=True)
        ax.set_xlabel('I (a.u.)')
        ax.set_ylabel('Probability density')
        ax.set_title('I-projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{qubit_id} Readout Characterization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_file = figures_dir / f'{qubit_id.lower()}_iq_plot.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Plot saved to {plot_file}")
    
    def save_metadata(self, df_all):
        """Save generation metadata"""
        
        metadata = {
            'date': datetime.now().isoformat(),
            'config': {
                'readout': self.readout,
                'noise': self.noise,
                'qubits': self.qubits
            },
            'total_samples': len(df_all),
            'samples_per_qubit': {
                qubit: len(df_all[df_all['qubit_id'] == qubit]) 
                for qubit in self.qubits.keys()
            },
            'files': [str(f.relative_to(iq_dir.parent)) for f in iq_dir.glob('*.csv')]
        }
        
        metadata_file = metadata_dir / 'generation_params.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📊 Metadata saved to {metadata_file}")
        
        return metadata

def main():
    """Main execution for Step 4.1"""
    
    # Create generator
    generator = ReadoutDataGenerator(config)
    
    # Generate all data
    print("\n🚀 Generating readout data...")
    df_all = generator.generate_all_data()
    
    # Save metadata
    metadata = generator.save_metadata(df_all)
    
    # Summary statistics
    print("\n" + "="*60)
    print("📋 GENERATION SUMMARY")
    print("="*60)
    print(f"Total samples generated: {len(df_all):,}")
    print(f"Qubits processed: {list(generator.qubits.keys())}")
    print(f"Samples per qubit: {len(df_all) // len(generator.qubits):,}")
    print(f"Samples per state: {generator.readout['n_samples_per_state']:,}")
    
    # Data integrity check
    print("\n🔍 Data integrity check:")
    for qubit in generator.qubits.keys():
        df_q = df_all[df_all['qubit_id'] == qubit]
        n0 = len(df_q[df_q['state'] == 0])
        n1 = len(df_q[df_q['state'] == 1])
        print(f"  {qubit}: {n0} |0⟩, {n1} |1⟩")
        
        if n0 == n1 == generator.readout['n_samples_per_state']:
            print(f"    ✅ Balanced")
        else:
            print(f"    ❌ Imbalanced!")
    
    print(f"\n📁 Data saved to: {iq_dir}")
    print(f"📁 Metadata saved to: {metadata_dir}")
    print(f"📁 Plots saved to: {figures_dir}")
    
    print("\n" + "="*60)
    print("✅ STEP 4.1 COMPLETE")
    print("="*60)
    print("\nNext: Step 4.2 - Train Readout Classifiers")

if __name__ == "__main__":
    main()
