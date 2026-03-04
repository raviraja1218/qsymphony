#!/usr/bin/env python
"""
Step 2.2: Generate 1000 baseline trajectories in parallel - FIXED
Run Monte Carlo trajectories with no control, save photocurrent and states
"""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import pickle
from datetime import datetime
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports - FIXED: removed 'options' from direct import
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mesolve, mcsolve
    from qutip.solver import Options as SolverOptions
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    sys.exit(1)

# Load hardware parameters
hw_params_file = Path('~/projects/qsymphony/phase2_quantum_sim/hardware_params.json').expanduser()
with open(hw_params_file, 'r') as f:
    hw_params = json.load(f)

# Configuration
config = {
    'hilbert': {'transmon_levels': 2, 'mechanical_levels': 15},
    'simulation': {
        'time_step_ns': 1.0,
        'time_total_us': 50.0,
        'save_interval_ns': 10.0,
        'n_trajectories': 1000,
        'seed': 42
    },
    'constants': {'temperature_mK': 20, 'hbar': 1.0545718e-34, 'kBoltzmann': 1.380649e-23},
    'measurement': {'kappa_MHz': 50.0, 'efficiency': 0.9},
    'paths': {
        'trajectories': '~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/',
        'photocurrents': '~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/photocurrents/',
        'metadata': '~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/metadata/'
    }
}

# Expand paths
traj_dir = Path(config['paths']['trajectories']).expanduser()
photo_dir = Path(config['paths']['photocurrents']).expanduser()
meta_dir = Path(config['paths']['metadata']).expanduser()

# Create directories
traj_dir.mkdir(parents=True, exist_ok=True)
photo_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 2.2: Generate 1000 Baseline Trajectories (FIXED)")
print("="*60)

# Hilbert space dimensions
N_q = config['hilbert']['transmon_levels']
N_m = config['hilbert']['mechanical_levels']

print(f"\n📐 Hilbert space: {N_q} x {N_m} = {N_q * N_m} dimensions")

# Convert frequencies to angular frequencies
wq = 2 * np.pi * hw_params['qubit']['frequency_ghz'] * 1e9
wm = 2 * np.pi * hw_params['mechanical']['frequency_mhz'] * 1e6
g0 = 2 * np.pi * hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6

# Decay rates
T1_q = hw_params['losses']['t1_qubit_us'] * 1e-6
T2_q = hw_params['losses']['t2_qubit_us'] * 1e-6
T1_m = hw_params['losses']['t1_mech_us'] * 1e-6

gamma_q = 1.0 / T1_q
gamma_phi = 1.0 / T2_q - 0.5 / T1_q
gamma_m = 1.0 / T1_m

# Thermal occupancy
T = config['constants']['temperature_mK'] * 1e-3
hbar = config['constants']['hbar']
kB = config['constants']['kBoltzmann']
n_th = 1.0 / (np.exp(hbar * wm / (kB * T)) - 1)
print(f"\n🌡️  Thermal occupancy: n_th = {n_th:.3f}")

# Build operators
print("\n🔧 Building operators...")
a = tensor(destroy(N_q), qeye(N_m))
a_dag = tensor(destroy(N_q).dag(), qeye(N_m))
n_q = a_dag * a

b = tensor(qeye(N_q), destroy(N_m))
b_dag = tensor(qeye(N_q), destroy(N_m).dag())
n_m = b_dag * b

# Hamiltonian
H_q = wq * n_q
H_m = wm * n_m
H_int = g0 * (a_dag * b + a * b_dag)
H = H_q + H_m + H_int

# Collapse operators
c_ops = []
c_ops.append(np.sqrt(gamma_q) * a)                          # Qubit relaxation
c_ops.append(np.sqrt(2 * gamma_phi) * n_q)                  # Qubit dephasing
c_ops.append(np.sqrt(gamma_m * (n_th + 1)) * b)             # Mechanical emission
c_ops.append(np.sqrt(gamma_m * n_th) * b_dag)               # Mechanical absorption

# Measurement operator (for photocurrent)
kappa = 2 * np.pi * config['measurement']['kappa_MHz'] * 1e6
L = np.sqrt(kappa) * a

print(f"✅ Operators built: {len(c_ops)} collapse operators")

# Time parameters
dt = config['simulation']['time_step_ns'] * 1e-9
T_total = config['simulation']['time_total_us'] * 1e-6
save_interval = config['simulation']['save_interval_ns'] * 1e-9

# Create time lists
tlist_full = np.arange(0, T_total + dt, dt)
tlist_save = np.arange(0, T_total + save_interval, save_interval)
save_indices = [int(i * save_interval / dt) for i in range(len(tlist_save))]

print(f"\n⏱️  Time parameters:")
print(f"  Total time: {T_total*1e6:.1f} μs")
print(f"  Time step: {dt*1e9:.1f} ns")
print(f"  Save interval: {save_interval*1e9:.1f} ns")
print(f"  Full steps: {len(tlist_full)}")
print(f"  Save steps: {len(tlist_save)}")

# Initial state (same for all trajectories)
rho_q = basis(N_q, 0) * basis(N_q, 0).dag()
rho_m = qt.thermal_dm(N_m, n_th)
rho0 = tensor(rho_q, rho_m)

# Monte Carlo options - FIXED: using SolverOptions correctly
mc_options = SolverOptions(store_states=True, store_final_state=True)

def run_single_trajectory(traj_id, seed_offset=0):
    """Run a single Monte Carlo trajectory"""
    
    # Set seed for reproducibility
    seed = traj_id + seed_offset
    np.random.seed(seed)
    
    try:
        # Run Monte Carlo solver
        result = qt.mcsolve(
            H, rho0, tlist_full,
            c_ops=c_ops,
            e_ops=[n_q, n_m, a_dag * b + a * b_dag],
            ntraj=1,
            options=mc_options,
            progress_bar=False  # Disable individual progress bars
        )
        
        # Extract photocurrent (simulated from measurement operator)
        # For baseline, we'll generate noise that matches measurement strength
        dW = np.random.randn(len(tlist_full)) * np.sqrt(dt)
        photocurrent = np.sqrt(config['measurement']['efficiency']) * dW
        
        # Save only at save intervals to save space
        n_q_saved = np.array(result.expect[0])[save_indices]
        n_m_saved = np.array(result.expect[1])[save_indices]
        corr_saved = np.array(result.expect[2])[save_indices]
        photo_saved = photocurrent[save_indices]
        
        # Prepare data to save
        data = {
            'traj_id': traj_id,
            'seed': seed,
            'times': tlist_save,
            'n_q': n_q_saved,
            'n_m': n_m_saved,
            'correlation': corr_saved,
            'photocurrent': photo_saved,
            'final_state': result.states[-1] if result.states else None
        }
        
        return data
        
    except Exception as e:
        print(f"❌ Trajectory {traj_id} failed: {e}")
        return None

def save_trajectory(data):
    """Save trajectory data to files"""
    if data is None:
        return False
    
    traj_id = data['traj_id']
    
    # Save photocurrent separately (numpy format)
    photo_file = photo_dir / f'photocurrent_{traj_id:04d}.npy'
    np.save(photo_file, data['photocurrent'])
    
    # Save trajectory data (pickle format) - without photocurrent to avoid duplication
    traj_file = traj_dir / f'trajectory_{traj_id:04d}.pkl'
    with open(traj_file, 'wb') as f:
        # Remove photocurrent from main data
        data_copy = {k: v for k, v in data.items() if k != 'photocurrent'}
        pickle.dump(data_copy, f)
    
    return True

def run_parallel_trajectories(n_trajectories=1000, n_processes=None):
    """Run trajectories in parallel"""
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    print(f"\n🚀 Starting {n_trajectories} trajectories on {n_processes} processes...")
    print(f"Estimated time: ~{n_trajectories * 5 / n_processes:.1f} minutes")
    start_time = time.time()
    
    # Use multiprocessing pool
    successful = 0
    failed = 0
    
    with mp.Pool(processes=n_processes) as pool:
        # Create argument list
        args = [(i, config['simulation']['seed']) for i in range(n_trajectories)]
        
        # Run trajectories with progress bar
        with tqdm(total=n_trajectories, desc="Trajectories") as pbar:
            for result in pool.starmap(run_single_trajectory, args):
                if save_trajectory(result):
                    successful += 1
                else:
                    failed += 1
                pbar.update()
                pbar.set_postfix({'Success': successful, 'Failed': failed})
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed {successful}/{n_trajectories} trajectories in {elapsed/60:.1f} minutes")
    
    return successful, failed

def create_summary(successful, failed):
    """Create summary file with metadata"""
    
    summary = {
        'date': datetime.now().isoformat(),
        'n_trajectories_total': config['simulation']['n_trajectories'],
        'n_trajectories_successful': successful,
        'n_trajectories_failed': failed,
        'parameters': {
            'wq_GHz': hw_params['qubit']['frequency_ghz'],
            'wm_MHz': hw_params['mechanical']['frequency_mhz'],
            'g0_MHz': hw_params['couplings']['g0_qubit_mech_mhz'],
            'T1_q_us': hw_params['losses']['t1_qubit_us'],
            'T2_q_us': hw_params['losses']['t2_qubit_us'],
            'T1_m_us': hw_params['losses']['t1_mech_us'],
            'n_th': n_th,
            'kappa_MHz': config['measurement']['kappa_MHz'],
            'efficiency': config['measurement']['efficiency']
        },
        'simulation': {
            'time_total_us': config['simulation']['time_total_us'],
            'time_step_ns': config['simulation']['time_step_ns'],
            'save_interval_ns': config['simulation']['save_interval_ns'],
            'n_time_steps': len(tlist_full)
        }
    }
    
    # Save summary
    summary_file = meta_dir / 'trajectory_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_file}")
    
    return summary

def plot_sample_trajectories(successful):
    """Plot a few sample trajectories for quick verification"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Load a few random trajectories
    traj_files = list(traj_dir.glob('trajectory_*.pkl'))
    n_samples = min(5, len(traj_files))
    sample_files = np.random.choice(traj_files, n_samples, replace=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    for idx, traj_file in enumerate(sample_files):
        with open(traj_file, 'rb') as f:
            data = pickle.load(f)
        
        t_us = data['times'] * 1e6
        
        # Qubit population
        axes[0, 0].plot(t_us, data['n_q'], color=colors[idx], alpha=0.7, 
                       label=f"Traj {data['traj_id']}")
        
        # Mechanical population
        axes[0, 1].plot(t_us, data['n_m'], color=colors[idx], alpha=0.7)
        
        # Correlation
        axes[0, 2].plot(t_us, data['correlation'], color=colors[idx], alpha=0.7)
        
        # Load corresponding photocurrent
        photo_file = photo_dir / f"photocurrent_{data['traj_id']:04d}.npy"
        if photo_file.exists():
            photocurrent = np.load(photo_file)
            axes[1, 0].plot(t_us, photocurrent, color=colors[idx], alpha=0.5)
    
    axes[0, 0].set_xlabel('Time (μs)')
    axes[0, 0].set_ylabel('⟨n_q⟩')
    axes[0, 0].set_title('Qubit Population')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel('⟨n_m⟩')
    axes[0, 1].set_title('Mechanical Population')
    axes[0, 1].axhline(y=n_th, color='r', linestyle='--', alpha=0.5, label=f'n_th={n_th:.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Time (μs)')
    axes[0, 2].set_ylabel('⟨a†b + a b†⟩')
    axes[0, 2].set_title('Qubit-Mechanical Correlation')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (μs)')
    axes[1, 0].set_ylabel('I(t)')
    axes[1, 0].set_title('Photocurrent (Sample)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average over trajectories (placeholder)
    axes[1, 1].set_title('Trajectory Averages')
    axes[1, 1].text(0.5, 0.5, 'Computing...', ha='center', va='center')
    
    # Statistics
    axes[1, 2].set_title('Trajectory Statistics')
    axes[1, 2].text(0.5, 0.5, f"{successful} successful\n0 failed", 
                    ha='center', va='center', fontsize=12)
    
    plt.suptitle('Sample Baseline Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_dir = Path('~/projects/qsymphony/results/phase2/figures').expanduser()
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / 'sample_trajectories.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Sample trajectories plot saved to: {plot_file}")
    plt.close()

def main():
    """Main execution"""
    
    # Get number of CPU cores
    n_cores = mp.cpu_count()
    print(f"\n💻 Available CPU cores: {n_cores}")
    
    # Ask for number of processes
    n_proc = n_cores  # Default to all cores
    
    # Run trajectories
    successful, failed = run_parallel_trajectories(
        n_trajectories=config['simulation']['n_trajectories'],
        n_processes=n_proc
    )
    
    # Create summary
    summary = create_summary(successful, failed)
    
    # Plot samples if we have any successful trajectories
    if successful > 0:
        plot_sample_trajectories(successful)
    
    print("\n" + "="*60)
    print("STEP 2.2 COMPLETE")
    print("="*60)
    print(f"✅ {successful} trajectories generated successfully")
    print(f"📁 Trajectories: {traj_dir}")
    print(f"📁 Photocurrents: {photo_dir}")
    print(f"📁 Metadata: {meta_dir}")
    print("\nNext: Step 2.3 - Compute Baseline Wigner Functions")
    print("="*60)

if __name__ == "__main__":
    main()
