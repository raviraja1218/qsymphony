#!/usr/bin/env python
"""
Step 2.2: Generate 1000 baseline trajectories - SEQUENTIAL but faster
Run Monte Carlo trajectories with no control, save photocurrent and states
Using sequential runs to avoid multiprocessing pickling issues
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
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mcsolve
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
print("STEP 2.2: Generate 1000 Baseline Trajectories (SEQUENTIAL)")
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

# Initial state vector
psi_q = basis(N_q, 0)
psi_m = basis(N_m, 0)  # Start mechanical mode in ground state
psi0 = tensor(psi_q, psi_m)
print(f"\n🎯 Initial state: |0,0⟩")

# Monte Carlo options
mc_options = SolverOptions(store_states=True, store_final_state=True)

def run_single_trajectory(traj_id):
    """Run a single Monte Carlo trajectory"""
    
    # Set seed for reproducibility
    seed = traj_id + config['simulation']['seed']
    np.random.seed(seed)
    
    try:
        # Run Monte Carlo solver
        result = qt.mcsolve(
            H, psi0, tlist_full,
            c_ops=c_ops,
            e_ops=[n_q, n_m, a_dag * b + a * b_dag],
            ntraj=1,
            options=mc_options,
            progress_bar=False
        )
        
        # Generate photocurrent (simulated)
        dW = np.random.randn(len(tlist_full)) * np.sqrt(dt)
        photocurrent = np.sqrt(config['measurement']['efficiency']) * dW
        
        # Save only at save intervals
        data = {
            'traj_id': traj_id,
            'seed': seed,
            'times': tlist_save,
            'n_q': np.array(result.expect[0])[save_indices],
            'n_m': np.array(result.expect[1])[save_indices],
            'correlation': np.array(result.expect[2])[save_indices],
            'photocurrent': photocurrent[save_indices],
        }
        
        # Save files
        traj_file = traj_dir / f'trajectory_{traj_id:04d}.pkl'
        with open(traj_file, 'wb') as f:
            # Don't save photocurrent in main file
            data_copy = {k: v for k, v in data.items() if k != 'photocurrent'}
            pickle.dump(data_copy, f)
        
        photo_file = photo_dir / f'photocurrent_{traj_id:04d}.npy'
        np.save(photo_file, data['photocurrent'])
        
        return True
        
    except Exception as e:
        print(f"\n❌ Trajectory {traj_id} failed: {e}")
        return False

def main():
    """Main execution"""
    
    n_trajectories = config['simulation']['n_trajectories']
    print(f"\n🚀 Starting {n_trajectories} trajectories sequentially...")
    print(f"Estimated time: ~{n_trajectories * 5 / 60:.1f} hours")  # ~5 min per trajectory
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Run trajectories with progress bar
    for i in tqdm(range(n_trajectories), desc="Trajectories"):
        if run_single_trajectory(i):
            successful += 1
        else:
            failed += 1
        
        # Update progress bar description
        tqdm.write(f"Progress: {successful}/{i+1} successful")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Completed {successful}/{n_trajectories} trajectories in {elapsed/60:.1f} minutes")
    
    # Save summary
    summary = {
        'date': datetime.now().isoformat(),
        'n_trajectories_total': n_trajectories,
        'n_trajectories_successful': successful,
        'n_trajectories_failed': failed,
        'elapsed_minutes': elapsed/60,
        'parameters': {
            'wq_GHz': hw_params['qubit']['frequency_ghz'],
            'wm_MHz': hw_params['mechanical']['frequency_mhz'],
            'g0_MHz': hw_params['couplings']['g0_qubit_mech_mhz'],
            'n_th': n_th,
        }
    }
    
    summary_file = meta_dir / 'trajectory_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_file}")
    print("\n" + "="*60)
    print("STEP 2.2 COMPLETE")
    print("="*60)
    print(f"✅ {successful} trajectories generated")
    print(f"📁 Trajectories: {traj_dir}")
    print(f"📁 Photocurrents: {photo_dir}")
    print("\nNext: Step 2.3 - Compute Baseline Wigner Functions")
    print("="*60)

if __name__ == "__main__":
    main()
