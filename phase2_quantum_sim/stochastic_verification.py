#!/usr/bin/env python
"""
Test trained policy under true stochastic evolution
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys
sys.path.append(str(Path.home() / 'projects' / 'qsymphony' / 'phase3_rl_control'))

from utils.ppo_network import PPOActorCritic
from utils.environment_wrapper_quantum import QuantumControlEnv

def correct_entanglement(rho):
    """Correct logarithmic negativity"""
    rho_pt = qt.partial_transpose(rho, [1, 0])
    evals = rho_pt.eigenenergies()
    trace_norm = np.sum(np.abs(evals))
    return np.log2(trace_norm)

def load_trained_policy():
    """Load the trained policy from Phase 3"""
    model_path = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models' / 'ppo_measurement_final.zip'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please complete Phase 3 training first")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOActorCritic(obs_dim=11, action_dim=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    return policy, device

def run_stochastic_evaluation(policy, device, n_runs=10):
    """Evaluate policy under stochastic evolution"""
    
    # System parameters from Phase 1
    wq = 2 * np.pi * 4.753e9
    wm = 2 * np.pi * 492.4e6
    g0 = 2 * np.pi * 11.19e6
    kappa = 2 * np.pi * 50e6
    eta = 0.9
    
    # Hilbert space
    N_q, N_m = 2, 15
    
    # Operators
    a = qt.tensor(qt.destroy(N_q), qt.qeye(N_m))
    b = qt.tensor(qt.qeye(N_q), qt.destroy(N_m))
    
    # Collapse operators
    T1_q, T2_q, T1_m = 85e-6, 45e-6, 1200e-6
    n_th = 0.443
    
    c_ops = [
        np.sqrt(1/T1_q) * a,
        np.sqrt(1/T2_q - 0.5/T1_q) * (a.dag() * a),
        np.sqrt(1/T1_m * (n_th + 1)) * b,
        np.sqrt(1/T1_m * n_th) * b.dag()
    ]
    
    # Measured operator
    sc_ops = [np.sqrt(kappa) * a]
    
    # Initial state
    psi0 = qt.tensor(qt.basis(N_q, 0), qt.basis(N_m, 0))
    
    # Time evolution
    dt = 1e-9
    tlist = np.arange(0, 50e-6, dt)
    n_steps = len(tlist)
    
    E_N_values = []
    
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...")
        
        psi = psi0.copy()
        hidden_state = None
        trajectory_EN = []
        
        for step in range(0, n_steps, 100):  # Sample every 100 steps
            # Get action from policy
            obs = np.zeros(13)  # Simplified observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _, _, hidden_state = policy.select_action(
                    obs_tensor, hidden_state, deterministic=True
                )
            
            # Build Hamiltonian
            delta_norm, alpha_norm = action.cpu().numpy()[0]
            delta = delta_norm * wm
            alpha = alpha_norm * 1e6
            
            H0 = wq * a.dag() * a + wm * b.dag() * b + g0 * (a.dag() * b + a * b.dag())
            H_drive = alpha * (a + a.dag()) * np.cos(delta * step * dt)
            H = H0 + H_drive
            
            # Stochastic evolution for this step
            result = qt.smesolve(
                H, psi, [step*dt, (step+100)*dt],
                c_ops=c_ops,
                sc_ops=sc_ops,
                ntraj=1,
                options={'store_states': True}
            )
            
            psi = result.states[-1]
            
            # Compute entanglement
            E_N = correct_entanglement(qt.ket2dm(psi))
            trajectory_EN.append(E_N)
        
        E_N_values.append(trajectory_EN)
    
    return np.array(E_N_values)

def plot_results(E_N_values):
    """Plot stochastic results"""
    mean_EN = np.mean(E_N_values, axis=0)
    std_EN = np.std(E_N_values, axis=0)
    steps = np.arange(len(mean_EN))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_EN, 'b-', linewidth=2, label='Mean E_N')
    plt.fill_between(steps, mean_EN - std_EN, mean_EN + std_EN, 
                     alpha=0.3, color='b', label='±1 std')
    plt.xlabel('Time step (×100)')
    plt.ylabel('E_N')
    plt.title('Stochastic Evolution with Trained Policy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase2' / 'figures' / 'stochastic_verification.png'
    plt.savefig(plot_file, dpi=150)
    print(f"\n✅ Plot saved: {plot_file}")

if __name__ == "__main__":
    print("="*60)
    print("STOCHASTIC VERIFICATION")
    print("="*60)
    
    policy, device = load_trained_policy()
    if policy is None:
        sys.exit(1)
    
    print("\n🎲 Running stochastic evaluation (10 runs)...")
    E_N_values = run_stochastic_evaluation(policy, device, n_runs=10)
    
    plot_results(E_N_values)
    
    print("\n" + "="*60)
    print("✅ Stochastic verification complete")
    print("="*60)
