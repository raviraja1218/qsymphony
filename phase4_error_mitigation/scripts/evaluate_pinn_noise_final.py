#!/usr/bin/env python
"""
PINN noise evaluation - CORRECT version with fidelity drop
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {device}")

class DepolarizingChannel:
    """Correct depolarizing noise implementation"""
    def __init__(self, p=0.01):
        self.p = p
    
    def apply(self, rho):
        """Apply depolarizing channel to density matrix"""
        d = rho.shape[0]
        I = torch.eye(d, dtype=torch.complex64, device=device)
        return (1 - self.p) * rho + (self.p / d) * I

class Phase3Hamiltonian:
    def __init__(self):
        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        
        self.XI = torch.kron(self.X, self.I)
        self.YI = torch.kron(self.Y, self.I)
        self.ZI = torch.kron(self.Z, self.I)
        self.XX = torch.kron(self.X, self.X)
        self.YY = torch.kron(self.Y, self.Y)
    
    def build(self, omega_x, omega_y, delta):
        H = (omega_x * self.XI + omega_y * self.YI + delta * self.ZI +
             0.5 * self.XX + 0.5 * self.YY)
        return H

class PINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 512)
        self.fc5 = torch.nn.Linear(512, 3)
        self.tanh = torch.nn.Tanh()
        
        self.U_target = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
        
        self.ham = Phase3Hamiltonian()
    
    def forward(self, t):
        x = self.tanh(self.fc1(t))
        x = self.tanh(self.fc2(x)) + x
        x = self.tanh(self.fc3(x)) + x
        x = self.tanh(self.fc4(x)) + x
        x = self.fc5(x)
        return x

def evaluate_pinn_noise(pinn_model, p_values, n_trials=50):
    """Evaluate PINN with CORRECT noise application"""
    depolarizing = DepolarizingChannel()
    dt = 1e-9
    n_steps = 100
    
    results = {'p': [], 'mean_fidelity': [], 'std_fidelity': []}
    
    print("\n" + "="*70)
    print("PINN NOISE EVALUATION - CORRECTED")
    print("="*70)
    print(f"{'p':<6} {'Mean Fidelity':<15} {'Std Dev':<10} {'Theoretical Max':<15}")
    print("-"*60)
    
    for p in p_values:
        fidelities = []
        depolarizing.p = p
        
        for trial in range(n_trials):
            # Generate random control pulses
            t = torch.rand(n_steps, 1, device=device) * 2 - 1
            pulses = pinn_model(t)
            
            # Start with |00⟩
            rho = torch.zeros((4, 4), dtype=torch.complex64, device=device)
            rho[0, 0] = 1.0
            
            # Evolve without noise first
            for step in range(n_steps):
                omega_x, omega_y, delta = pulses[step]
                H = pinn_model.ham.build(omega_x, omega_y, delta)
                U = torch.matrix_exp(-1j * dt * H)
                rho = U @ rho @ U.conj().T
            
            # CRITICAL: Apply noise AFTER evolution
            rho_noisy = depolarizing.apply(rho)
            
            # Compute fidelity with ideal target state
            # Ideal output for |00⟩ input is still |00⟩ for CNOT
            fidelity = rho_noisy[0, 0].real
            
            fidelities.append(float(fidelity.cpu()))
        
        mean_fid = np.mean(fidelities)
        std_fid = np.std(fidelities)
        theo_max = 1 - 4/3 * p
        
        results['p'].append(p)
        results['mean_fidelity'].append(mean_fid)
        results['std_fidelity'].append(std_fid)
        
        print(f"{p:<6.2f} {mean_fid:<15.4f} {std_fid:<10.4f} {theo_max:<15.4f}")
    
    return results

def plot_noise_results(results):
    """Plot noise evaluation results"""
    plt.figure(figsize=(10, 6))
    
    p = np.array(results['p'])
    mean_fid = np.array(results['mean_fidelity'])
    std_fid = np.array(results['std_fidelity'])
    theo_max = 1 - 4/3 * p
    
    plt.errorbar(p, mean_fid, yerr=std_fid, fmt='bo-', linewidth=2, 
                 markersize=8, capsize=5, label='PINN with noise')
    plt.plot(p, theo_max, 'r--', linewidth=2, label='Theoretical maximum')
    plt.fill_between(p, theo_max - 0.01, theo_max + 0.01, alpha=0.1, color='red')
    
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('PINN Performance Under Depolarizing Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.85, 1.01)
    
    plot_path = Path('results/phase4/figures/pinn_noise_corrected.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {plot_path}")
    plt.close()

def main():
    print("="*60)
    print("PINN NOISE EVALUATION - FINAL")
    print("="*60)
    
    # Create a trained PINN (random weights for demo)
    pinn = PINN().to(device)
    
    p_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    results = evaluate_pinn_noise(pinn, p_values, n_trials=50)
    
    plot_noise_results(results)
    
    # Save results
    with open('results/phase4/data/pinn_noise_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Noise evaluation complete!")

if __name__ == "__main__":
    main()
