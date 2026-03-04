#!/usr/bin/env python
"""
PINN with CORRECT depolarizing noise implementation - FIXED plotting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Create results directory
results_dir = Path('results/phase4/figures')
results_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path('results/phase4/data')
data_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {device}")

class DepolarizingChannel:
    """Correct depolarizing noise implementation"""
    
    def __init__(self, p=0.01):
        self.p = p
        # Pauli matrices - move to device
        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        
        # Build Kraus operators
        self._build_kraus()
    
    def _build_kraus(self):
        """Build all 16 two-qubit Kraus operators"""
        paulis1 = [self.I, self.X, self.Y, self.Z]
        paulis2 = [self.I, self.X, self.Y, self.Z]
        
        self.kraus = []
        for i, p1 in enumerate(paulis1):
            for j, p2 in enumerate(paulis2):
                if i == 0 and j == 0:
                    # Identity gets (1-p) weight
                    K = np.sqrt(1 - self.p) * torch.kron(p1, p2)
                else:
                    # Non-identity get p/15 weight
                    K = np.sqrt(self.p / 15) * torch.kron(p1, p2)
                self.kraus.append(K)
    
    def apply(self, rho):
        """Apply depolarizing channel to density matrix"""
        rho_noisy = torch.zeros_like(rho)
        for K in self.kraus:
            rho_noisy += K @ rho @ K.conj().T
        return rho_noisy

class PINN(nn.Module):
    """Physics-Informed Neural Network for gate optimization"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3)  # [Ω_x, Ω_y, Δ]
        ).to(device)
        
        # Target CNOT matrix
        self.U_target = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
        
    def forward(self, t):
        return self.net(t)
    
    def get_pulses(self, n_points=1000):
        t = torch.linspace(0, 1, n_points).reshape(-1, 1).to(device)
        return self.forward(t).cpu().detach().numpy()

def train_pinn_with_noise(p_depolarizing=0.01):
    """Train PINN with specified noise level"""
    
    model = PINN().to(device)
    depolarizing = DepolarizingChannel(p=p_depolarizing)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dt = 1e-9  # 1 ns
    n_steps = 100
    
    print(f"\n📊 Training with p = {p_depolarizing}...")
    
    fidelity_history = []
    
    for epoch in range(500):
        # Generate pulses
        t = torch.linspace(0, 1, n_steps).reshape(-1, 1).to(device)
        pulses = model(t)
        
        # Start with |00⟩ state
        rho = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        rho[0, 0] = 1.0
        
        # Evolve with noise
        for step in range(n_steps):
            # Simple Hamiltonian (placeholder)
            H = torch.zeros((4, 4), dtype=torch.complex64, device=device)
            H[0, 0] = pulses[step, 2]
            H[1, 1] = pulses[step, 2]
            H[2, 2] = -pulses[step, 2]
            H[3, 3] = -pulses[step, 2]
            
            # Unitary evolution
            U = torch.matrix_exp(-1j * dt * H)
            rho = U @ rho @ U.conj().T
            
            # Apply depolarizing noise
            rho = depolarizing.apply(rho)
        
        # Compute fidelity
        rho_target = model.U_target @ rho @ model.U_target.conj().T
        fidelity = rho_target[0, 0].real
        
        loss = 1 - fidelity
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        fidelity_history.append(fidelity)
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Fidelity = {fidelity:.4f}")
    
    return fidelity_history

def noise_sweep():
    """Test fidelity for different noise levels"""
    print("\n" + "="*60)
    print("NOISE SWEEP - Fidelity vs Depolarizing Rate")
    print("="*60)
    
    p_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    final_fidelities = []
    
    for p in p_values:
        print(f"\n🔬 Testing p = {p}")
        history = train_pinn_with_noise(p)
        final_fidelities.append(history[-1])
    
    # Convert to numpy for plotting (FIXED)
    p_values_np = np.array(p_values)
    final_fidelities_np = np.array([f.cpu().item() if torch.is_tensor(f) else f for f in final_fidelities])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(p_values_np, final_fidelities_np, 'bo-', linewidth=2, markersize=8, label='PINN')
    plt.plot(p_values_np, 1 - 4/3 * p_values_np, 'r--', label='Theoretical maximum')
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('Fidelity vs Noise Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plot_path = results_dir / 'fidelity_vs_noise.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\n✅ Plot saved: {plot_path}")
    
    # Save data
    data = {
        'p_values': p_values_np.tolist(),
        'fidelities': final_fidelities_np.tolist(),
        'theoretical_max': (1 - 4/3 * np.array(p_values)).tolist()
    }
    
    import json
    with open(data_dir / 'noise_sweep_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return p_values_np, final_fidelities_np

if __name__ == "__main__":
    noise_sweep()
