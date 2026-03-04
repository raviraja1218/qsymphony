#!/usr/bin/env python
"""
PINN with CORRECT two-qubit Hamiltonian from Phase 3
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
        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        self._build_kraus()
    
    def _build_kraus(self):
        """Build all 16 two-qubit Kraus operators"""
        paulis1 = [self.I, self.X, self.Y, self.Z]
        paulis2 = [self.I, self.X, self.Y, self.Z]
        
        self.kraus = []
        for i, p1 in enumerate(paulis1):
            for j, p2 in enumerate(paulis2):
                if i == 0 and j == 0:
                    K = np.sqrt(1 - self.p) * torch.kron(p1, p2)
                else:
                    K = np.sqrt(self.p / 15) * torch.kron(p1, p2)
                self.kraus.append(K)
    
    def apply(self, rho):
        rho_noisy = torch.zeros_like(rho)
        for K in self.kraus:
            rho_noisy += K @ rho @ K.conj().T
        return rho_noisy

class ImprovedPINN(nn.Module):
    """PINN with correct two-qubit Hamiltonian"""
    
    def __init__(self):
        super().__init__()
        # Deeper network
        self.net = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 3)
        ).to(device)
        
        # Target CNOT
        self.U_target = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
        
    def forward(self, t):
        return self.net(t)
    
    def build_hamiltonian(self, pulse):
        """Build correct two-qubit Hamiltonian"""
        omega_x, omega_y, delta = pulse
        
        # Pauli matrices
        I = torch.eye(2, dtype=torch.complex64, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        
        # Two-qubit interactions from Phase 3
        H = (omega_x * torch.kron(X, I) + 
             omega_y * torch.kron(Y, I) + 
             delta * torch.kron(Z, I) +
             0.5 * torch.kron(X, X) +   # Add entanglement terms
             0.5 * torch.kron(Y, Y) +
             0.5 * torch.kron(Z, Z))
        
        return H

def train_pinn(p_depolarizing=0.01):
    """Train PINN with improved Hamiltonian"""
    
    model = ImprovedPINN()
    depolarizing = DepolarizingChannel(p=p_depolarizing)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    dt = 1e-9
    n_steps = 200
    fidelity_history = []
    
    print(f"\n📊 Training with p = {p_depolarizing}...")
    
    for epoch in range(1000):
        # Generate pulses
        t = torch.linspace(0, 1, n_steps).reshape(-1, 1).to(device)
        pulses = model(t)
        
        # Start with |00⟩
        rho = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        rho[0, 0] = 1.0
        
        # Evolve with correct Hamiltonian
        for step in range(n_steps):
            H = model.build_hamiltonian(pulses[step])
            
            # Unitary evolution
            U = torch.matrix_exp(-1j * dt * H)
            rho = U @ rho @ U.conj().T
            
            # Apply noise
            rho = depolarizing.apply(rho)
        
        # Compute fidelity
        rho_target = model.U_target @ rho @ model.U_target.conj().T
        fidelity = rho_target[0, 0].real
        
        loss = 1 - fidelity
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        fidelity_history.append(fidelity)
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Fidelity = {fidelity:.4f}")
    
    return fidelity_history

def noise_sweep():
    """Test fidelity for different noise levels"""
    print("\n" + "="*60)
    print("IMPROVED PINN - Fidelity vs Depolarizing Rate")
    print("="*60)
    
    p_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    final_fidelities = []
    
    for p in p_values:
        print(f"\n🔬 Testing p = {p}")
        history = train_pinn(p)
        final_fidelities.append(history[-1])
    
    # Convert to numpy
    p_np = np.array(p_values)
    f_np = np.array([f.cpu().item() if torch.is_tensor(f) else f for f in final_fidelities])
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(p_np, f_np, 'bo-', linewidth=2, markersize=8, label='Improved PINN')
    plt.plot(p_np, 1 - 4/3 * p_np, 'r--', label='Theoretical max')
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('Improved PINN Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare with previous
    plt.subplot(1, 2, 2)
    previous = [1.0, 0.5066, 0.3368, 0.2790, 0.2596, 0.2531]
    plt.plot(p_np, previous, 'gs-', label='Previous PINN')
    plt.plot(p_np, f_np, 'bo-', label='Improved PINN')
    plt.plot(p_np, 1 - 4/3 * p_np, 'r--', label='Theoretical max')
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('Improvement Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'fidelity_improved.png', dpi=150)
    print(f"\n✅ Plot saved: {results_dir / 'fidelity_improved.png'}")
    
    # Save data
    import json
    with open(data_dir / 'improved_results.json', 'w') as f:
        json.dump({
            'p_values': p_np.tolist(),
            'fidelities': f_np.tolist(),
            'previous': previous,
            'theoretical': (1 - 4/3 * p_np).tolist()
        }, f, indent=2)
    
    return p_np, f_np

if __name__ == "__main__":
    noise_sweep()
