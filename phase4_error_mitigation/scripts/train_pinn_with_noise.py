#!/usr/bin/env python
"""
PINN with CORRECT noise implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

class DepolarizingChannel:
    """Correct depolarizing noise implementation"""
    
    def __init__(self, p=0.01):
        self.p = p
        # Pauli matrices
        self.I = torch.eye(2, dtype=torch.complex64)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        
        # Two-qubit Kraus operators
        self._build_kraus_ops()
    
    def _build_kraus_ops(self):
        """Build all 16 two-qubit Kraus operators"""
        paulis1 = [self.I, self.X, self.Y, self.Z]
        paulis2 = [self.I, self.X, self.Y, self.Z]
        
        self.kraus_ops = []
        for i, p1 in enumerate(paulis1):
            for j, p2 in enumerate(paulis2):
                if i == 0 and j == 0:
                    # Identity operator gets (1-p) weight
                    K = np.sqrt(1 - self.p) * torch.kron(p1, p2)
                else:
                    # Non-identity get p/15 weight
                    K = np.sqrt(self.p / 15) * torch.kron(p1, p2)
                self.kraus_ops.append(K)
    
    def apply(self, rho):
        """Apply depolarizing channel to density matrix"""
        rho_noisy = torch.zeros_like(rho)
        for K in self.kraus_ops:
            rho_noisy += K @ rho @ K.conj().T
        return rho_noisy

class PINNWithNoise(nn.Module):
    """PINN with correct noise during training"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3)
        )
        
        self.depolarizing = DepolarizingChannel(p=0.01)
        self.U_target = self._get_cnot()
    
    def _get_cnot(self):
        U = torch.zeros((4, 4), dtype=torch.complex64)
        U[0, 0] = 1
        U[1, 1] = 1
        U[2, 3] = 1
        U[3, 2] = 1
        return U
    
    def forward(self, t):
        return self.net(t)
    
    def compute_fidelity_with_noise(self, pulses, dt):
        """Compute fidelity including depolarizing noise"""
        n_steps = pulses.shape[0]
        
        # Start with |00⟩ state
        rho = torch.zeros((4, 4), dtype=torch.complex64)
        rho[0, 0] = 1.0
        
        # Evolve with noise at each step
        for step in range(n_steps):
            # Build Hamiltonian
            H = self._build_hamiltonian(pulses[step])
            
            # Unitary evolution
            U = torch.matrix_exp(-1j * dt * H)
            rho = U @ rho @ U.conj().T
            
            # Apply depolarizing noise
            rho = self.depolarizing.apply(rho)
        
        # Compute fidelity with ideal CNOT
        rho_ideal = self.U_target @ rho @ self.U_target.conj().T
        fidelity = rho_ideal[0, 0].real
        
        return fidelity
    
    def _build_hamiltonian(self, pulse):
        """Build two-qubit Hamiltonian from control pulse"""
        omega_x, omega_y, delta = pulse
        
        # Pauli matrices
        I = torch.eye(2, dtype=torch.complex64)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        
        # Build Hamiltonian
        H = (omega_x * torch.kron(X, I) + 
             omega_y * torch.kron(Y, I) + 
             0.5 * delta * torch.kron(Z, I) +
             omega_x * torch.kron(I, X) +
             omega_y * torch.kron(I, Y) +
             0.5 * delta * torch.kron(I, Z))
        
        return H

def train_with_noise_check():
    """Train and verify noise dependence"""
    model = PINNWithNoise()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    p_values = np.linspace(0, 0.05, 11)
    final_fidelities = []
    
    for p in p_values:
        model.depolarizing = DepolarizingChannel(p=p)
        
        # Train for this p
        for epoch in range(100):
            t = torch.rand(1000, 1) * 2 - 1  # [-1, 1]
            pulses = model(t)
            
            fidelity = model.compute_fidelity_with_noise(pulses, 1e-9)
            loss = 1 - fidelity
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_fidelities.append(fidelity.item())
    
    # Plot fidelity vs p
    plt.figure()
    plt.plot(p_values, final_fidelities, 'bo-')
    plt.plot(p_values, 1 - 4/3 * np.array(p_values), 'r--', label='Theoretical max')
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('Fidelity decreases with noise')
    plt.legend()
    plt.grid(True)
    plt.savefig('fidelity_vs_noise.png')
    
    return final_fidelities

if __name__ == "__main__":
    train_with_noise_check()
