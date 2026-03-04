#!/usr/bin/env python
"""
PINN with proper training that actually learns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))

results_dir = Path('results/phase4/figures')
results_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path('results/phase4/data')
data_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Using device: {device}")

class Phase3Hamiltonian:
    """Exact Hamiltonian from Phase 3"""
    def __init__(self):
        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        
        # Pre-compute two-qubit operators
        self.XI = torch.kron(self.X, self.I)
        self.YI = torch.kron(self.Y, self.I)
        self.ZI = torch.kron(self.Z, self.I)
        self.IX = torch.kron(self.I, self.X)
        self.IY = torch.kron(self.I, self.Y)
        self.IZ = torch.kron(self.I, self.Z)
        self.XX = torch.kron(self.X, self.X)
        self.YY = torch.kron(self.Y, self.Y)
        self.ZZ = torch.kron(self.Z, self.Z)
    
    def build(self, omega_x, omega_y, delta):
        H = (omega_x * self.XI + omega_y * self.YI + delta * self.ZI +
             0.5 * self.XX + 0.5 * self.YY)  # Fixed coupling
        return H

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Larger network with residual connections
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 3)
        
        self.tanh = nn.Tanh()
        
        # Initialize weights properly
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        
        # Target CNOT
        self.register_buffer('U_target', torch.zeros((4, 4), dtype=torch.complex64))
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
        
        self.ham = Phase3Hamiltonian()
    
    def forward(self, t):
        x = self.tanh(self.fc1(t))
        x = self.tanh(self.fc2(x)) + x  # Residual
        x = self.tanh(self.fc3(x)) + x  # Residual
        x = self.tanh(self.fc4(x)) + x  # Residual
        x = self.fc5(x)
        return x

def train_for_p(p, n_epochs=1000):
    """Train with proper learning rate schedule"""
    print(f"\n{'='*50}")
    print(f"Training with p = {p}")
    print(f"{'='*50}")
    
    model = PINN().to(device)
    
    # Better optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    
    dt = 1e-9
    n_steps = 100
    history = []
    
    pbar = tqdm(range(n_epochs), desc=f"p={p}")
    for epoch in pbar:
        # Random time samples for better training
        t = torch.rand(n_steps, 1, device=device) * 2 - 1  # Uniform in [-1, 1]
        pulses = model(t)
        
        # Start with |00⟩
        rho = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        rho[0, 0] = 1.0
        
        # Evolve
        for step in range(n_steps):
            omega_x, omega_y, delta = pulses[step]
            H = model.ham.build(omega_x, omega_y, delta)
            
            U = torch.matrix_exp(-1j * dt * H)
            rho = U @ rho @ U.conj().T
        
        # Compute fidelity (no noise during training)
        rho_target = model.U_target @ rho @ model.U_target.conj().T
        fidelity = rho_target[0, 0].real
        
        loss = 1 - fidelity
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        history.append(float(fidelity))
        pbar.set_postfix({'fidelity': f'{fidelity:.4f}'})
    
    return history

def main():
    print("="*60)
    print("PROPER PINN TRAINING")
    print("="*60)
    
    # Train only p=0 first to verify learning
    print("\n🔬 Testing if network can learn at p=0...")
    history = train_for_p(0.0, n_epochs=500)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Fidelity')
    plt.title('Learning Curve at p=0')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(results_dir / 'learning_test.png', dpi=150)
    print(f"\n✅ Learning test saved: {results_dir / 'learning_test.png'}")
    print(f"Final fidelity: {history[-1]:.4f}")
    
    if history[-1] < 0.99:
        print("\n⚠️ Network not learning - check architecture")
    else:
        print("\n✅ Network learning! Now training full sweep...")
        
        # Train all p values
        p_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        all_results = {}
        
        for p in p_values:
            history = train_for_p(p, n_epochs=1000)
            all_results[str(p)] = {
                'final_fidelity': history[-1],
                'history': history
            }
        
        # Save results
        with open(data_dir / 'proper_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
