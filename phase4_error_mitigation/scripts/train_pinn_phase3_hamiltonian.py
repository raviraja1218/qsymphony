#!/usr/bin/env python
"""
PINN using the EXACT Hamiltonian from Phase 3 that achieved E_N=0.914
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
    """Exact Hamiltonian from Phase 3 that produced E_N=0.914"""
    
    def __init__(self):
        # Pauli matrices
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
    
    def build(self, omega_x, omega_y, delta, g_tms=1583.3e6):
        """
        Build Hamiltonian: 
        H = ω_x XI + ω_y YI + δ ZI + g_tms*(XX + YY)/2
        """
        H = (omega_x * self.XI + 
             omega_y * self.YI + 
             delta * self.ZI +
             0.5 * g_tms * self.XX +
             0.5 * g_tms * self.YY)
        return H

class DepolarizingChannel:
    def __init__(self, p=0.01):
        self.p = p
        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        self._build_kraus()
    
    def _build_kraus(self):
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

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3)  # [ω_x, ω_y, δ]
        ).to(device)
        
        # Target CNOT
        self.U_target = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
        
        self.ham = Phase3Hamiltonian()
    
    def forward(self, t):
        return self.net(t)

def train_for_p(p, n_epochs=500):
    """Train with Phase 3 Hamiltonian"""
    print(f"\n{'='*50}")
    print(f"Training with p = {p} (Phase 3 Hamiltonian)")
    print(f"{'='*50}")
    
    model = PINN()
    depolarizing = DepolarizingChannel(p=p)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    dt = 1e-9
    n_steps = 100
    history = []
    
    pbar = tqdm(range(n_epochs), desc=f"p={p}")
    for epoch in pbar:
        t = torch.linspace(0, 1, n_steps).reshape(-1, 1).to(device)
        pulses = model(t)
        
        rho = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        rho[0, 0] = 1.0
        
        for step in range(n_steps):
            omega_x, omega_y, delta = pulses[step]
            H = model.ham.build(omega_x, omega_y, delta)
            
            U = torch.matrix_exp(-1j * dt * H)
            rho = U @ rho @ U.conj().T
            rho = depolarizing.apply(rho)
        
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
    print("PINN WITH PHASE 3 HAMILTONIAN")
    print("="*60)
    
    p_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    all_results = {}
    start_time = time.time()
    
    for p in p_values:
        history = train_for_p(p, n_epochs=500)
        all_results[str(p)] = {
            'final_fidelity': history[-1],
            'history': history
        }
    
    # Plot comparison
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    p_np = np.array(p_values)
    f_np = np.array([all_results[str(p)]['final_fidelity'] for p in p_values])
    
    plt.plot(p_np, f_np, 'bo-', linewidth=2, markersize=8, label='Phase 3 Ham')
    plt.plot(p_np, 1 - 4/3 * p_np, 'r--', label='Theoretical max')
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('With Phase 3 Hamiltonian')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    previous = [1.0, 0.5066, 0.3368, 0.2790, 0.2596, 0.2531]
    plt.plot(p_np, previous, 'gs-', label='Simple Ham')
    plt.plot(p_np, f_np, 'bo-', label='Phase 3 Ham')
    plt.plot(p_np, 1 - 4/3 * p_np, 'r--', label='Theoretical')
    plt.xlabel('Depolarizing rate p')
    plt.ylabel('Gate fidelity')
    plt.title('Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    improvement = [(f_np[i] - previous[i])/previous[i]*100 for i in range(len(p_np))]
    plt.bar([f'p={p}' for p in p_values], improvement, color='purple', alpha=0.7)
    plt.xlabel('Noise level')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement over Simple Ham')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'phase3_hamiltonian_results.png', dpi=150)
    print(f"\n✅ Plot saved: {results_dir / 'phase3_hamiltonian_results.png'}")
    
    # Save results
    with open(data_dir / 'phase3_hamiltonian_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
    print(f"Final fidelities:")
    for p in p_values:
        print(f"  p={p}: {all_results[str(p)]['final_fidelity']:.4f}")

if __name__ == "__main__":
    main()
