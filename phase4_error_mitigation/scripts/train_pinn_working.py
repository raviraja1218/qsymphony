#!/usr/bin/env python
"""
Step 4.3: PINN for Gate Optimization - WORKING VERSION
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase4_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Paths
models_dir = Path(config['paths']['models']).expanduser()
figures_dir = Path(config['paths']['figures']).expanduser()

# Create directories
models_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.3: PINN for Gate Optimization - WORKING VERSION")
print("="*60)

class PINN(nn.Module):
    def __init__(self, n_hidden=5, n_neurons=256, gate_time_ns=100):
        super().__init__()
        
        self.gate_time = gate_time_ns * 1e-9
        self.n_time_points = 1000
        
        # Deeper network with batch norm
        layers = []
        layers.append(nn.Linear(1, n_neurons))
        layers.append(nn.BatchNorm1d(n_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.BatchNorm1d(n_neurons))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
        
        layers.append(nn.Linear(n_neurons, 3))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"\n🏗️ PINN Architecture:")
        print(f"  Hidden layers: {n_hidden} × {n_neurons}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
    
    def forward(self, t):
        return self.net(t)
    
    def get_pulses(self, n_points=1000):
        t = torch.linspace(0, 1, n_points).reshape(-1, 1).to(next(self.parameters()).device)
        with torch.no_grad():
            pulses = self.forward(t)
        return pulses.cpu().numpy()

class PINNTrainer:
    def __init__(self, config):
        self.config = config
        self.pinn = PINN(
            n_hidden=config['pinn']['hidden_layers'],
            n_neurons=config['pinn']['neurons_per_layer'],
            gate_time_ns=config['pinn']['gate_time_ns']
        ).to(device)
        
        # Use Adam with higher learning rate
        self.optimizer = optim.Adam(
            self.pinn.parameters(), 
            lr=1e-2
        )
        
        # Step scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.5
        )
        
        self.dt = self.pinn.gate_time / self.pinn.n_time_points
        
        self.loss_history = []
        self.fidelity_history = []
        
        # Pauli matrices
        self.I = torch.eye(2, dtype=torch.complex64, device=device)
        self.X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        self.Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        self.Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        
        # Target CNOT
        self.U_target = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
    
    def compute_fidelity(self, pulses):
        n_steps = pulses.shape[0]
        
        # Start with identity
        U = torch.eye(4, dtype=torch.complex64, device=device)
        
        # Simulate evolution
        for t in range(n_steps):
            # Build Hamiltonian for two qubits
            # H = Ω_x(t) (X⊗I + I⊗X) + Ω_y(t) (Y⊗I + I⊗Y) + Δ(t) (Z⊗I + I⊗Z)/2
            H1 = torch.kron(self.X, self.I) + torch.kron(self.I, self.X)
            H2 = torch.kron(self.Y, self.I) + torch.kron(self.I, self.Y)
            H3 = 0.5 * (torch.kron(self.Z, self.I) + torch.kron(self.I, self.Z))
            
            H = pulses[t, 0] * H1 + pulses[t, 1] * H2 + pulses[t, 2] * H3
            
            # Evolve
            U = U @ torch.matrix_exp(-1j * self.dt * H)
        
        # Compute average gate fidelity
        # F_avg = (|Tr(U_target† U)|² + 2) / 6 for two qubits
        overlap = torch.abs(torch.trace(self.U_target.conj().T @ U))
        fidelity = (overlap**2 + 2) / 6
        
        return fidelity, U
    
    def train_step(self):
        self.optimizer.zero_grad()
        
        # Get pulses
        t = torch.linspace(0, 1, self.pinn.n_time_points).reshape(-1, 1).to(device)
        pulses = self.pinn(t)
        
        # Compute fidelity
        fidelity, _ = self.compute_fidelity(pulses)
        
        # Loss = 1 - fidelity
        loss = 1 - fidelity
        
        # Add small regularization to encourage smooth pulses
        pulse_diff = pulses[1:] - pulses[:-1]
        reg_loss = 0.01 * torch.mean(pulse_diff**2)
        
        total_loss = loss + reg_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item(), fidelity.item()
    
    def train(self, n_epochs=2000):
        print(f"\n🚀 Starting PINN training for {n_epochs} epochs...")
        
        best_fidelity = 0
        
        for epoch in range(n_epochs):
            loss, fidelity = self.train_step()
            
            self.loss_history.append(loss)
            self.fidelity_history.append(fidelity)
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                if epoch % 100 == 0:
                    self.save_checkpoint('pinn_best.pt')
            
            if epoch % 100 == 0:
                print(f"\n📊 Epoch {epoch}:")
                print(f"  Loss: {loss:.6f}")
                print(f"  Fidelity: {fidelity:.4f}")
                print(f"  Best fidelity: {best_fidelity:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        print(f"\n✅ Training complete!")
        print(f"  Best fidelity: {best_fidelity:.4f}")
        
        return self.loss_history, self.fidelity_history
    
    def save_checkpoint(self, filename):
        path = models_dir / filename
        torch.save({
            'model_state_dict': self.pinn.state_dict(),
            'fidelity_history': self.fidelity_history,
        }, path)
    
    def save_final_model(self):
        path = models_dir / 'pinn_gate_optimizer.zip'
        torch.save({
            'model_state_dict': self.pinn.state_dict(),
            'fidelity_history': self.fidelity_history,
        }, path)
        print(f"\n✅ Final model saved to {path}")
        return path
    
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curve
        ax = axes[0, 0]
        ax.plot(self.loss_history, 'b-', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Fidelity curve
        ax = axes[0, 1]
        ax.plot(self.fidelity_history, 'g-', alpha=0.7)
        ax.axhline(y=0.99, color='r', linestyle='--', label='99% target')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fidelity')
        ax.set_title('Gate Fidelity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pulses
        pulses = self.pinn.get_pulses()
        t = np.linspace(0, self.pinn.gate_time*1e9, len(pulses))
        
        ax = axes[1, 0]
        ax.plot(t, pulses[:, 0], 'b-', label='Ω_x', linewidth=2)
        ax.plot(t, pulses[:, 1], 'g-', label='Ω_y', linewidth=2)
        ax.plot(t, pulses[:, 2], 'r-', label='Δ', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Amplitude (MHz)')
        ax.set_title('Optimized Control Pulses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final stats
        ax = axes[1, 1]
        ax.text(0.5, 0.7, f'Final Fidelity:\n{self.fidelity_history[-1]:.4f}', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.4, f'Best Fidelity:\n{max(self.fidelity_history):.4f}',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.1, f'Total Parameters:\n{sum(p.numel() for p in self.pinn.parameters()):,}',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        
        plt.suptitle('PINN Training Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = figures_dir / 'pinn_results.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Results plot saved to {plot_file}")
        plt.close()

def main():
    trainer = PINNTrainer(config)
    trainer.train(n_epochs=2000)
    trainer.plot_results()
    trainer.save_final_model()
    
    final_fidelity = max(trainer.fidelity_history)
    print("\n" + "="*60)
    print("📋 TRAINING SUMMARY")
    print("="*60)
    print(f"Best fidelity achieved: {final_fidelity:.4f}")
    
    if final_fidelity > 0.99:
        print("\n✅ TARGET ACHIEVED: Fidelity > 99%")
    else:
        print(f"\n⚠️ Target not achieved: {final_fidelity*100:.2f}% < 99%")
    
    print("\n" + "="*60)
    print("✅ STEP 4.3 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
