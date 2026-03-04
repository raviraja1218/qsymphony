#!/usr/bin/env python
"""
Step 4.3: Implement PINN for Gate Optimization - FIXED VERSION
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
print("STEP 4.3: Physics-Informed Neural Network (FIXED)")
print("="*60)

class ImprovedPINN(nn.Module):
    """PINN architecture"""
    
    def __init__(self, n_hidden=5, n_neurons=256, gate_time_ns=100):
        super().__init__()
        
        self.gate_time = gate_time_ns * 1e-9
        self.n_time_points = 1000
        
        # Fourier feature embedding
        self.fourier_scale = 30.0
        self.fourier_features = 64
        
        # Fourier layer
        self.fourier = nn.Linear(1, self.fourier_features * 2, bias=False)
        
        # Main network
        layers = []
        layers.append(nn.Linear(self.fourier_features * 2, n_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(n_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(n_neurons, 3))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize Fourier features
        nn.init.normal_(self.fourier.weight, mean=0.0, std=self.fourier_scale)
        
        print(f"\n🏗️ PINN Architecture:")
        print(f"  Fourier features: {self.fourier_features}")
        print(f"  Hidden layers: {n_hidden} × {n_neurons}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module is not self.fourier:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)
    
    def forward(self, t):
        # Fourier features
        fourier_out = self.fourier(t)
        x = torch.cat([torch.sin(fourier_out), torch.cos(fourier_out)], dim=-1)
        return self.net(x)
    
    def get_pulses(self, n_points=1000):
        t = torch.linspace(0, 1, n_points).reshape(-1, 1).to(next(self.parameters()).device)
        pulses = self.forward(t)
        return pulses.cpu().detach().numpy()

class PINNTrainer:
    """Trainer for PINN"""
    
    def __init__(self, config):
        self.config = config
        self.pinn = ImprovedPINN(
            n_hidden=config['pinn']['hidden_layers'],
            n_neurons=config['pinn']['neurons_per_layer'],
            gate_time_ns=config['pinn']['gate_time_ns']
        ).to(device)
        
        # Use AdamW
        self.optimizer = optim.AdamW(
            self.pinn.parameters(), 
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500, eta_min=1e-6
        )
        
        self.dt = self.pinn.gate_time / self.pinn.n_time_points
        
        # Training history
        self.loss_history = []
        self.fidelity_history = []
        
        # Target CNOT matrix
        self.U_target = torch.zeros((4, 4), dtype=torch.complex64, device=device)
        self.U_target[0, 0] = 1
        self.U_target[1, 1] = 1
        self.U_target[2, 3] = 1
        self.U_target[3, 2] = 1
    
    def compute_fidelity(self, pulses):
        """Compute gate fidelity by simulating evolution"""
        n_steps = pulses.shape[0]
        
        # Start with identity
        U = torch.eye(4, dtype=torch.complex64, device=device)
        
        # Simulate evolution
        for t in range(n_steps):
            # Simple Hamiltonian (simplified)
            H = torch.zeros((4, 4), dtype=torch.complex64, device=device)
            H[0, 0] = pulses[t, 2]  # Simple diagonal
            H[1, 1] = pulses[t, 2]
            H[2, 2] = -pulses[t, 2]
            H[3, 3] = -pulses[t, 2]
            
            # Add coupling
            H[2, 3] = pulses[t, 0] + 1j * pulses[t, 1]
            H[3, 2] = pulses[t, 0] - 1j * pulses[t, 1]
            
            # Evolve
            U = U @ torch.matrix_exp(-1j * self.dt * H)
        
        # Compute fidelity
        overlap = torch.abs(torch.trace(self.U_target.conj().T @ U))**2
        fidelity = overlap / 16
        
        return fidelity, U
    
    def train_step(self):
        self.optimizer.zero_grad()
        
        # Get pulses
        t = torch.linspace(0, 1, self.pinn.n_time_points).reshape(-1, 1).to(device)
        pulses = self.pinn(t)
        
        # Compute fidelity
        fidelity, U = self.compute_fidelity(pulses)
        
        # Loss = 1 - fidelity + regularization
        loss = 1 - fidelity
        
        # Smoothness regularization
        pulse_diff = pulses[1:] - pulses[:-1]
        reg_loss = 0.001 * torch.mean(pulse_diff**2)
        
        total_loss = loss + reg_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return total_loss.item(), fidelity.item()
    
    def train(self, n_epochs=1000):
        print(f"\n🚀 Starting PINN training for {n_epochs} epochs...")
        
        best_fidelity = 0
        
        for epoch in range(n_epochs):
            loss, fidelity = self.train_step()
            
            self.loss_history.append(loss)
            self.fidelity_history.append(fidelity)
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                self.save_checkpoint('pinn_best.pt')
            
            if epoch % 100 == 0:
                print(f"\n📊 Epoch {epoch}:")
                print(f"  Loss: {loss:.6f}")
                print(f"  Fidelity: {fidelity:.4f}")
                print(f"  Best fidelity: {best_fidelity:.4f}")
        
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
        
        # Training curves
        ax = axes[0, 0]
        ax.plot(self.loss_history, 'b-', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
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
        ax.plot(t, pulses[:, 0], 'b-', label='Ω_x')
        ax.plot(t, pulses[:, 1], 'g-', label='Ω_y')
        ax.plot(t, pulses[:, 2], 'r-', label='Δ')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Control Pulses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final fidelity
        ax = axes[1, 1]
        ax.text(0.5, 0.6, f'Final Fidelity:\n{self.fidelity_history[-1]:.4f}', 
                ha='center', va='center', fontsize=14)
        ax.text(0.5, 0.3, f'Best Fidelity:\n{max(self.fidelity_history):.4f}',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        plt.suptitle('PINN Training Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = figures_dir / 'pinn_results.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Results plot saved to {plot_file}")
        plt.close()

def main():
    trainer = PINNTrainer(config)
    trainer.train(n_epochs=1000)
    trainer.plot_results()
    trainer.save_final_model()
    
    print("\n" + "="*60)
    print("✅ STEP 4.3 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
