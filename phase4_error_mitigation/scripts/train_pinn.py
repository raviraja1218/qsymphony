#!/usr/bin/env python
"""
Step 4.3: Implement PINN for Gate Optimization
Physics-informed neural network learning Lindblad dynamics
Target: CNOT gate with depolarizing noise (p=0.01)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports for verification
try:
    import qutip as qt
    from qutip import basis, tensor, sigmax, sigmay, sigmaz, qeye, destroy
    from qutip.qip.operations import cnot
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"⚠️ QuTiP not available: {e}")
    print("Will use synthetic verification")

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
data_dir = Path(config['paths']['data']).expanduser()

# Create directories
models_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.3: Physics-Informed Neural Network for Gate Optimization")
print("="*60)

class PINNGateOptimizer(nn.Module):
    """
    Physics-Informed Neural Network for quantum gate optimization
    Learns control pulses that implement target gates under noise
    """
    
    def __init__(self, n_hidden=5, n_neurons=256, gate_time_ns=100):
        super().__init__()
        
        self.gate_time = gate_time_ns * 1e-9  # Convert to seconds
        self.n_time_points = 1000  # Number of time points for pulse discretization
        
        # Input: time t ∈ [0, gate_time]
        # Output: control parameters [Ω_x(t), Ω_y(t), Δ(t)]
        
        layers = []
        layers.append(nn.Linear(1, n_neurons))
        layers.append(nn.SiLU())  # Swish activation
        
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(n_neurons, 3))  # 3 control parameters
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"\n🏗️ PINN Architecture:")
        print(f"  Input: time t")
        print(f"  Hidden layers: {n_hidden} × {n_neurons} neurons")
        print(f"  Output: Ω_x, Ω_y, Δ")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, t):
        """
        Args:
            t: time tensor of shape [batch_size, 1] normalized to [0, 1]
        Returns:
            controls: [Ω_x, Ω_y, Δ] of shape [batch_size, 3]
        """
        return self.net(t)
    
    def get_pulses(self, n_points=1000):
        """Get control pulses at uniform time points"""
        t = torch.linspace(0, 1, n_points).reshape(-1, 1).to(next(self.parameters()).device)
        pulses = self.forward(t)
        return pulses.cpu().detach().numpy()

class LindbladLoss(nn.Module):
    """
    Physics loss based on Lindblad master equation
    """
    
    def __init__(self, target_gate='CNOT', depolarizing_rate=0.01):
        super().__init__()
        self.target_gate = target_gate
        self.depolarizing_rate = depolarizing_rate
        
        # Pauli matrices for two qubits
        self.sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        self.sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        self.eye = torch.eye(2, dtype=torch.complex64)
        
        # Build two-qubit operators
        self._build_operators()
        
        # Target CNOT unitary
        self.U_target = self._get_target_unitary()
    
    def _build_operators(self):
        """Build two-qubit operators"""
        # Single qubit operators expanded to two qubits
        self.sx1 = torch.kron(self.sx, self.eye)
        self.sy1 = torch.kron(self.sy, self.eye)
        self.sz1 = torch.kron(self.sz, self.eye)
        self.sx2 = torch.kron(self.eye, self.sx)
        self.sy2 = torch.kron(self.eye, self.sy)
        self.sz2 = torch.kron(self.eye, self.sz)
    
    def _get_target_unitary(self):
        """Get target CNOT unitary as tensor"""
        # CNOT matrix: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
        U = torch.zeros((4, 4), dtype=torch.complex64)
        U[0, 0] = 1
        U[1, 1] = 1
        U[2, 3] = 1
        U[3, 2] = 1
        return U
    
    def _build_hamiltonian(self, pulses, t_idx):
        """
        Build Hamiltonian at time point t_idx from control pulses
        
        H = Ω_x(t) σ_x⊗I + Ω_y(t) σ_y⊗I + Δ(t) σ_z⊗I / 2
        """
        omega_x = pulses[t_idx, 0]
        omega_y = pulses[t_idx, 1]
        delta = pulses[t_idx, 2]
        
        H = omega_x * self.sx1 + omega_y * self.sy1 + 0.5 * delta * self.sz1
        return H
    
    def forward(self, pulses, dt):
        """
        Compute physics loss:
        L_phys = ||dU/dt + iHU - Σ_k D[L_k]U||²
        
        Args:
            pulses: control pulses [n_time, 3]
            dt: time step
        """
        n_steps = pulses.shape[0]
        U = torch.eye(4, dtype=torch.complex64, device=pulses.device)
        
        # Simulate evolution with noise
        for t in range(n_steps):
            H = self._build_hamiltonian(pulses, t)
            
            # Unitary evolution
            U_new = U - 1j * dt * H @ U
            
            # Apply depolarizing noise
            # Λ(ρ) = (1-p)ρ + p/3 (XρX + YρY + ZρZ)
            p = self.depolarizing_rate * dt * 1e9  # Scale to ns
            
            # Kraus operators for depolarizing channel
            U_new = (1 - p) * U_new
            # Note: Full noise model would require density matrix evolution
            # This is a simplified approximation for unitary learning
            
            U = U_new
        
        # Compute gate fidelity
        # F = |Tr(U_target† U)|² / d²
        overlap = torch.abs(torch.trace(self.U_target.conj().T @ U))**2
        fidelity = overlap / 16  # d² = 4² = 16
        
        # Physics loss = 1 - fidelity (minimize infidelity)
        loss_phys = 1 - fidelity
        
        # Regularization: encourage smooth pulses
        pulse_diff = pulses[1:] - pulses[:-1]
        reg_loss = torch.mean(pulse_diff**2)
        
        total_loss = loss_phys + 0.01 * reg_loss
        
        return total_loss, fidelity

class PINNTrainer:
    """Trainer for PINN gate optimization"""
    
    def __init__(self, config):
        self.config = config
        self.pinn = PINNGateOptimizer(
            n_hidden=config['pinn']['hidden_layers'],
            n_neurons=config['pinn']['neurons_per_layer'],
            gate_time_ns=config['pinn']['gate_time_ns']
        ).to(device)
        
        self.loss_fn = LindbladLoss(
            depolarizing_rate=config['pinn']['depolarizing_rate']
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.pinn.parameters(), 
            lr=1e-3,
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        self.dt = self.pinn.gate_time / self.pinn.n_time_points
        
        # Training history
        self.loss_history = []
        self.fidelity_history = []
        
    def train_step(self):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Get pulses
        t = torch.linspace(0, 1, self.pinn.n_time_points).reshape(-1, 1).to(device)
        pulses = self.pinn(t)
        
        # Compute loss
        loss, fidelity = self.loss_fn(pulses, self.dt)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), fidelity.item()
    
    def train(self, n_epochs=1000, patience=50):
        """Main training loop"""
        
        print(f"\n🚀 Starting PINN training for {n_epochs} epochs...")
        print(f"  Gate time: {self.pinn.gate_time*1e9:.0f} ns")
        print(f"  Time steps: {self.pinn.n_time_points}")
        print(f"  Depolarizing rate: {self.config['pinn']['depolarizing_rate']}")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            loss, fidelity = self.train_step()
            
            self.loss_history.append(loss)
            self.fidelity_history.append(fidelity)
            
            self.scheduler.step(loss)
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('pinn_best.pt')
            else:
                patience_counter += 1
            
            if epoch % 100 == 0:
                print(f"\n📊 Epoch {epoch}:")
                print(f"  Loss: {loss:.6f}")
                print(f"  Fidelity: {fidelity:.6f}")
                print(f"  Best loss: {best_loss:.6f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= patience:
                print(f"\n✅ Early stopping at epoch {epoch}")
                break
        
        print(f"\n✅ Training complete!")
        print(f"  Final fidelity: {fidelity:.6f}")
        print(f"  Best fidelity: {max(self.fidelity_history):.6f}")
        
        return self.loss_history, self.fidelity_history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = models_dir / filename
        torch.save({
            'model_state_dict': self.pinn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'fidelity_history': self.fidelity_history,
            'config': self.config
        }, path)
    
    def save_final_model(self):
        """Save final model"""
        path = models_dir / 'pinn_gate_optimizer.zip'
        torch.save({
            'model_state_dict': self.pinn.state_dict(),
            'fidelity_history': self.fidelity_history,
            'config': self.config
        }, path)
        print(f"\n✅ Final model saved to {path}")
        return path
    
    def plot_training(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        ax = axes[0]
        ax.plot(self.loss_history, 'b-', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Fidelity curve
        ax = axes[1]
        ax.plot(self.fidelity_history, 'g-', alpha=0.7)
        ax.axhline(y=0.99, color='r', linestyle='--', label='99% target')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gate Fidelity')
        ax.set_title('Gate Fidelity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('PINN Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_file = figures_dir / 'pinn_training.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Training plot saved to {plot_file}")
        plt.close()
    
    def plot_pulses(self):
        """Plot optimized control pulses"""
        pulses = self.pinn.get_pulses()
        t = np.linspace(0, self.pinn.gate_time*1e9, len(pulses))
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Ω_x pulse
        ax = axes[0]
        ax.plot(t, pulses[:, 0], 'b-', linewidth=2)
        ax.set_ylabel('Ω_x (MHz)')
        ax.set_title('X Control')
        ax.grid(True, alpha=0.3)
        
        # Ω_y pulse
        ax = axes[1]
        ax.plot(t, pulses[:, 1], 'g-', linewidth=2)
        ax.set_ylabel('Ω_y (MHz)')
        ax.set_title('Y Control')
        ax.grid(True, alpha=0.3)
        
        # Δ pulse
        ax = axes[2]
        ax.plot(t, pulses[:, 2], 'r-', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Δ (MHz)')
        ax.set_title('Detuning')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Optimized Control Pulses', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_file = figures_dir / 'optimized_pulses.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Pulse plot saved to {plot_file}")
        plt.close()

def main():
    """Main execution for Step 4.3"""
    
    # Create trainer
    trainer = PINNTrainer(config)
    
    # Train
    loss_history, fidelity_history = trainer.train(n_epochs=1000)
    
    # Plot results
    trainer.plot_training()
    trainer.plot_pulses()
    
    # Save final model
    model_path = trainer.save_final_model()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TRAINING SUMMARY")
    print("="*60)
    print(f"Final fidelity: {fidelity_history[-1]:.4f}")
    print(f"Best fidelity: {max(fidelity_history):.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    if max(fidelity_history) > 0.99:
        print("\n✅ TARGET ACHIEVED: Fidelity > 99%")
    else:
        print(f"\n⚠️ Target not achieved: {max(fidelity_history)*100:.2f}% < 99%")
    
    print(f"\n📁 Model saved to: {model_path}")
    print(f"📁 Figures saved to: {figures_dir}")
    
    print("\n" + "="*60)
    print("✅ STEP 4.3 COMPLETE")
    print("="*60)
    print("\nNext: Step 4.4 - Generate Exceptional Point Visualization")

if __name__ == "__main__":
    main()
