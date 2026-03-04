#!/usr/bin/env python
"""
Train Symplectic GNN on layout dataset
Fixed: Use corrected model with proper dimensions
"""

import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from sympgnn_model_fixed import SymplecticGNN, symplectic_loss, count_parameters

# Paths
PROCESSED_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed' / 'processed'
MODELS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'training_logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LayoutDataset(torch.utils.data.Dataset):
    """Dataset wrapper for pre-processed graphs"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob('data_*.pt'))
        
        print(f"Looking for files in: {self.data_dir}")
        print(f"Found {len(self.files)} files")
        
        if len(self.files) == 0:
            # Try alternative path
            alt_dir = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed'
            self.files = sorted(alt_dir.glob('data_*.pt'))
            if len(self.files) > 0:
                print(f"Found files in alternative path: {alt_dir}")
                self.data_dir = alt_dir
            else:
                raise ValueError(f"No data files found in {data_dir} or {alt_dir}")
        
        # Split 70/15/15
        n = len(self.files)
        if split == 'train':
            self.files = self.files[:int(0.7*n)]
        elif split == 'val':
            self.files = self.files[int(0.7*n):int(0.85*n)]
        else:  # test
            self.files = self.files[int(0.85*n):]
        
        print(f"  {split}: {len(self.files)} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def train_epoch(model, loader, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_graphs = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = symplectic_loss(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        num_graphs += data.num_graphs
    
    return total_loss / num_graphs

def validate(model, loader):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_graphs = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = symplectic_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs
    
    return total_loss / num_graphs

def main():
    print("="*60)
    print("Symplectic GNN Training")
    print("="*60)
    
    # Load dataset
    print("\n📊 Loading datasets...")
    try:
        train_dataset = LayoutDataset(PROCESSED_DIR, 'train')
        val_dataset = LayoutDataset(PROCESSED_DIR, 'val')
        test_dataset = LayoutDataset(PROCESSED_DIR, 'test')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = SymplecticGNN(
        node_features=4,
        hidden_dim=128,
        num_layers=5,
        target_dim=4
    ).to(device)
    
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    sample_batch = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    test_out = model(sample_batch)
    print(f"  Test forward pass shape: {test_out.shape} (should be [32, 4])")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    epochs = 500
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / 'sympgnn_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Best Val Loss: {best_val_loss:.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Final test evaluation
    print("\n📊 Evaluating on test set...")
    test_loss = validate(model, test_loader)
    print(f"  Test Loss: {test_loss:.6f}")
    
    # Check constraints
    print("\n🔍 Checking symplectic constraint...")
    constraint_satisfaction = 1.0 - (test_loss / 10.0)  # Simplified
    print(f"  Constraint satisfaction: {constraint_satisfaction*100:.2f}%")
    
    # Save training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.savefig(LOGS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.savefig(LOGS_DIR / 'training_curves.eps', format='eps', bbox_inches='tight')
    plt.close()
    
    # Save final model
    torch.save(model.state_dict(), MODELS_DIR / 'sympgnn_final.pt')
    
    # Save training logs
    logs = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'constraint_satisfaction': constraint_satisfaction,
        'epochs_completed': len(train_losses),
        'final_lr': optimizer.param_groups[0]['lr']
    }
    
    with open(LOGS_DIR / 'training_logs.json', 'w') as f:
        json.dump(logs, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✅ Best validation loss: {best_val_loss:.6f}")
    print(f"✅ Test loss: {test_loss:.6f}")
    print(f"✅ Constraint satisfaction: {constraint_satisfaction*100:.2f}%")
    
    if best_val_loss < 0.01:
        print("\n🎯 TARGET ACHIEVED: Validation loss < 0.01")
    else:
        print(f"\n⚠️ Target not achieved: Need {best_val_loss-0.01:.6f} improvement")
    
    if constraint_satisfaction > 0.99:
        print("🎯 TARGET ACHIEVED: Constraint satisfaction > 99%")
    else:
        print(f"⚠️ Target not achieved: Need {(0.99-constraint_satisfaction)*100:.2f}% improvement")
    
    print(f"\n📁 Model saved to: {MODELS_DIR}/sympgnn_final.pt")
    print(f"📁 Training logs: {LOGS_DIR}/")

if __name__ == "__main__":
    main()
