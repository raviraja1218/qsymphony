#!/usr/bin/env python
"""Continue training from checkpoint with proper PyG DataLoader"""

import torch
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from train_sympgnn_fixed3 import train_epoch, validate, LayoutDataset
from sympgnn_model_fixed import SymplecticGNN

# Paths
MODELS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models'
PROCESSED_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed' / 'processed'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
print("Loading datasets...")
train_dataset = LayoutDataset(PROCESSED_DIR, 'train')
val_dataset = LayoutDataset(PROCESSED_DIR, 'val')

# Use PyG DataLoader instead of torch DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load previous model
print("Loading previous model...")
model = SymplecticGNN(
    node_features=4,
    hidden_dim=128,
    num_layers=5,
    target_dim=4
).to(device)

# Load the trained weights
model.load_state_dict(torch.load(MODELS_DIR / 'sympgnn_final.pt'))
print("Model loaded successfully")

# New optimizer with lower LR for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

print("\n🚀 Continuing training with LR=1e-5 for 200 more epochs...")
print("-" * 50)

best_val_loss = 0.069142  # Current best

for epoch in range(200):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # Save if improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODELS_DIR / 'sympgnn_continued.pt')
        print(f"✓ New best: {best_val_loss:.6f}")
    
    # Check if target achieved
    if val_loss < 0.01:
        print(f"\n🎯 TARGET ACHIEVED at epoch {epoch+1}!")
        print(f"Final validation loss: {val_loss:.6f}")
        torch.save(model.state_dict(), MODELS_DIR / 'sympgnn_target.pt')
        break
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_val_loss:.6f}")
        
        # If no improvement in 50 epochs, increase LR temporarily
        if epoch > 50 and val_loss > best_val_loss * 0.99:
            print("  ↻ Plateau detected, bumping LR temporarily")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
else:
    print(f"\n📊 Training complete after 200 epochs")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Target needed: <0.01")
    print(f"Gap to target: {best_val_loss - 0.01:.6f}")

print(f"\n📁 Model saved to: {MODELS_DIR}/sympgnn_continued.pt")
