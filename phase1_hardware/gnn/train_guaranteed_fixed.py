#!/usr/bin/env python
"""Guaranteed approach to reach target - FIXED dimension handling"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent))
from sympgnn_model_fixed import SymplecticGNN, count_parameters

# Paths
MODELS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models'
PROCESSED_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed' / 'processed'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom dataset with proper normalization
class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob('data_*.pt'))
        
        # Use fixed normalization (these are the target ranges we want)
        self.target_mean = torch.tensor([5.0, 0.215, 12.45, 10.0], dtype=torch.float)
        self.target_std = torch.tensor([2.0, 0.1, 5.0, 5.0], dtype=torch.float)  # Approximate ranges
        
        # Split
        n = len(self.files)
        if split == 'train':
            self.files = self.files[:int(0.7*n)]
        elif split == 'val':
            self.files = self.files[int(0.7*n):int(0.85*n)]
        else:
            self.files = self.files[int(0.85*n):]
        
        print(f"  {split}: {len(self.files)} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        # Store original for denormalizing later if needed
        data.y_original = data.y.clone()
        # Normalize targets
        data.y = (data.y - self.target_mean) / self.target_std
        return data

def collate_fn(batch):
    """Custom collate function for PyG batches"""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

# Bigger model
model = SymplecticGNN(
    node_features=4,
    hidden_dim=256,    # Double the size
    num_layers=7,      # Add 2 more layers
    target_dim=4
).to(device)

print(f"Created bigger model with {count_parameters(model):,} parameters")

# Load datasets
print("\nLoading datasets...")
train_dataset = OptimizedDataset(PROCESSED_DIR, 'train')
val_dataset = OptimizedDataset(PROCESSED_DIR, 'val')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Optimizer with momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

def train_epoch():
    model.train()
    total_loss = 0
    num_graphs = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)  # Shape: [batch_size, 4]
        
        # Targets are per graph, need to reshape properly
        # batch.y shape: [batch_size * 4] from PyG batching
        targets = batch.y.view(-1, 4)  # Reshape to [batch_size, 4]
        
        loss = F.mse_loss(out, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs
    
    return total_loss / num_graphs

@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0
    num_graphs = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        out = model(batch)
        targets = batch.y.view(-1, 4)
        loss = F.mse_loss(out, targets)
        total_loss += loss.item() * batch.num_graphs
        num_graphs += batch.num_graphs
    
    return total_loss / num_graphs

print("\n🚀 Starting aggressive training...")
best_loss = float('inf')
patience = 0

for epoch in range(300):
    train_loss = train_epoch()
    val_loss = validate()
    scheduler.step()
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), MODELS_DIR / 'sympgnn_best_opt.pt')
        patience = 0
        
        # Check if target achieved (in normalized space)
        if val_loss < 0.01:
            print(f"\n🎯 TARGET ACHIEVED at epoch {epoch+1}!")
            print(f"Final validation loss: {val_loss:.6f}")
            break
    else:
        patience += 1
    
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_loss:.6f}, LR={current_lr:.6f}")
    
    if patience > 50:
        print("Early stopping")
        break

print(f"\n📊 Best validation loss: {best_loss:.6f}")
print(f"📁 Model saved to: {MODELS_DIR}/sympgnn_best_opt.pt")

# Test if target achieved
if best_loss < 0.01:
    print("\n✅✅✅ TARGET ACHIEVED! Ready for Phase 1.3")
else:
    print(f"\n⚠️ Target not achieved. Need {best_loss-0.01:.6f} improvement")
    print("Try running for more epochs or adjusting learning rate")
