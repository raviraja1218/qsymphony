#!/usr/bin/env python
"""Guaranteed approach to reach target - bigger model + better training"""

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
        
        # Calculate targets statistics for better normalization
        all_targets = []
        for f in self.files[:100]:  # Sample first 100 files
            data = torch.load(f)
            all_targets.append(data.y.numpy())
        
        all_targets = np.array(all_targets)
        self.target_mean = torch.tensor(all_targets.mean(axis=0), dtype=torch.float)
        self.target_std = torch.tensor(all_targets.std(axis=0) + 1e-8, dtype=torch.float)
        
        print(f"Target mean: {self.target_mean}")
        print(f"Target std: {self.target_std}")
        
        # Split
        n = len(self.files)
        if split == 'train':
            self.files = self.files[:int(0.7*n)]
        elif split == 'val':
            self.files = self.files[int(0.7*n):int(0.85*n)]
        else:
            self.files = self.files[int(0.85*n):]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        # Normalize targets
        data.y = (data.y - self.target_mean) / self.target_std
        return data

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Larger batch
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Optimizer with momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

def train_epoch():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0
    for data in val_loader:
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out, data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_loader.dataset)

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
        
        # Check if target achieved (in normalized space, target ~0.01)
        if val_loss < 0.01:
            print(f"\n🎯 TARGET ACHIEVED at epoch {epoch+1}!")
            break
    else:
        patience += 1
    
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.6f}")
    
    if patience > 50:
        print("Early stopping")
        break

print(f"\n📊 Best validation loss: {best_loss:.6f}")
print(f"📁 Model saved to: {MODELS_DIR}/sympgnn_best_opt.pt")
