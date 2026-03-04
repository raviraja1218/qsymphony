#!/usr/bin/env python
"""Train larger model with more capacity"""

import torch
from train_sympgnn_fixed3 import train_epoch, validate, LayoutDataset, device
from sympgnn_model_fixed import SymplecticGNN
from pathlib import Path

# Bigger model
model = SymplecticGNN(
    node_features=4,
    hidden_dim=256,  # 128 → 256
    num_layers=7,    # 5 → 7
    target_dim=4
).to(device)

print(f"Larger model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Load datasets
PROCESSED_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed' / 'processed'
train_dataset = LayoutDataset(PROCESSED_DIR, 'train')
val_dataset = LayoutDataset(PROCESSED_DIR, 'val')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train with cosine annealing
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

best_loss = float('inf')
for epoch in range(500):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    scheduler.step()
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), Path.home() / 'projects' / 'qsymphony' / 'results' / 'models' / 'sympgnn_bigger.pt')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_loss:.6f}")
        
    if best_loss < 0.01:
        print(f"✅ Target achieved at epoch {epoch+1}!")
        break
