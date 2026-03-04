#!/usr/bin/env python
"""Continue training from checkpoint with lower learning rate"""

import torch
from pathlib import Path
from train_sympgnn_fixed3 import train_epoch, validate, LayoutDataset
from sympgnn_model_fixed import SymplecticGNN

# Paths
MODELS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models'
PROCESSED_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed' / 'processed'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load previous model
model = SymplecticGNN(node_features=4, hidden_dim=128, num_layers=5, target_dim=4).to(device)
model.load_state_dict(torch.load(MODELS_DIR / 'sympgnn_final.pt'))

# Load datasets
train_dataset = LayoutDataset(PROCESSED_DIR, 'train')
val_dataset = LayoutDataset(PROCESSED_DIR, 'val')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# New optimizer with lower LR
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

print("Continuing training with LR=1e-5 for 200 more epochs...")
for epoch in range(200):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
    if val_loss < 0.01:
        print(f"✅ Target achieved at epoch {epoch+1}!")
        break

torch.save(model.state_dict(), MODELS_DIR / 'sympgnn_final_v2.pt')
