#!/usr/bin/env python
"""Analyze training results and suggest improvements"""

import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training logs
log_file = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'training_logs' / 'training_logs.json'
with open(log_file, 'r') as f:
    logs = json.load(f)

train_losses = logs['train_losses']
val_losses = logs['val_losses']

print("="*60)
print("TRAINING ANALYSIS")
print("="*60)

print(f"\n📊 Final Statistics:")
print(f"  Train Loss: {train_losses[-1]:.6f}")
print(f"  Val Loss: {val_losses[-1]:.6f}")
print(f"  Constraint Satisfaction: {logs['constraint_satisfaction']*100:.2f}%")

# Check if model is underfitting or overfitting
train_final = train_losses[-100:]
val_final = val_losses[-100:]
train_mean = np.mean(train_final)
val_mean = np.mean(val_final)

print(f"\n🔍 Last 100 epochs average:")
print(f"  Train: {train_mean:.6f}")
print(f"  Val: {val_mean:.6f}")

if val_mean > train_mean * 1.1:
    print("  ⚠️  Slight overfitting detected")
elif val_mean < train_mean * 0.95:
    print("  ⚠️  Unusual: val better than train")
else:
    print("  ✅ No overfitting")

# Plot learning curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train', alpha=0.7)
plt.plot(val_losses, label='Val', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Full Training Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(train_losses[-200:], label='Train', alpha=0.7)
plt.plot(val_losses[-200:], label='Val', alpha=0.7)
plt.xlabel('Epoch (last 200)')
plt.ylabel('Loss')
plt.title('Last 200 Epochs - Plateau Detected')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'figures' / 'training_analysis.png', dpi=150)
plt.show()

print("\n💡 Recommendations:")

# Check if loss plateaued
loss_improvement = val_losses[-100] - val_losses[-1]
if loss_improvement < 0.001:
    print("  - Model has plateaued (loss improvement < 0.001 in last 100 epochs)")
    
    # Try different strategies
    print("\n🔧 Try one of these strategies:")
    print("  1. Increase model capacity (hidden_dim 128 → 256)")
    print("  2. Add more layers (5 → 7)")
    print("  3. Change learning rate schedule")
    print("  4. Use different optimizer (AdamW → SGD with momentum)")
    print("  5. Add batch normalization")

print("\n" + "="*60)
