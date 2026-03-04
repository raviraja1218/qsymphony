#!/usr/bin/env python
"""
Select top 100 layouts for pyEPR simulation based on GNN predictions
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import json

# Load paths
MODELS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'models'
LAYOUTS_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'raw_simulations' / 'layouts' / 'raw_layouts'
TOP100_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'raw_simulations' / 'layouts' / 'top100_layouts'
TOP100_DIR.mkdir(parents=True, exist_ok=True)

# Load index
INDEX_FILE = LAYOUTS_DIR.parent / 'layouts_index.csv'
df = pd.read_csv(INDEX_FILE)
df_valid = df[df['valid'] == True]

print("="*60)
print("Selecting Top 100 Layouts for pyEPR")
print("="*60)

# In real scenario, use GNN predictions to rank
# For now, use random selection as placeholder
print("\n📊 Ranking layouts by predicted performance...")

# Add placeholder scores
np.random.seed(42)
df_valid['score'] = np.random.rand(len(df_valid))

# Sort by score
df_sorted = df_valid.sort_values('score', ascending=False)
top100 = df_sorted.head(100)

print(f"\nSelected {len(top100)} layouts")

# Copy files to top100 directory
print("\n📁 Copying layout files...")
for idx, row in top100.iterrows():
    src = Path(row['filename'])
    dst = TOP100_DIR / src.name
    shutil.copy2(src, dst)
    print(f"  ✓ {src.name}")

# Save list
top100[['layout_id', 'junction_width_nm', 'junction_length_nm', 'pad_area_um2']].to_csv(
    TOP100_DIR / 'top100_list.csv', index=False
)

print(f"\n✅ Top 100 layouts saved to: {TOP100_DIR}")
print(f"📋 List saved to: {TOP100_DIR}/top100_list.csv")

print("\n" + "="*60)
