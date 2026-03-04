#!/usr/bin/env python
"""Simple test for dataset preparation"""

from pathlib import Path
import pandas as pd
import torch

print("="*60)
print("Testing dataset preparation")
print("="*60)

# Check if index exists
index_file = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'raw_simulations' / 'layouts' / 'layouts_index.csv'
if not index_file.exists():
    print(f"❌ Index file not found: {index_file}")
    exit(1)

print(f"✅ Index file found: {index_file}")

# Load index
df = pd.read_csv(index_file)
print(f"✅ Loaded {len(df)} layouts")

# Check first few rows
print("\nFirst 5 layouts:")
print(df.head())

# Test one conversion manually
row = df.iloc[0]
print(f"\nTesting conversion for layout: {row['layout_id']}")

# Create features
transmon_features = [
    row['junction_width_nm'] / 500.0,
    row['junction_length_nm'] / 300.0,
    row['pad_area_um2'] / 200.0,
    row['gap_to_ground_um'] / 50.0,
]

print(f"Transmon features: {transmon_features}")

# Convert to tensor
x = torch.tensor([transmon_features], dtype=torch.float)
print(f"Tensor shape: {x.shape}")
print("✅ Conversion successful")

print("\n" + "="*60)
