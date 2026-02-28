#!/usr/bin/env python
"""Test with first 10 layouts from batch 0"""

import pandas as pd
from generate_layouts_fixed import create_single_layout
import os

# Load first 10 layouts from batch 0
params_file = "../datasets/raw_simulations/layouts/batch_000_params.csv"
df = pd.read_csv(params_file)
df_first10 = df.head(10)

print("="*60)
print("Testing with first 10 layouts")
print("="*60)

output_dir = "../datasets/raw_simulations/layouts/test_batch"
os.makedirs(output_dir, exist_ok=True)

success = 0
for idx, row in df_first10.iterrows():
    params = row.to_dict()
    print(f"\nGenerating {params['layout_id']}...")
    
    result = create_single_layout(params, output_dir)
    
    if result['status'] == 'success':
        success += 1
        print(f"  ✅ Success - GDS: {result['gds_status']}")
    else:
        print(f"  ❌ Failed: {result.get('error', 'Unknown')}")

print(f"\n✅ {success}/10 layouts generated successfully")
