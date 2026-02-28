#!/usr/bin/env python
"""
Generate 10,000 parameter combinations using Latin Hypercube Sampling
Saves parameters to CSV for batch processing
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml
import os
from datetime import datetime

def load_config():
    with open('configs/parameter_sweep.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_parameters(num_samples, ranges):
    """Generate Latin Hypercube samples"""
    # Create sampler
    sampler = qmc.LatinHypercube(d=len(ranges))
    samples = sampler.random(n=num_samples)
    
    # Scale to parameter ranges
    param_names = list(ranges.keys())
    lower_bounds = [ranges[p]['min'] for p in param_names]
    upper_bounds = [ranges[p]['max'] for p in param_names]
    
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    
    # Create DataFrame
    df = pd.DataFrame(scaled_samples, columns=param_names)
    
    # Add layout IDs
    df.insert(0, 'layout_id', [f'L{str(i).zfill(5)}' for i in range(num_samples)])
    
    # Add metadata
    df['batch'] = df.index // 1000
    df['generated'] = datetime.now().strftime('%Y-%m-%d')
    
    return df

def check_constraints(df, constraints):
    """Apply basic geometric constraints"""
    # Minimum feature size (rounded to nearest 0.5μm for fabrication)
    for col in ['transmon_width_um', 'transmon_height_um', 'coupling_gap_um']:
        df[col] = (df[col] / 0.5).round() * 0.5
    
    # Ensure coupling gap >= minimum
    min_gap = constraints.get('min_coupling_gap_um', 50)
    df.loc[df['coupling_gap_um'] < min_gap, 'coupling_gap_um'] = min_gap
    
    # Aspect ratio check
    df['aspect_ratio'] = df['transmon_width_um'] / df['transmon_height_um']
    max_ratio = constraints.get('max_aspect_ratio', 3.0)
    df['valid'] = (df['aspect_ratio'] <= max_ratio) & (df['aspect_ratio'] >= 1/max_ratio)
    
    return df

def main():
    print("="*60)
    print("Generating 10,000 Layout Parameters using Latin Hypercube")
    print("="*60)
    
    # Load config
    config = load_config()
    ranges = config['parameter_ranges']
    constraints = config['constraints']
    num_layouts = config['output']['num_layouts']
    
    print(f"Parameter space dimensions: {len(ranges)}")
    for name, r in ranges.items():
        print(f"  {name}: {r['min']} → {r['max']} {name.split('_')[-1]}")
    
    # Generate samples
    print(f"\nGenerating {num_layouts} samples...")
    df = generate_parameters(num_layouts, ranges)
    
    # Apply constraints
    print("Applying fabrication constraints...")
    df = check_constraints(df, constraints)
    
    # Save full parameter set
    output_dir = "../datasets/raw_simulations/layouts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all parameters
    csv_file = f"{output_dir}/all_parameters.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved full parameter set to: {csv_file}")
    
    # Save by batch
    for batch in range(10):
        batch_df = df[df['batch'] == batch]
        batch_file = f"{output_dir}/batch_{str(batch).zfill(3)}_params.csv"
        batch_df.to_csv(batch_file, index=False)
        print(f"✅ Batch {batch}: {len(batch_df)} layouts → {batch_file}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Parameter Range Coverage:")
    for col in ranges.keys():
        print(f"  {col}: {df[col].min():.1f} - {df[col].max():.1f} (mean: {df[col].mean():.1f})")
    
    print(f"\nValid layouts: {df['valid'].sum()} / {len(df)}")
    print("="*60)
    
    # Create manifest
    manifest = {
        'num_layouts': len(df),
        'num_batches': 10,
        'parameters': list(ranges.keys()),
        'constraints': constraints,
        'valid_layouts': int(df['valid'].sum()),
        'generation_date': datetime.now().isoformat(),
        'files': {
            'full': 'all_parameters.csv',
            'batches': [f'batch_{str(i).zfill(3)}_params.csv' for i in range(10)]
        }
    }
    
    import json
    with open(f"{output_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Manifest saved to: {output_dir}/manifest.json")

if __name__ == "__main__":
    main()
