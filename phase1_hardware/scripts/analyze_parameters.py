#!/usr/bin/env python
"""
Analyze parameter distribution of generated layouts
"""

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase1_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def expand_path(path):
    return str(Path(os.path.expanduser(path)).expanduser())

LAYOUTS_DIR = Path(expand_path(config['paths']['layouts_raw']))
FIGURES_DIR = Path(expand_path(config['paths']['figures']))
INDEX_FILE = LAYOUTS_DIR.parent / 'layouts_index.csv'

def main():
    print("="*60)
    print("Parameter Distribution Analysis")
    print("="*60)
    
    # Load index file
    if not INDEX_FILE.exists():
        print(f"Index file not found: {INDEX_FILE}")
        print("Run generate_layouts.py first")
        return
    
    df = pd.read_csv(INDEX_FILE)
    print(f"Loaded {len(df)} layout records")
    
    # Filter valid layouts only
    df_valid = df[df['valid'] == True]
    print(f"Valid layouts: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("No valid layouts found")
        return
    
    # Create parameter distribution plots
    print("\n📊 Creating parameter distribution plots...")
    
    # Select numeric columns
    param_cols = ['junction_width_nm', 'junction_length_nm', 'pad_area_um2',
                  'gap_to_ground_um', 'finger_length_um', 'finger_width_um',
                  'finger_count', 'finger_gap_um', 'hbar_thickness_um',
                  'beam_length_um', 'beam_width_um']
    
    # Histograms
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()
    
    for i, col in enumerate(param_cols):
        if col in df_valid.columns:
            axes[i].hist(df_valid[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(col)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
    
    # Hide empty subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    hist_file = FIGURES_DIR / 'parameter_histograms.png'
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Histograms: {hist_file}")
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df_valid[param_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    corr_file = FIGURES_DIR / 'parameter_correlation.png'
    plt.savefig(corr_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Correlation matrix: {corr_file}")
    
    # Summary statistics
    print("\n📋 Parameter Statistics:")
    print(df_valid[param_cols].describe())
    
    # Save statistics
    stats_file = FIGURES_DIR.parent / 'data' / 'parameter_statistics.csv'
    df_valid[param_cols].describe().to_csv(stats_file)
    print(f"\n✅ Statistics saved to: {stats_file}")
    
    # Check if target achieved
    if len(df_valid) >= 10000:
        print("\n✅ TARGET ACHIEVED: 10,000 valid layouts generated!")
    else:
        print(f"\n⚠️  Target not achieved: Need {10000 - len(df_valid)} more valid layouts")

if __name__ == "__main__":
    main()
