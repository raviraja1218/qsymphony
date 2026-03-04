#!/usr/bin/env python
"""
Visualize generated layouts and create preview images
Fixed import paths for Qiskit Metal 0.1.2
"""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import pandas as pd

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase1_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def expand_path(path):
    return str(Path(os.path.expanduser(path)).expanduser())

LAYOUTS_DIR = Path(expand_path(config['paths']['layouts_raw']))
INDEX_FILE = LAYOUTS_DIR.parent / 'layouts_index.csv'
FIGURES_DIR = Path(expand_path(config['paths']['figures']))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_layout_metadata(layout_id):
    """Load just the metadata from a layout file without using Qiskit Metal"""
    filename = LAYOUTS_DIR / f'layout_{layout_id:06d}.json'
    if filename.exists():
        with open(filename, 'r') as f:
            data = json.load(f)
            return data.get('metadata', {})
    return None

def create_parameter_scatter():
    """Create scatter plot of parameters"""
    if not INDEX_FILE.exists():
        print(f"Index file not found: {INDEX_FILE}")
        return
    
    df = pd.read_csv(INDEX_FILE)
    df_valid = df[df['valid'] == True]
    
    if len(df_valid) == 0:
        print("No valid layouts found")
        return
    
    # Create scatter plot matrix (just a few key parameters)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Junction width vs length
    axes[0, 0].scatter(df_valid['junction_width_nm'], df_valid['junction_length_nm'], 
                       alpha=0.3, s=1, c='blue')
    axes[0, 0].set_xlabel('Junction Width (nm)')
    axes[0, 0].set_ylabel('Junction Length (nm)')
    axes[0, 0].set_title('Junction Dimensions')
    
    # Plot 2: Pad area vs gap
    axes[0, 1].scatter(df_valid['pad_area_um2'], df_valid['gap_to_ground_um'], 
                       alpha=0.3, s=1, c='red')
    axes[0, 1].set_xlabel('Pad Area (µm²)')
    axes[0, 1].set_ylabel('Gap to Ground (µm)')
    axes[0, 1].set_title('Transmon Geometry')
    
    # Plot 3: Finger length vs count
    axes[1, 0].scatter(df_valid['finger_length_um'], df_valid['finger_count'], 
                       alpha=0.3, s=1, c='green')
    axes[1, 0].set_xlabel('Finger Length (µm)')
    axes[1, 0].set_ylabel('Finger Count')
    axes[1, 0].set_title('Capacitor Design')
    
    # Plot 4: Beam length vs width
    axes[1, 1].scatter(df_valid['beam_length_um'], df_valid['beam_width_um'], 
                       alpha=0.3, s=1, c='purple')
    axes[1, 1].set_xlabel('Beam Length (µm)')
    axes[1, 1].set_ylabel('Beam Width (µm)')
    axes[1, 1].set_title('Resonator Dimensions')
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / 'parameter_scatter.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Scatter plot saved to: {output_file}")

def create_parameter_histograms():
    """Create histogram of parameters (already done by analyze_parameters.py, but we'll create a cleaner version)"""
    if not INDEX_FILE.exists():
        return
    
    df = pd.read_csv(INDEX_FILE)
    df_valid = df[df['valid'] == True]
    
    # Select key parameters
    params = ['junction_width_nm', 'junction_length_nm', 'pad_area_um2', 
              'finger_length_um', 'finger_count', 'beam_length_um']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        axes[i].hist(df_valid[param], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[i].set_xlabel(param.replace('_', ' '))
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'Distribution of {param}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = FIGURES_DIR / 'parameter_histograms_clean.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Clean histograms saved to: {output_file}")

def create_sample_summary():
    """Create a text summary of the dataset"""
    if not INDEX_FILE.exists():
        return
    
    df = pd.read_csv(INDEX_FILE)
    df_valid = df[df['valid'] == True]
    
    summary = f"""
    ========================================
    DATASET SUMMARY
    ========================================
    Total layouts: {len(df)}
    Valid layouts: {len(df_valid)}
    Validation rate: {len(df_valid)/len(df)*100:.1f}%
    
    Parameter ranges:
      Junction width: {df_valid['junction_width_nm'].min()} - {df_valid['junction_width_nm'].max()} nm
      Junction length: {df_valid['junction_length_nm'].min()} - {df_valid['junction_length_nm'].max()} nm
      Pad area: {df_valid['pad_area_um2'].min()} - {df_valid['pad_area_um2'].max()} µm²
      Finger length: {df_valid['finger_length_um'].min()} - {df_valid['finger_length_um'].max()} µm
      Beam length: {df_valid['beam_length_um'].min()} - {df_valid['beam_length_um'].max()} µm
    
    File locations:
      Layouts: {LAYOUTS_DIR}
      Index: {INDEX_FILE}
      Figures: {FIGURES_DIR}
    ========================================
    """
    
    print(summary)
    
    # Save summary
    summary_file = FIGURES_DIR.parent / 'data' / 'dataset_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"✅ Summary saved to: {summary_file}")

def main():
    print("="*60)
    print("Layout Visualization (without Qiskit Metal)")
    print("="*60)
    
    # Check if layouts exist
    layout_files = list(LAYOUTS_DIR.glob("*.json"))
    print(f"Found {len(layout_files)} layout files")
    
    if len(layout_files) == 0:
        print("No layouts to visualize. Run generate_layouts.py first.")
        return
    
    # Create visualizations
    print("\n📊 Creating scatter plots...")
    create_parameter_scatter()
    
    print("\n📊 Creating histograms...")
    create_parameter_histograms()
    
    print("\n📊 Creating summary...")
    create_sample_summary()
    
    print(f"\n✅ All visualizations saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
