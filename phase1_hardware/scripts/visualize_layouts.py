#!/usr/bin/env python
"""
Visualize generated layouts and create preview images
Corrected paths for Qiskit Metal 0.1.2
"""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Qiskit Metal imports - corrected paths
try:
    from qiskit_metal import designs, draw
    from qiskit_metal.toolbox_python.load_design import load_design_from_json
except ImportError as e:
    print(f"Qiskit Metal import error: {e}")
    print("Checking available modules...")
    import qiskit_metal.toolbox_python as toolbox
    print(dir(toolbox))
    raise

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase1_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def expand_path(path):
    return str(Path(os.path.expanduser(path)).expanduser())

LAYOUTS_DIR = Path(expand_path(config['paths']['layouts_raw']))
FIGURES_DIR = Path(expand_path(config['paths']['figures']))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def create_layout_preview(layout_file, output_file):
    """Create a preview image of a layout"""
    
    try:
        # Load design from JSON
        design = load_design_from_json(str(layout_file))
        
        # Plot the design
        fig, ax = plt.subplots(figsize=(10, 8))
        design.plot(ax=ax)
        
        # Customize
        ax.set_title(f"Layout: {layout_file.stem}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        
        # Save
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error visualizing {layout_file}: {e}")
        return False

def create_sample_grid(num_samples=16):
    """Create a grid of sample layouts"""
    
    # Get all layout files
    layout_files = list(LAYOUTS_DIR.glob("*.json"))
    
    if len(layout_files) == 0:
        print("No layout files found")
        return
    
    # Randomly select samples
    samples = random.sample(layout_files, min(num_samples, len(layout_files)))
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(len(samples))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, layout_file in enumerate(samples):
        try:
            design = load_design_from_json(str(layout_file))
            design.plot(ax=axes[i])
            axes[i].set_title(layout_file.stem, fontsize=8)
            axes[i].grid(True, alpha=0.3)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error\n{layout_file.stem}\n{str(e)[:30]}", 
                        ha='center', va='center')
    
    # Hide empty subplots
    for i in range(len(samples), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    output_file = FIGURES_DIR / 'layout_sample_grid.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample grid saved to: {output_file}")

def main():
    print("="*60)
    print("Layout Visualization")
    print("="*60)
    
    layout_files = list(LAYOUTS_DIR.glob("*.json"))
    print(f"Found {len(layout_files)} layout files")
    
    if len(layout_files) == 0:
        print("No layouts to visualize. Run generate_layouts.py first.")
        return
    
    # Create sample grid
    print("\n📊 Creating sample grid...")
    create_sample_grid(16)
    
    # Create individual previews for first 10 layouts
    print("\n🖼️  Creating individual previews...")
    for i, layout_file in enumerate(layout_files[:10]):
        output_file = FIGURES_DIR / f'preview_{layout_file.stem}.png'
        success = create_layout_preview(layout_file, output_file)
        if success:
            print(f"  ✓ {layout_file.stem}")
    
    print(f"\n✅ Visualizations saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
