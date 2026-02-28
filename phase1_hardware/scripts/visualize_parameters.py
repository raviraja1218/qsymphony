#!/usr/bin/env python
"""
Visualize the parameter space coverage
Creates publication-quality figures
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_visualizations():
    """Generate parameter space visualizations"""
    
    # Load full parameter set
    df = pd.read_csv('../datasets/raw_simulations/layouts/all_parameters.csv')
    
    # Set style for publication
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    parameters = [
        'transmon_width_um',
        'transmon_height_um', 
        'coupling_gap_um',
        'resonator_length_um',
        'junction_area_nm2',
        'substrate_thickness_um'
    ]
    
    titles = [
        'Transmon Width (μm)',
        'Transmon Height (μm)',
        'Coupling Gap (μm)',
        'Resonator Length (μm)',
        'Junction Area (nm²)',
        'Substrate Thickness (μm)'
    ]
    
    for i, (param, title) in enumerate(zip(parameters, titles)):
        ax = axes[i]
        ax.hist(df[param], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(title)
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Distribution\nMean: {df[param].mean():.1f}')
        
        # Add vertical lines for min/max
        ax.axvline(df[param].min(), color='red', linestyle='--', alpha=0.5)
        ax.axvline(df[param].max(), color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/parameter_distributions.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/parameter_distributions.png")
    
    # Create pairplot for correlations
    fig2 = plt.figure(figsize=(12, 10))
    sample_df = df.sample(n=1000)  # Sample for performance
    
    # Correlation matrix
    corr = sample_df[parameters].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Parameter Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/parameter_correlations.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/parameter_correlations.png")
    
    # 3D scatter of three most important parameters
    from mpl_toolkits.mplot3d import Axes3D
    
    fig3 = plt.figure(figsize=(10, 8))
    ax = fig3.add_subplot(111, projection='3d')
    
    # Color by valid/invalid
    colors = df['valid'].map({True: 'green', False: 'red'})
    
    ax.scatter(df['transmon_width_um'][::10], 
              df['transmon_height_um'][::10], 
              df['coupling_gap_um'][::10],
              c=colors[::10], alpha=0.6, s=20)
    
    ax.set_xlabel('Transmon Width (μm)')
    ax.set_ylabel('Transmon Height (μm)')
    ax.set_zlabel('Coupling Gap (μm)')
    ax.set_title('Parameter Space Coverage\n(Green=Valid, Red=Invalid)')
    
    plt.tight_layout()
    plt.savefig('outputs/parameter_space_3d.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: outputs/parameter_space_3d.png")
    
    # Summary statistics
    with open('outputs/parameter_summary.txt', 'w') as f:
        f.write("PARAMETER SPACE SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total layouts: {len(df)}\n")
        f.write(f"Valid layouts: {df['valid'].sum()}\n")
        f.write(f"Invalid layouts: {(~df['valid']).sum()}\n\n")
        
        f.write("Parameter Ranges:\n")
        for param in parameters:
            f.write(f"\n{param}:\n")
            f.write(f"  Min: {df[param].min():.2f}\n")
            f.write(f"  Max: {df[param].max():.2f}\n")
            f.write(f"  Mean: {df[param].mean():.2f}\n")
            f.write(f"  Std: {df[param].std():.2f}\n")
    
    print("✅ Saved: outputs/parameter_summary.txt")

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    create_visualizations()
