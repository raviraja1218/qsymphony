#!/usr/bin/env python
"""
Generate IQ scatter plots for all qubit variants
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
iq_dir = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'raw_simulations' / 'readout_data' / 'iq_data'
fig_dir = Path('results/phase4/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

qubits = ['q2', 'q3', 'q4', 'q5', 'q6']
titles = ['RQC Q2', 'RQC Q3', 'RQC Q4', 'RQC Q5', 'RQC Q6']

print("="*60)
print("Generating IQ Scatter Plots")
print("="*60)

for qubit, title in zip(qubits, titles):
    # Load data
    file_path = iq_dir / f"{qubit}_iq_perfect.csv"
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        continue
    
    df = pd.read_csv(file_path)
    df0 = df[df['state'] == 0]
    df1 = df[df['state'] == 1]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(df0['I'], df0['Q'], alpha=0.1, s=1, label='|0⟩', c='blue')
    ax.scatter(df1['I'], df1['Q'], alpha=0.1, s=1, label='|1⟩', c='red')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title(f'{title} IQ Scatter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Histogram
    ax = axes[1]
    ax.hist(df0['I'], bins=50, alpha=0.5, label='|0⟩', color='blue', density=True)
    ax.hist(df1['I'], bins=50, alpha=0.5, label='|1⟩', color='red', density=True)
    ax.set_xlabel('I')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} I-projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} Readout Characterization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plot_path = fig_dir / f'{qubit}_iq_scatter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()

print("\n✅ All IQ scatter plots generated!")
