#!/usr/bin/env python
"""Create Figure 1b: Symplectic GNN Architecture diagram"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_gnn_architecture_figure():
    """Create a diagram of the Symplectic GNN architecture"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#98FB98',      # Light green
        'symp': '#87CEEB',        # Sky blue
        'output': '#FFB6C1',      # Light pink
        'text': 'black'
    }
    
    # Input layer (graph representation)
    ax.add_patch(patches.Rectangle((0.5, 2.5), 1.5, 1.5, 
                                   facecolor=colors['input'], edgecolor='black', alpha=0.7))
    ax.text(1.25, 3.25, 'Input\nGraph', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw small graph inside
    ax.plot([0.8, 1.0, 1.2, 1.4], [2.8, 3.0, 2.7, 3.1], 'ko', markersize=3)
    
    # Hidden layers
    x_positions = [2.5, 4.0, 5.5, 7.0]
    for i, x in enumerate(x_positions):
        ax.add_patch(patches.Rectangle((x, 1.5), 1.0, 3.0, 
                                       facecolor=colors['symp'], edgecolor='black', alpha=0.7))
        ax.text(x+0.5, 3.0, f'Symp\nLayer\n{i+1}', ha='center', va='center', fontsize=10)
    
    # Output layer
    ax.add_patch(patches.Rectangle((8.0, 2.5), 1.5, 1.5, 
                                   facecolor=colors['output'], edgecolor='black', alpha=0.7))
    ax.text(8.75, 3.25, 'Output\nParameters', ha='center', va='center', fontsize=12, weight='bold')
    
    # Arrows between layers
    arrow_props = dict(arrowstyle='->', lw=2, color='gray')
    ax.annotate('', xy=(2.5, 3), xytext=(2.0, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(4.0, 3), xytext=(3.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 3), xytext=(5.0, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(7.0, 3), xytext=(6.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(8.0, 3), xytext=(7.5, 3), arrowprops=arrow_props)
    
    # Symplectic constraint annotation
    ax.text(5, 0.5, 'Symplectic Constraint: $J^T \Omega J = \Omega$', 
            ha='center', fontsize=14, weight='bold', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Title
    ax.set_title('Figure 1b: Symplectic Graph Neural Network Architecture', 
                 fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_file = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'figures' / 'fig1b_sympgnn_arch.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.eps'), format='eps', bbox_inches='tight')
    plt.close()
    
    print(f"✅ Figure 1b saved to: {output_file}")

if __name__ == "__main__":
    from pathlib import Path
    create_gnn_architecture_figure()
