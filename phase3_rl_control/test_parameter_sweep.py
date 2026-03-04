#!/usr/bin/env python
"""
Test how entanglement depends on key parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

def run_sweep():
    """Systematic parameter sweep"""
    
    # Parameter ranges
    kappa_MHz = np.linspace(10, 100, 10)
    T1_us = np.linspace(20, 200, 10)
    n_th = np.linspace(0, 1, 10)
    
    # Store results
    results = {
        'kappa': [],
        'T1': [],
        'n_th': [],
        'E_N': []
    }
    
    # Sweep over parameters
    for kappa, T1, nth in tqdm(product(kappa_MHz, T1_us, n_th), 
                                total=len(kappa_MHz)*len(T1_us)*len(n_th)):
        
        # Run simulation with these parameters
        E_N = simulate_with_params(kappa, T1, nth)
        
        results['kappa'].append(kappa)
        results['T1'].append(T1)
        results['n_th'].append(nth)
        results['E_N'].append(E_N)
    
    # Convert to arrays for plotting
    for key in results:
        results[key] = np.array(results[key])
    
    # Create 2D heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # E_N vs kappa and T1 (fixed n_th=0.1)
    mask = results['n_th'] == 0.1
    X = results['kappa'][mask].reshape(10, 10)
    Y = results['T1'][mask].reshape(10, 10)
    Z = results['E_N'][mask].reshape(10, 10)
    
    im = axes[0].pcolormesh(X, Y, Z, shading='auto')
    axes[0].set_xlabel('κ (MHz)')
    axes[0].set_ylabel('T₁ (μs)')
    axes[0].set_title('E_N (n_th=0.1)')
    plt.colorbar(im, ax=axes[0])
    
    # E_N vs kappa and n_th (fixed T1=85)
    mask = results['T1'] == 85
    X = results['kappa'][mask].reshape(10, 10)
    Y = results['n_th'][mask].reshape(10, 10)
    Z = results['E_N'][mask].reshape(10, 10)
    
    im = axes[1].pcolormesh(X, Y, Z, shading='auto')
    axes[1].set_xlabel('κ (MHz)')
    axes[1].set_ylabel('n_th')
    axes[1].set_title('E_N (T₁=85μs)')
    plt.colorbar(im, ax=axes[1])
    
    # E_N vs T1 and n_th (fixed kappa=50)
    mask = results['kappa'] == 50
    X = results['T1'][mask].reshape(10, 10)
    Y = results['n_th'][mask].reshape(10, 10)
    Z = results['E_N'][mask].reshape(10, 10)
    
    im = axes[2].pcolormesh(X, Y, Z, shading='auto')
    axes[2].set_xlabel('T₁ (μs)')
    axes[2].set_ylabel('n_th')
    axes[2].set_title('E_N (κ=50MHz)')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_results.png', dpi=150)
    
    return results

def simulate_with_params(kappa, T1, n_th):
    """Simulate one parameter combination"""
    # This would call your actual simulation
    # For now, return synthetic data
    return 0.5 * np.exp(-kappa/100) * np.exp(-T1/200) * np.exp(-n_th)

if __name__ == "__main__":
    results = run_sweep()
    print("\nSweep complete! Check parameter_sweep_results.png")
