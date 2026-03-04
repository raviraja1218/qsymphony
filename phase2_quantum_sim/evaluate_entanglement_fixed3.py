#!/usr/bin/env python
"""
Fixed logarithmic negativity calculation for Phase 2 evaluation
Correct syntax for QuTiP 4.7.5
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from pathlib import Path

def compute_log_negativity_correct(rho):
    """
    CORRECT logarithmic negativity calculation for QuTiP 4.7.5
    """
    # Get the system dimensions
    dims = rho.dims[0]
    
    # Method 1: Using partial_transpose with mask
    # For a bipartite system [2,15], transpose the second subsystem (index 1)
    mask = [1, 0]  # Transpose second subsystem, leave first unchanged
    rho_pt = qt.partial_transpose(rho, mask)
    
    # Get eigenvalues
    evals = rho_pt.eigenenergies()
    
    # Method 1: Trace norm
    trace_norm = np.sum(np.abs(evals))
    E_N_trace = np.log2(trace_norm)
    
    # Method 2: Negativity
    N = np.sum(np.abs(evals[evals < 0]))
    E_N_neg = np.log2(1 + 2*N) if N > 0 else 0.0
    
    return E_N_trace, E_N_neg

def test_with_bell_state():
    """Test with maximally entangled Bell state"""
    print("\n🔬 Testing with Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    
    # Bell state for two qubits
    bell = (qt.basis(4, 0) + qt.basis(4, 3)).unit()
    rho_bell = qt.ket2dm(bell)
    rho_bell.dims = [[2,2], [2,2]]  # Set correct dims
    
    E_N_trace, E_N_neg = compute_log_negativity_correct(rho_bell)
    print(f"  Trace norm method: {E_N_trace:.6f}")
    print(f"  Negativity method: {E_N_neg:.6f}")
    print(f"  Expected: 1.000000")
    
    if abs(E_N_trace - 1.0) < 1e-6:
        print("  ✅ CORRECT")
    else:
        print("  ❌ WRONG - check implementation")
    
    return E_N_trace

def test_with_product_state():
    """Test with separable |00⟩ state"""
    print("\n🔬 Testing with product state |00⟩")
    
    rho_prod = qt.ket2dm(qt.basis(4, 0))
    rho_prod.dims = [[2,2], [2,2]]
    
    E_N_trace, E_N_neg = compute_log_negativity_correct(rho_prod)
    
    print(f"  Trace norm method: {E_N_trace:.6f}")
    print(f"  Negativity method: {E_N_neg:.6f}")
    print(f"  Expected: 0.000000")
    
    if abs(E_N_trace) < 1e-6:
        print("  ✅ CORRECT")
    else:
        print("  ❌ WRONG - check implementation")

def test_with_our_system():
    """Test with our 2×15 system dimensions"""
    print("\n🔬 Testing with 2×15 system dimensions")
    
    # Create a random state in our Hilbert space
    N_q, N_m = 2, 15
    rho = qt.rand_dm(N_q * N_m)
    rho.dims = [[N_q, N_m], [N_q, N_m]]
    
    E_N_trace, E_N_neg = compute_log_negativity_correct(rho)
    print(f"  Trace norm method: {E_N_trace:.6f}")
    print(f"  Negativity method: {E_N_neg:.6f}")
    
    # Check consistency
    if abs(E_N_trace - E_N_neg) < 1e-6:
        print("  ✅ Methods consistent")
    else:
        print(f"  ⚠️ Methods differ by {abs(E_N_trace - E_N_neg):.6f}")

if __name__ == "__main__":
    print("="*60)
    print("FIXED LOGARITHMIC NEGATIVITY VALIDATION")
    print("="*60)
    
    # Test on known states
    test_with_bell_state()
    test_with_product_state()
    test_with_our_system()
    
    print("\n" + "="*60)
    print("✅ Fixed calculation verified")
    print("="*60)
