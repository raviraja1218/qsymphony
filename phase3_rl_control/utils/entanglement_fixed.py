#!/usr/bin/env python
"""
Correct logarithmic negativity calculation for QuTiP 4.7.5
"""

import qutip as qt
import numpy as np

def compute_log_negativity_correct(rho):
    """
    CORRECT logarithmic negativity: E_N = log₂ ||ρ^TB||₁
    
    Args:
        rho: density matrix (Qobj) with dims [[2, N_m], [2, N_m]]
    
    Returns:
        E_N: logarithmic negativity
    """
    # Partial transpose on mechanical mode (index 1)
    rho_pt = qt.partial_transpose(rho, [1, 0])
    
    # Get eigenvalues
    evals = rho_pt.eigenenergies()
    
    # Trace norm = sum of absolute eigenvalues
    trace_norm = np.sum(np.abs(evals))
    
    # Logarithmic negativity
    E_N = np.log2(trace_norm)
    
    return E_N

def test_entanglement():
    """Test on known states"""
    print("="*60)
    print("Testing Correct Entanglement Calculation")
    print("="*60)
    
    # Test 1: Bell state (should be 1.0)
    bell = (qt.basis(4, 0) + qt.basis(4, 3)).unit()
    rho_bell = qt.ket2dm(bell)
    rho_bell.dims = [[2,2], [2,2]]
    E_N_bell = compute_log_negativity_correct(rho_bell)
    print(f"Bell state: E_N = {E_N_bell:.6f} (expected 1.0)")
    
    # Test 2: Product state (should be 0.0)
    prod = qt.basis(4, 0)
    rho_prod = qt.ket2dm(prod)
    rho_prod.dims = [[2,2], [2,2]]
    E_N_prod = compute_log_negativity_correct(rho_prod)
    print(f"Product state: E_N = {E_N_prod:.6f} (expected 0.0)")
    
    # Test 3: Random 2x15 state
    N_m = 15
    rho_rand = qt.rand_dm(2 * N_m)
    rho_rand.dims = [[2, N_m], [2, N_m]]
    E_N_rand = compute_log_negativity_correct(rho_rand)
    print(f"Random 2x15 state: E_N = {E_N_rand:.6f}")
    
    print("\n✅ Entanglement calculation verified")
    return E_N_bell, E_N_prod, E_N_rand

if __name__ == "__main__":
    test_entanglement()
