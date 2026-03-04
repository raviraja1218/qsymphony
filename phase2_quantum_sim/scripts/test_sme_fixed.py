#!/usr/bin/env python
"""Quick test of SME solver without full visualization"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import from the fixed script
from setup_sme_fixed import SME_Solver, hw_params, config

print("="*60)
print("Quick SME Solver Test")
print("="*60)

# Initialize solver
solver = SME_Solver(hw_params, config)

# Run short trajectory (10 μs)
dt = config['simulation']['time_step_ns'] * 1e-9
tlist = np.arange(0, 10e-6, dt)

result = solver.run_single_trajectory(tlist=tlist, store_states=False)

if result:
    print("\n✅ SME solver works!")
    
    # Quick checks
    if result.expect:
        n_q_final = result.expect[0][-1]
        n_m_final = result.expect[1][-1]
        print(f"  Final ⟨n_q⟩: {n_q_final:.4f}")
        print(f"  Final ⟨n_m⟩: {n_m_final:.4f}")
        print(f"  Thermal n_th: {solver.n_th:.4f}")
    
    print("\n✅ Ready for Step 2.2")
else:
    print("\n❌ SME solver failed")

print("="*60)
