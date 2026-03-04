#!/usr/bin/env python
"""Check what's failing in the step function"""

from qsymphony_env import QSymphonyEnv
import numpy as np
import traceback

env = QSymphonyEnv(seed=42)
obs, _ = env.reset()

print("Testing step function with detailed error catching...")
action = np.array([0.0, 0.5])

try:
    # Let's step through the step function manually
    print("\n1. Clipping action...")
    action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
    print(f"   Clipped action: {action_clipped}")
    
    print("\n2. Building Hamiltonian...")
    H = env._build_hamiltonian(action_clipped)
    print(f"   Hamiltonian built: {H}")
    
    print("\n3. Getting collapse operators...")
    c_ops = env._get_collapse_operators()
    print(f"   Got {len(c_ops)} collapse operators")
    
    print("\n4. Setting up time list...")
    tlist = [env.t, env.t + env.dt]
    print(f"   tlist: {tlist}")
    
    print("\n5. Running mesolve...")
    import qutip as qt
    result = qt.mesolve(
        H, env.psi, tlist,
        c_ops=c_ops,
        e_ops=[env.n_q, env.n_m],
        progress_bar=False
    )
    print(f"✅ mesolve succeeded!")
    
except Exception as e:
    print(f"\n❌ Failed at step: {e}")
    traceback.print_exc()

print("\n" + "="*60)
