#!/usr/bin/env python
"""Debug the quantum environment step by step"""

from qsymphony_env_quantum import QuantumEnv
import numpy as np
import traceback

print("="*60)
print("DEBUGGING QUANTUM ENVIRONMENT")
print("="*60)

# Create environment
env = QuantumEnv(seed=42)
print("\n✅ Environment created")

# Reset
obs, info = env.reset()
print(f"✅ Reset successful")
print(f"  obs shape: {obs.shape}")
print(f"  obs: {obs}")

# Test one step
print("\n🔄 Testing single step...")
action = np.array([0.0, 0.5])

try:
    # Step through manually
    print("\n1. Clipping action...")
    action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
    print(f"   Clipped: {action_clipped}")
    
    print("\n2. Building Hamiltonian...")
    H = env._build_hamiltonian(action_clipped)
    print(f"   Hamiltonian built")
    
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
        e_ops=[env.n_q, env.n_m]
    )
    print(f"✅ mesolve succeeded!")
    print(f"   result.states: {len(result.states)} states")
    print(f"   result.expect[0]: {result.expect[0]}")
    
    if len(result.states) > 0:
        print("\n6. Updating state...")
        env.psi = result.states[-1]
        print(f"   State updated")
        
        print("\n7. Computing entanglement...")
        E_N = env._compute_entanglement(env.psi)
        print(f"   E_N = {E_N:.6f}")
    else:
        print("\n❌ No states in result!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()

print("\n" + "="*60)
