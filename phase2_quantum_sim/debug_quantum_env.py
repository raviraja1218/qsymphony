#!/usr/bin/env python
"""Debug the quantum environment to find the index error"""

from qsymphony_env_fixed3 import QSymphonyEnv
import numpy as np
import traceback

print("="*60)
print("DEBUGGING QUANTUM ENVIRONMENT")
print("="*60)

# Create environment
env = QSymphonyEnv(seed=42)
print("\n✅ Environment created")

# Reset
obs, info = env.reset()
print(f"✅ Reset successful")
print(f"  obs shape: {obs.shape}")
print(f"  obs: {obs}")

# Test a single step
print("\n🔄 Testing single step...")
action = np.array([0.0, 0.5])  # Simple test action

try:
    # Step through the step function manually
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
        e_ops=[env.n_q, env.n_m]
    )
    print(f"✅ mesolve succeeded!")
    print(f"   result.expect[0]: {result.expect[0]}")
    print(f"   result.expect[1]: {result.expect[1]}")
    print(f"   result.states: {len(result.states)} states")
    
    print("\n6. Updating state...")
    env.psi = result.states[-1]
    env.t += env.dt
    env.step_idx += 1
    print(f"   t: {env.t*1e6:.2f} μs")
    print(f"   step_idx: {env.step_idx}")
    
    print("\n7. Extracting expectation values...")
    if len(result.expect[0]) > 0:
        n_q_val = float(result.expect[0][-1])
        n_m_val = float(result.expect[1][-1])
        print(f"   n_q: {n_q_val}")
        print(f"   n_m: {n_m_val}")
    else:
        print("❌ result.expect[0] is empty!")
    
    print("\n8. Updating photocurrent history...")
    dW = np.random.randn() * np.sqrt(env.dt)
    photocurrent = float(np.sqrt(0.9) * dW)
    print(f"   photocurrent: {photocurrent}")
    
    env.photo_history = np.roll(env.photo_history, 1)
    env.photo_history[0] = photocurrent
    print(f"   photo_history: {env.photo_history}")
    
    print("\n✅ All steps completed successfully!")
    
except Exception as e:
    print(f"\n❌ Failed at step: {e}")
    traceback.print_exc()

print("\n" + "="*60)
