#!/usr/bin/env python
"""
Debug environment to see what attributes are available
"""

from utils.environment_wrapper_quantum import QuantumControlEnv

env = QuantumControlEnv(mode='oracle')
obs, _ = env.reset()

print("\n=== ENVIRONMENT DEBUG ===")
print(f"Environment type: {type(env)}")
print(f"Unwrapped type: {type(env.unwrapped)}")
print("\nAttributes in unwrapped:")
for attr in dir(env.unwrapped):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Try to step and see what's in info
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"\nInfo dict keys: {list(info.keys())}")
