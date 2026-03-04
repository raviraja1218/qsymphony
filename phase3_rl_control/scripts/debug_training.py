#!/usr/bin/env python
"""
Minimal debug script to test if training can start
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.environment_wrapper_quantum import QuantumControlEnv
from utils.ppo_network import PPOActorCritic
import torch
import time

print("="*60)
print("DEBUG: Testing environment and network")
print("="*60)

# Create environment
print("\n1. Creating environment...")
env = QuantumControlEnv(mode='oracle')
print("✅ Environment created")

# Reset environment
print("\n2. Resetting environment...")
obs, _ = env.reset()
print(f"✅ Reset successful")
print(f"   Observation shape: {obs.shape}")
print(f"   Observation: {obs[:5]}...")

# Create network
print("\n3. Creating network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = PPOActorCritic(
    obs_dim=len(obs),
    action_dim=env.action_space.shape[0],
    hidden_dim=256,
    lstm_dim=128
).to(device)
print(f"✅ Network created with {sum(p.numel() for p in network.parameters()):,} parameters")

# Test forward pass
print("\n4. Testing forward pass...")
obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
with torch.no_grad():
    action_mean, log_std, value, hidden = network(obs_tensor)
print(f"✅ Forward pass successful")
print(f"   Action mean shape: {action_mean.shape}")
print(f"   Value shape: {value.shape}")

# Test action selection
print("\n5. Testing action selection...")
action, log_prob, value, hidden = network.select_action(obs_tensor)
print(f"✅ Action selection successful")
print(f"   Action: {action.cpu().numpy()[0]}")
print(f"   Log prob: {log_prob.item():.4f}")

# Test environment step
print("\n6. Testing environment step...")
action_np = action.cpu().numpy()[0]
obs, reward, terminated, truncated, info = env.step(action_np)
print(f"✅ Step successful")
print(f"   Reward: {reward:.4f}")
print(f"   E_N: {info.get('E_N', 0):.4f}")

print("\n" + "="*60)
print("✅ All tests passed! Training should work.")
print("="*60)
