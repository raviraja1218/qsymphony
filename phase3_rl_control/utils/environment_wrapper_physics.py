#!/usr/bin/env python
"""
Wrapper for PhysicsQuantumEnv for Phase 3 training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'phase2_quantum_sim'))

from qsymphony_env_physics_clean import PhysicsQuantumEnv

class PhysicsControlEnv(gym.Env):
    """Wrapper for physics environment with RL interface"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, mode='oracle', seed=None):
        super().__init__()
        
        self.mode = mode
        self.seed = seed
        self.quantum_env = PhysicsQuantumEnv(seed=seed)
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32)
        )
        
        # Observation space
        if mode == 'oracle':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )
        
        self.current_step = 0
        self.max_steps = 50000
    
    def reset(self, seed=None):
        obs, info = self.quantum_env.reset()
        
        if self.mode == 'oracle':
            # Add E_N to observation
            e_n = info.get('E_N', 0)
            obs = np.concatenate([obs, [e_n]])
        
        self.current_step = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.quantum_env.step(action)
        
        if self.mode == 'oracle':
            e_n = info.get('E_N', 0)
            obs = np.concatenate([obs, [e_n]])
        
        self.current_step += 1
        terminated = terminated or (self.current_step >= self.max_steps)
        
        # Reward based on E_N
        reward = info.get('E_N', 0) - 0.1 * obs[10]  # E_N - 0.1*n_q
        
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    print("="*60)
    print("Testing Physics Control Environment")
    print("="*60)
    
    env = PhysicsControlEnv(mode='oracle')
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    E_N_values = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        E_N_values.append(info['E_N'])
        print(f"Step {i+1:2d}: E_N = {info['E_N']:.4f}, reward = {reward:.4f}")
    
    print(f"\n📊 Mean E_N = {np.mean(E_N_values):.4f}")
    print("✅ Environment ready!")
