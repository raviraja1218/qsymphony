#!/usr/bin/env python
"""
Simplified quantum environment for testing - no errors guaranteed
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from pathlib import Path

class QSymphonyEnv(gym.Env):
    """Simplified quantum environment for testing"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None, seed=None, time_total_us=50.0):
        super().__init__()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Load hardware parameters for reference
        hw_params_file = Path(__file__).parent / 'hardware_params.json'
        if hw_params_file.exists():
            with open(hw_params_file, 'r') as f:
                self.hw_params = json.load(f)
        else:
            self.hw_params = {
                'qubit': {'frequency_ghz': 4.753},
                'mechanical': {'frequency_mhz': 492.4},
                'couplings': {'g0_qubit_mech_mhz': 11.19}
            }
        
        # Time parameters
        self.dt = 1e-9
        self.time_total = time_total_us * 1e-6
        self.n_steps = int(self.time_total / self.dt)
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32)
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # State
        self.t = 0.0
        self.step_idx = 0
        self.photo_history = np.zeros(10, dtype=np.float32)
        self.n_q = 0.0
        self.n_m = 0.0
        self.trajectory_data = []
        
        print(f"\n🎯 Simplified RL Environment initialized:")
        print(f"  ω_q/2π = {self.hw_params['qubit']['frequency_ghz']} GHz")
        print(f"  ω_m/2π = {self.hw_params['mechanical']['frequency_mhz']} MHz")
        print(f"  g₀/2π = {self.hw_params['couplings']['g0_qubit_mech_mhz']} MHz")
        print(f"  Episode steps: {self.n_steps}")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.step_idx = 0
        self.photo_history = np.zeros(10, dtype=np.float32)
        self.n_q = 0.0
        self.n_m = 0.0
        self.trajectory_data = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.concatenate([
            self.photo_history,
            [self.n_q],
            [self.n_m],
            [self.t * 1e6]
        ]).astype(np.float32)
        return obs
    
    def step(self, action):
        # Simple dynamics - no quantum simulation
        self.t += self.dt
        self.step_idx += 1
        
        # Simple thermalization model
        self.n_q = 0.1 * np.exp(-self.t / 85e-6)  # Decay with T1
        self.n_m = 0.443 * (1 - np.exp(-self.t / 1200e-6))  # Thermalize to n_th
        
        # Photocurrent
        photocurrent = float(np.random.randn() * 0.1)
        self.photo_history = np.roll(self.photo_history, 1)
        self.photo_history[0] = photocurrent
        
        # Reward
        reward = -0.1 * self.n_q
        
        terminated = self.step_idx >= self.n_steps
        truncated = False
        
        # Log data
        self.trajectory_data.append({
            'step': self.step_idx,
            'time_us': self.t * 1e6,
            'n_q': float(self.n_q),
            'n_m': float(self.n_m),
            'photocurrent': float(photocurrent),
            'reward': float(reward)
        })
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def save_trajectory(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2)
        print(f"✅ Saved {len(self.trajectory_data)} steps to {filename}")

if __name__ == "__main__":
    env = QSymphonyEnv(seed=42)
    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step {i+1}: t={obs[12]:.1f}μs, n_q={obs[10]:.4f}, reward={reward:.4f}")
    
    env.save_trajectory("test_simple.json")
    print("Test complete!")
