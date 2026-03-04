#!/usr/bin/env python
"""
Gymnasium environment wrapper for Phase 2 quantum simulator
Using REAL quantum environment with entanglement
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from pathlib import Path
import sys
import json

# Add Phase 2 directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'phase2_quantum_sim'))

# Import the fixed quantum environment
from qsymphony_env_quantum_fixed import QuantumEnv as QSymphonyEnv

class QuantumControlEnv(gym.Env):
    """
    Wrapper for quantum environment with RL-friendly interface
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, mode='oracle', golden_path_file=None, seed=None):
        super().__init__()
        
        self.mode = mode
        self.seed = seed
        
        # Create underlying quantum environment
        self.quantum_env = QSymphonyEnv(seed=seed)
        
        # Action space: [Δ_norm, α_norm] both in [-1, 1] (will be scaled)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )
        
        # Observation space depends on mode
        if mode == 'oracle':
            # Oracle gets full state info
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
            )
        else:
            # Measurement gets photocurrent history + time
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
            )
        
        # Load golden path if in measurement mode
        self.golden_path = None
        if mode == 'measurement' and golden_path_file:
            import pandas as pd
            self.golden_path = pd.read_csv(golden_path_file)
            self.target_times = self.golden_path['t_us'].values
            self.target_nq = self.golden_path['n_q_target'].values
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 50000
        self.trajectory_data = []
        
    def _get_oracle_obs(self, quantum_obs, state_info):
        """Construct oracle observation from full quantum state"""
        # quantum_obs is [13] from base env
        # Add true entanglement and correlations
        true_state = np.array([
            state_info.get('E_N', 0),
            state_info.get('n_q', 0),
            state_info.get('n_m', 0),
            state_info.get('time', 0)
        ], dtype=np.float32)
        
        return np.concatenate([quantum_obs, true_state])
    
    def _get_measurement_obs(self, quantum_obs):
        """Construct measurement-only observation"""
        return np.concatenate([
            quantum_obs[:10],  # photocurrent history
            quantum_obs[12:13]  # time only
        ])
    
    def _get_target_nq(self, t):
        """Get target photon number from golden path"""
        if self.golden_path is None:
            return 0
        return np.interp(t, self.target_times, self.target_nq)
    
    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            
        quantum_obs, info = self.quantum_env.reset(seed=self.seed)
        
        self.true_state = {
            'E_N': 0.0,
            'n_q': float(quantum_obs[10]),
            'n_m': float(quantum_obs[11]),
            'time': float(quantum_obs[12])
        }
        
        if self.mode == 'oracle':
            obs = self._get_oracle_obs(quantum_obs, self.true_state)
        else:
            obs = self._get_measurement_obs(quantum_obs)
        
        self.current_step = 0
        self.trajectory_data = []
        
        return obs, info
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Scale actions
        delta_norm = float(action[0]) * 2.0
        alpha_norm = (float(action[1]) + 1.0) / 2.0
        scaled_action = np.array([delta_norm, alpha_norm], dtype=np.float32)
        
        # Step quantum environment
        quantum_obs, reward_quantum, terminated, truncated, info = self.quantum_env.step(scaled_action)
        
        # Update true state
        self.true_state.update({
            'E_N': info.get('E_N', 0),
            'n_q': float(quantum_obs[10]),
            'n_m': float(quantum_obs[11]),
            'time': float(quantum_obs[12])
        })
        
        # Compute RL reward based on mode
        if self.mode == 'oracle':
            rl_reward = info.get('E_N', 0) - 0.1 * float(quantum_obs[10])
        else:
            target = self._get_target_nq(float(quantum_obs[12]))
            rl_reward = -abs(float(quantum_obs[10]) - target)
        
        # Build observation
        if self.mode == 'oracle':
            obs = self._get_oracle_obs(quantum_obs, self.true_state)
        else:
            obs = self._get_measurement_obs(quantum_obs)
        
        self.current_step += 1
        terminated = terminated or (self.current_step >= self.max_steps)
        
        # Store trajectory data
        self.trajectory_data.append({
            'step': int(self.current_step),
            'time': float(quantum_obs[12]),
            'action': [float(action[0]), float(action[1])],
            'n_q': float(quantum_obs[10]),
            'n_m': float(quantum_obs[11]),
            'E_N': float(info.get('E_N', 0)),
            'photocurrent': float(quantum_obs[0]),
            'reward': float(rl_reward)
        })
        
        return obs, float(rl_reward), terminated, truncated, info
    
    def save_trajectory(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2)
        print(f"✅ Saved {len(self.trajectory_data)} steps to {filename}")

if __name__ == "__main__":
    print("Testing QuantumControlEnv with quantum environment...")
    
    env = QuantumControlEnv(mode='oracle')
    obs, _ = env.reset()
    print(f"Oracle mode observation shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: E_N={info.get('E_N', 0):.4f}, reward={reward:.4f}")
    
    env.save_trajectory("test_quantum_wrapper.json")
    print("✅ Test complete!")
