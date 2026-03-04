#!/usr/bin/env python
"""
Gymnasium environment wrapper for Phase 2 quantum simulator - FIXED
Converts numpy types to Python native for JSON serialization
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

try:
    from qsymphony_env_quantum import QuantumEnv as QSymphonyEnv
except ImportError as e:
    print(f"Warning: Could not import QSymphonyEnv: {e}")
    print("Using mock environment for testing")
    QSymphonyEnv = None

class QuantumControlEnv(gym.Env):
    """
    Wrapper for QSymphony environment with RL-friendly interface
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, mode='oracle', golden_path_file=None, seed=None):
        """
        Args:
            mode: 'oracle' (full state) or 'measurement' (photocurrent only)
            golden_path_file: CSV with target photon numbers (for measurement mode)
            seed: random seed
        """
        super().__init__()
        
        self.mode = mode
        self.seed = seed
        
        # Create underlying quantum environment
        if QSymphonyEnv is not None:
            self.quantum_env = QSymphonyEnv(seed=seed)
        else:
            # Mock environment for testing
            self.quantum_env = MockQuantumEnv(seed=seed)
        
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
        self.max_steps = 50000  # 50 μs at 1 ns
        self.trajectory_data = []
        
    def _get_oracle_obs(self, quantum_obs, state_info):
        """
        Construct oracle observation from full quantum state
        Includes: photocurrent history + true quantum numbers
        """
        # quantum_obs is [13] from base env
        # Add true entanglement and correlations
        true_state = np.array([
            state_info.get('entanglement', 0),
            state_info.get('n_q_true', 0),
            state_info.get('n_m_true', 0),
            state_info.get('correlation', 0),
            state_info.get('purity', 1.0),
            state_info.get('time', 0),
            state_info.get('phase', 0)
        ], dtype=np.float32)
        
        return np.concatenate([quantum_obs, true_state])
    
    def _get_measurement_obs(self, quantum_obs):
        """
        Construct measurement-only observation
        Just photocurrent history + time
        """
        # quantum_obs has [10 photocurrent, n_q, n_m, time]
        # Keep only photocurrent and time
        return np.concatenate([
            quantum_obs[:10],  # photocurrent history
            quantum_obs[12:13]  # time only
        ])
    
    def _get_target_nq(self, t):
        """Get target photon number from golden path at time t"""
        if self.golden_path is None:
            return 0
        return np.interp(t, self.target_times, self.target_nq)
    
    def reset(self, seed=None):
        """Reset environment"""
        if seed is not None:
            self.seed = seed
            
        quantum_obs, info = self.quantum_env.reset(seed=self.seed)
        
        # Store true state info for oracle
        self.true_state = {
            'entanglement': 0.0,
            'n_q_true': float(quantum_obs[10]),
            'n_m_true': float(quantum_obs[11]),
            'time': float(quantum_obs[12]),
            'correlation': 0.0,
            'purity': 1.0,
            'phase': 0.0
        }
        
        if self.mode == 'oracle':
            obs = self._get_oracle_obs(quantum_obs, self.true_state)
        else:
            obs = self._get_measurement_obs(quantum_obs)
        
        self.current_step = 0
        self.trajectory_data = []
        
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: [Δ_norm, α_norm] in [-1,1] range
        
        Returns:
            obs: observation
            reward: reward value
            terminated: episode ended
            truncated: episode truncated
            info: additional info
        """
        # Scale actions to physical ranges
        # Δ: [-2ω_m, 2ω_m] -> map [-1,1] to [-2,2] in units of ω_m
        # α: [0, 10^6] -> map [-1,1] to [0,1] in normalized units
        delta_norm = float(action[0]) * 2.0  # [-2, 2] in ω_m units
        alpha_norm = (float(action[1]) + 1.0) / 2.0  # [0, 1] in normalized units
        
        scaled_action = np.array([delta_norm, alpha_norm], dtype=np.float32)
        
        # Step quantum environment
        quantum_obs, reward_quantum, terminated, truncated, info = self.quantum_env.step(scaled_action)
        
        # Update true state info
        self.true_state.update({
            'n_q_true': float(quantum_obs[10]),
            'n_m_true': float(quantum_obs[11]),
            'time': float(quantum_obs[12]),
        })
        
        # Compute RL reward based on mode
        if self.mode == 'oracle':
            # Oracle reward uses true entanglement
            # For now, simplified - will be replaced with proper E_N computation
            rl_reward = -abs(float(quantum_obs[10]))  # Penalize photons
            info['entanglement'] = 0.0
        else:
            # Measurement reward tracks golden path
            target = self._get_target_nq(float(quantum_obs[12]))
            rl_reward = -abs(float(quantum_obs[10]) - target)
            info['target_nq'] = float(target)
        
        # Build observation
        if self.mode == 'oracle':
            obs = self._get_oracle_obs(quantum_obs, self.true_state)
        else:
            obs = self._get_measurement_obs(quantum_obs)
        
        self.current_step += 1
        terminated = terminated or (self.current_step >= self.max_steps)
        
        # Store trajectory data - CONVERT ALL TO PYTHON NATIVE TYPES
        self.trajectory_data.append({
            'step': int(self.current_step),
            'time': float(quantum_obs[12]),
            'action': [float(action[0]), float(action[1])],
            'n_q': float(quantum_obs[10]),
            'n_m': float(quantum_obs[11]),
            'photocurrent': float(quantum_obs[0]),  # latest photocurrent
            'reward': float(rl_reward),
            'entanglement': float(info.get('entanglement', 0.0))
        })
        
        return obs, float(rl_reward), terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment state"""
        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step}: t={self.true_state['time']:.2f}μs")
    
    def close(self):
        """Clean up"""
        pass
    
    def save_trajectory(self, filename):
        """Save trajectory data to file"""
        with open(filename, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2)
        print(f"✅ Saved {len(self.trajectory_data)} steps to {filename}")

class MockQuantumEnv:
    """Mock environment for testing when QSymphonyEnv is not available"""
    
    def __init__(self, seed=None):
        self.seed = seed
        self.t = 0
        self.dt = 1e-9
        self.n_steps = 50000
        
    def reset(self, seed=None):
        self.t = 0
        # Mock observation: [10 photocurrent, n_q, n_m, time]
        return np.zeros(13, dtype=np.float32), {}
    
    def step(self, action):
        self.t += self.dt
        
        # Mock dynamics
        n_q = 0.1 * np.random.randn()
        n_m = 0.05 * np.random.randn()
        photocurrent = np.random.randn(10) * 0.1
        
        obs = np.concatenate([
            photocurrent,
            [n_q],
            [n_m],
            [self.t * 1e6]
        ]).astype(np.float32)
        
        terminated = self.t >= 50e-6
        truncated = False
        
        return obs, 0.0, terminated, truncated, {}

if __name__ == "__main__":
    # Test environment
    print("Testing QuantumControlEnv...")
    
    # Test oracle mode
    env = QuantumControlEnv(mode='oracle')
    obs, _ = env.reset()
    print(f"Oracle mode observation shape: {obs.shape}")
    
    # Test measurement mode
    env = QuantumControlEnv(mode='measurement')
    obs, _ = env.reset()
    print(f"Measurement mode observation shape: {obs.shape}")
    
    # Test stepping
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}")
    
    env.save_trajectory("test_trajectory.json")
    print("✅ Environment test passed!")
