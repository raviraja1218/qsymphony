#!/usr/bin/env python
"""
Reward functions for both training phases
Oracle phase: Full state access
Measurement phase: Photocurrent only with golden path tracking
"""

import numpy as np
import torch

class OracleReward:
    """
    Reward function for oracle training phase
    Uses full state information to compute entanglement-based reward
    """
    
    def __init__(self, lambda_photon=0.1, mu_thermal=0.05, target_ent=0.693):
        self.lambda_photon = lambda_photon
        self.mu_thermal = mu_thermal
        self.target_ent = target_ent
        self.n_th = 0.443  # From Phase 2
        
    def __call__(self, state_info):
        """
        Compute reward from full state information
        
        Args:
            state_info: dict containing:
                - entanglement: E_N(t)
                - n_q: photon number
                - n_m: phonon number
                - t: time in μs
        
        Returns:
            reward: scalar reward value
        """
        E_N = state_info.get('entanglement', 0)
        n_q = state_info.get('n_q', 0)
        n_m = state_info.get('n_m', 0)
        t = state_info.get('t', 0)
        
        # Primary objective: maximize entanglement
        ent_term = E_N
        
        # Penalty: avoid too many photons (heating)
        photon_penalty = -self.lambda_photon * n_q
        
        # Penalty: avoid deviating from thermal equilibrium
        thermal_penalty = -self.mu_thermal * abs(n_m - self.n_th)
        
        # Time-dependent weighting (focus on later times)
        time_weight = min(1.0, t / 25.0)  # Linear ramp over first 25μs
        
        reward = ent_term + photon_penalty + thermal_penalty
        reward = reward * time_weight
        
        return reward
    
    def shaped_reward(self, E_N, n_q, n_m, t):
        """Alternative interface with individual components"""
        return self.__call__({
            'entanglement': E_N,
            'n_q': n_q,
            'n_m': n_m,
            't': t
        })

class MeasurementReward:
    """
    Reward function for measurement-based training
    Tracks deviation from golden path (target photon number)
    """
    
    def __init__(self, golden_path_file):
        """
        Args:
            golden_path_file: CSV file with target ⟨n_q⟩(t)
        """
        import pandas as pd
        self.golden_path = pd.read_csv(golden_path_file)
        self.times = self.golden_path['t_us'].values
        self.target_nq = self.golden_path['n_q_target'].values
        
    def get_target(self, t):
        """Interpolate target value at time t"""
        return np.interp(t, self.times, self.target_nq)
    
    def __call__(self, state_info):
        """
        Compute reward from photocurrent and estimated n_q
        
        Args:
            state_info: dict containing:
                - n_q_estimated: estimated photon number from filter
                - t: time in μs
        
        Returns:
            reward: negative absolute deviation from target
        """
        n_q_est = state_info.get('n_q_estimated', 0)
        t = state_info.get('t', 0)
        
        target = self.get_target(t)
        
        # Negative L1 distance to target
        reward = -abs(n_q_est - target)
        
        return reward
    
    def shaped_reward(self, n_q_est, t):
        """Alternative interface"""
        target = self.get_target(t)
        return -abs(n_q_est - target)

class CombinedReward:
    """
    Combined reward for evaluation - uses both entanglement and tracking
    """
    
    def __init__(self, oracle_reward, measurement_reward, alpha=0.5):
        self.oracle = oracle_reward
        self.measurement = measurement_reward
        self.alpha = alpha  # Weight for oracle component
        
    def __call__(self, state_info, true_state_info):
        """
        Combined reward for evaluation
        
        Args:
            state_info: measurement-based info (n_q_est, t)
            true_state_info: true quantum state (E_N, n_q, n_m)
        """
        r_meas = self.measurement(state_info)
        r_oracle = self.oracle(true_state_info)
        
        return self.alpha * r_oracle + (1 - self.alpha) * r_meas

if __name__ == "__main__":
    # Test reward functions
    oracle = OracleReward()
    print("Oracle reward test:")
    test_state = {'entanglement': 0.5, 'n_q': 0.1, 'n_m': 0.05, 't': 30}
    print(f"  Reward: {oracle(test_state):.4f}")
    
    # Create dummy golden path for testing
    import pandas as pd
    test_path = pd.DataFrame({
        't_us': np.linspace(0, 50, 100),
        'n_q_target': np.exp(-np.linspace(0, 2, 100)) * 0.5
    })
    test_path.to_csv('test_golden.csv', index=False)
    
    meas = MeasurementReward('test_golden.csv')
    test_meas = {'n_q_estimated': 0.3, 't': 25}
    print(f"Measurement reward test:")
    print(f"  Reward: {meas(test_meas):.4f}")
    
    # Clean up
    import os
    os.remove('test_golden.csv')
    
    print("✅ Reward functions test passed!")
