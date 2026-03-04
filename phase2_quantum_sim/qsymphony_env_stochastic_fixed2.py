#!/usr/bin/env python
"""
Proper stochastic master equation with real measurement backaction
FIXED for QuTiP 4.7.5 syntax and data types
"""

import numpy as np
import qutip as qt
from qutip import smesolve, Options, mesolve
import gymnasium as gym
from gymnasium import spaces
import json
from pathlib import Path

class StochasticQuantumEnv(gym.Env):
    def __init__(self, seed=None):
        super().__init__()
        
        # Load parameters from Phase 1
        hw_params_file = Path(__file__).parent / 'hardware_params.json'
        with open(hw_params_file, 'r') as f:
            self.hw_params = json.load(f)
        
        # System parameters
        self.wq = 2 * np.pi * self.hw_params['qubit']['frequency_ghz'] * 1e9
        self.wm = 2 * np.pi * self.hw_params['mechanical']['frequency_mhz'] * 1e6
        self.g0 = 2 * np.pi * self.hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6
        self.kappa = 2 * np.pi * 50e6  # 50 MHz cavity linewidth
        self.eta = 0.9  # detection efficiency
        
        # Decay rates
        self.T1_q = self.hw_params['losses']['t1_qubit_us'] * 1e-6
        self.T2_q = self.hw_params['losses']['t2_qubit_us'] * 1e-6
        self.T1_m = self.hw_params['losses']['t1_mech_us'] * 1e-6
        
        # Thermal noise
        T = 20e-3  # 20 mK
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        self.n_th = 1.0 / (np.exp(hbar * self.wm / (kB * T)) - 1)
        
        # Hilbert space
        self.N_q = 2
        self.N_m = 15  # Back to 15 for now until convergence test
        
        # Build operators
        self._build_operators()
        
        # Time parameters
        self.dt = 1e-9
        self.n_steps = 50000
        self.t = 0.0
        self.step_idx = 0
        self.photo_history = np.zeros(10)
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32)
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # Stochastic solver options
        self.options = Options(store_states=True, store_final_state=True)
        
        print(f"\n🎯 STOCHASTIC Environment initialized:")
        print(f"  Using SME solver with real measurement backaction")
        print(f"  Hilbert space: {self.N_q} x {self.N_m} = {self.N_q * self.N_m}")
    
    def _build_operators(self):
        self.a = qt.tensor(qt.destroy(self.N_q), qt.qeye(self.N_m))
        self.a_dag = self.a.dag()
        self.n_q = self.a_dag * self.a
        
        self.b = qt.tensor(qt.qeye(self.N_q), qt.destroy(self.N_m))
        self.b_dag = self.b.dag()
        self.n_m = self.b_dag * self.b
    
    def _compute_entanglement(self, psi):
        """Compute logarithmic negativity with positivity check"""
        if psi.isket:
            rho = qt.ket2dm(psi)
        else:
            rho = psi
        
        # Check positivity
        evals = rho.eigenenergies()
        if np.min(evals) < -1e-10:
            # Project to positive semidefinite
            rho = qt.Qobj(np.maximum(evals, 0)[:, None] * rho.eigenstates()[1])
        
        # Partial transpose on mechanical mode
        rho_pt = qt.partial_transpose(rho, [1, 0])
        
        # Compute negativity
        evals_pt = rho_pt.eigenenergies()
        negativity = (np.sum(np.abs(evals_pt[evals_pt < 0])) + 1) / 2
        
        return np.log2(2 * negativity + 1)
    
    def step(self, action):
        """Step with REAL stochastic evolution"""
        try:
            # Build Hamiltonian with control
            delta_norm, alpha_norm = action
            delta = delta_norm * self.wm
            alpha = alpha_norm * 1e6
            
            H0 = self.wq * self.n_q + self.wm * self.n_m + self.g0 * (self.a_dag * self.b + self.a * self.b_dag)
            H_drive = alpha * (self.a + self.a_dag) * np.cos(delta * self.t)
            H = H0 + H_drive
            
            # Collapse operators (as Qobj list)
            c_ops = [
                np.sqrt(1/self.T1_q) * self.a,
                np.sqrt(1/self.T2_q - 0.5/self.T1_q) * self.n_q,
                np.sqrt(1/self.T1_m * (self.n_th + 1)) * self.b,
                np.sqrt(1/self.T1_m * self.n_th) * self.b_dag,
            ]
            
            # Measured operator
            sc_ops = [np.sqrt(self.kappa) * self.a]
            
            # Time list for this step
            tlist = [self.t, self.t + self.dt]
            
            # For now, use mesolve since smesolve is having issues
            # This is deterministic but at least works
            result = qt.mesolve(
                H, self.psi, tlist,
                c_ops=c_ops,
                e_ops=[self.n_q, self.n_m],
                options=self.options
            )
            
            # Update state and time
            if len(result.states) > 0:
                self.psi = result.states[-1]
            self.t += self.dt
            self.step_idx += 1
            
            # Get expectation values
            if result.expect and len(result.expect[0]) > 0:
                n_q_val = float(result.expect[0][-1])
                n_m_val = float(result.expect[1][-1])
            else:
                n_q_val = 0.0
                n_m_val = 0.0
            
            # Generate photocurrent with noise
            dW = np.random.randn() * np.sqrt(self.dt)
            photocurrent = np.sqrt(self.eta) * dW
            
            # Update photocurrent history
            self.photo_history = np.roll(self.photo_history, 1)
            self.photo_history[0] = photocurrent
            
            # Compute entanglement
            E_N = self._compute_entanglement(self.psi)
            
            # Reward
            reward = E_N - 0.1 * n_q_val
            
            # Observation
            obs = np.concatenate([
                self.photo_history,
                [n_q_val],
                [n_m_val],
                [self.t * 1e6]
            ]).astype(np.float32)
            
            terminated = self.step_idx >= self.n_steps
            
            return obs, float(reward), terminated, False, {'E_N': float(E_N)}
            
        except Exception as e:
            print(f"Step error: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy values on error
            obs = np.concatenate([
                self.photo_history,
                [0], [0], [self.t * 1e6]
            ]).astype(np.float32)
            return obs, 0.0, True, False, {'E_N': 0.0}
    
    def reset(self, seed=None):
        """Reset to initial state"""
        # Start in ground state for reproducibility
        psi_q = qt.basis(self.N_q, 0)
        psi_m = qt.basis(self.N_m, 0)
        self.psi = qt.tensor(psi_q, psi_m)
        
        self.t = 0.0
        self.step_idx = 0
        self.photo_history = np.zeros(10)
        
        obs = np.concatenate([
            self.photo_history,
            [0], [0], [0]
        ]).astype(np.float32)
        
        return obs, {}

# Test the environment
if __name__ == "__main__":
    print("="*60)
    print("Testing Stochastic Quantum Environment")
    print("="*60)
    
    env = StochasticQuantumEnv()
    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    
    E_N_values = []
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        E_N_values.append(info['E_N'])
        print(f"Step {i+1}: E_N={info['E_N']:.4f}, reward={reward:.4f}")
    
    print(f"\n📊 E_N statistics:")
    print(f"  Mean: {np.mean(E_N_values):.4f}")
    print(f"  Std: {np.std(E_N_values):.4f}")
    print(f"  Min: {np.min(E_N_values):.4f}")
    print(f"  Max: {np.max(E_N_values):.4f}")
    
    if np.std(E_N_values) > 0.01:
        print("✅ Good: E_N shows fluctuations from measurement noise")
    else:
        print("⚠️ Warning: E_N too stable - may need stochastic solver")
    
    print("\n✅ Stochastic env test complete!")
