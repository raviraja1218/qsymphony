#!/usr/bin/env python
"""
Quantum environment with CORRECT two-mode squeezing Hamiltonian
H = ω_q a†a + ω_m b†b + g_bs (a†b + a b†) + g_tms (a†b† + a b)
"""

import numpy as np
import qutip as qt
from qutip import Options
import gymnasium as gym
from gymnasium import spaces
import json
from pathlib import Path

class PhysicsQuantumEnv(gym.Env):
    """Quantum environment with correct two-mode squeezing physics"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, seed=None):
        super().__init__()
        
        # Load hardware parameters from Phase 1
        hw_params_file = Path(__file__).parent / 'hardware_params.json'
        with open(hw_params_file, 'r') as f:
            self.hw_params = json.load(f)
        
        # System parameters (from Phase 1)
        self.wq = 2 * np.pi * self.hw_params['qubit']['frequency_ghz'] * 1e9
        self.wm = 2 * np.pi * self.hw_params['mechanical']['frequency_mhz'] * 1e6
        self.g_bs = 2 * np.pi * self.hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6
        
        # Two-mode squeezing parameter (from optimization)
        self.g_tms = 2 * np.pi * 1583.33e6  # 1583 MHz optimal
        
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
        self.N_m = 15
        
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
        
        # State
        self.psi = None
        
        print(f"\n🎯 PHYSICS Environment initialized:")
        print(f"  ω_q/2π = {self.hw_params['qubit']['frequency_ghz']} GHz")
        print(f"  ω_m/2π = {self.hw_params['mechanical']['frequency_mhz']} MHz")
        print(f"  g_bs/2π = {self.g_bs/2/np.pi/1e6:.1f} MHz (beam splitter)")
        print(f"  g_tms/2π = {self.g_tms/2/np.pi/1e6:.1f} MHz (squeezing)")
        print(f"  n_th = {self.n_th:.3f}")
        print(f"  Hilbert space: {self.N_q} x {self.N_m} = {self.N_q*self.N_m}")
    
    def _build_operators(self):
        """Build quantum operators"""
        self.a = qt.tensor(qt.destroy(self.N_q), qt.qeye(self.N_m))
        self.a_dag = self.a.dag()
        self.n_q = self.a_dag * self.a
        
        self.b = qt.tensor(qt.qeye(self.N_q), qt.destroy(self.N_m))
        self.b_dag = self.b.dag()
        self.n_m = self.b_dag * self.b
    
    def _compute_entanglement(self, psi):
        """Correct logarithmic negativity"""
        if psi.isket:
            rho = qt.ket2dm(psi)
        else:
            rho = psi
        
        rho_pt = qt.partial_transpose(rho, [1, 0])
        evals = rho_pt.eigenenergies()
        trace_norm = np.sum(np.abs(evals))
        return np.log2(trace_norm)
    
    def reset(self, seed=None):
        """Reset to initial state |0,0⟩"""
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
    
    def step(self, action):
        """Take a step with correct Hamiltonian"""
        try:
            # Parse action
            delta_norm, alpha_norm = action
            delta = delta_norm * self.wm
            alpha = alpha_norm * 1e6
            
            # Base Hamiltonian with TWO-MODE SQUEEZING
            H0 = (self.wq * self.n_q + self.wm * self.n_m + 
                  self.g_bs * (self.a_dag * self.b + self.a * self.b_dag) +
                  self.g_tms * (self.a_dag * self.b_dag + self.a * self.b))
            
            # Drive term
            H_drive = alpha * (self.a + self.a_dag) * np.cos(delta * self.t)
            H = H0 + H_drive
            
            # Collapse operators
            gamma_q = 1.0 / self.T1_q
            gamma_phi = 1.0 / self.T2_q - 0.5 / self.T1_q
            gamma_m = 1.0 / self.T1_m
            
            c_ops = [
                np.sqrt(gamma_q) * self.a,
                np.sqrt(2 * gamma_phi) * self.n_q,
                np.sqrt(gamma_m * (self.n_th + 1)) * self.b,
                np.sqrt(gamma_m * self.n_th) * self.b_dag,
            ]
            
            # Evolve
            tlist = [self.t, self.t + self.dt]
            options = Options(store_states=True, nsteps=10000)
            
            result = qt.mesolve(H, self.psi, tlist, c_ops=c_ops, 
                               e_ops=[self.n_q, self.n_m], options=options)
            
            # Update state
            if len(result.states) > 0:
                self.psi = result.states[-1]
            
            self.t += self.dt
            self.step_idx += 1
            
            # Get observables
            n_q_val = result.expect[0][-1] if result.expect else 0
            n_m_val = result.expect[1][-1] if result.expect else 0
            
            # Compute entanglement
            E_N = self._compute_entanglement(self.psi)
            
            # Photocurrent (simulated)
            dW = np.random.randn() * np.sqrt(self.dt)
            photocurrent = float(np.sqrt(0.9) * dW)
            
            # Update history
            self.photo_history = np.roll(self.photo_history, 1)
            self.photo_history[0] = photocurrent
            
            # Observation
            obs = np.concatenate([
                self.photo_history,
                [float(n_q_val)],
                [float(n_m_val)],
                [self.t * 1e6]
            ]).astype(np.float32)
            
            terminated = self.step_idx >= self.n_steps
            
            return obs, 0.0, terminated, False, {'E_N': float(E_N)}
            
        except Exception as e:
            print(f"Step error: {e}")
            return self._get_obs(), 0.0, True, False, {'E_N': 0.0}
    
    def _get_obs(self):
        return np.concatenate([
            self.photo_history,
            [0], [0], [self.t * 1e6]
        ]).astype(np.float32)

# Test
if __name__ == "__main__":
    env = PhysicsQuantumEnv()
    obs, _ = env.reset()
    print("\nTesting environment...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: E_N = {info['E_N']:.4f}")
    print("\n✅ Environment ready!")
