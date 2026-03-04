#!/usr/bin/env python
"""
Proper stochastic master equation with real measurement backaction
"""

import numpy as np
import qutip as qt
from qutip import smesolve, Options
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
        self.N_m = 25  # Increased for convergence test
        
        # Build operators
        self._build_operators()
        
        # Time parameters
        self.dt = 1e-9
        self.n_steps = 50000
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0]),
            high=np.array([2.0, 1.0])
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,)
        )
        
        # Stochastic solver options
        self.options = Options(store_states=True, store_measurement=True, method='platen')
        
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
            print(f"⚠️ Warning: Negative eigenvalues detected: {np.min(evals)}")
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
        # Build Hamiltonian with control
        delta_norm, alpha_norm = action
        delta = delta_norm * self.wm
        alpha = alpha_norm * 1e6
        
        H0 = self.wq * self.n_q + self.wm * self.n_m + self.g0 * (self.a_dag * self.b + self.a * self.b_dag)
        H_drive = alpha * (self.a + self.a_dag) * np.cos(delta * self.t)
        H = H0 + H_drive
        
        # Collapse operators
        c_ops = [
            np.sqrt(1/self.T1_q) * self.a,  # qubit relaxation
            np.sqrt(1/self.T2_q - 0.5/self.T1_q) * self.n_q,  # qubit dephasing
            np.sqrt(1/self.T1_m * (self.n_th + 1)) * self.b,  # mechanical emission
            np.sqrt(1/self.T1_m * self.n_th) * self.b_dag,  # mechanical absorption
        ]
        
        # Measured operator
        sc_ops = [np.sqrt(self.kappa) * self.a]
        
        # Time list for this step
        tlist = [self.t, self.t + self.dt]
        
        # Stochastic evolution
        result = qt.smesolve(
            H, self.psi, tlist,
            c_ops=c_ops,
            sc_ops=sc_ops,
            e_ops=[self.n_q, self.n_m],
            ntraj=1,
            options=self.options
        )
        
        # Update state and time
        self.psi = result.states[-1]
        self.t += self.dt
        self.step_idx += 1
        
        # Get REAL photocurrent (not just Gaussian noise)
        photocurrent = result.measurement[0][-1].real
        
        # Update photocurrent history
        self.photo_history = np.roll(self.photo_history, 1)
        self.photo_history[0] = photocurrent
        
        # Compute entanglement
        E_N = self._compute_entanglement(self.psi)
        
        # Reward
        reward = E_N - 0.1 * qt.expect(self.n_q, self.psi)
        
        # Observation
        obs = np.concatenate([
            self.photo_history,
            [qt.expect(self.n_q, self.psi)],
            [qt.expect(self.n_m, self.psi)],
            [self.t * 1e6]
        ])
        
        terminated = self.step_idx >= self.n_steps
        
        return obs, reward, terminated, False, {'E_N': E_N}
    
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
        ])
        
        return obs, {}
