#!/usr/bin/env python
"""
Full quantum environment with real entanglement calculation - FIXED
Uses QuTiP for accurate quantum dynamics with state storage
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
from pathlib import Path
import sys

# QuTiP imports
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mesolve
    from qutip.solver import Options
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    raise

class QuantumEnv(gym.Env):
    """Full quantum environment with real entanglement"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, seed=None, time_total_us=50.0):
        super().__init__()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Load hardware parameters
        hw_params_file = Path(__file__).parent / 'hardware_params.json'
        with open(hw_params_file, 'r') as f:
            self.hw_params = json.load(f)
        
        # System parameters
        self.wq = 2 * np.pi * self.hw_params['qubit']['frequency_ghz'] * 1e9
        self.wm = 2 * np.pi * self.hw_params['mechanical']['frequency_mhz'] * 1e6
        self.g0 = 2 * np.pi * self.hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6
        
        # Decay rates
        self.T1_q = self.hw_params['losses']['t1_qubit_us'] * 1e-6
        self.T2_q = self.hw_params['losses']['t2_qubit_us'] * 1e-6
        self.T1_m = self.hw_params['losses']['t1_mech_us'] * 1e-6
        
        self.gamma_q = 1.0 / self.T1_q
        self.gamma_phi = 1.0 / self.T2_q - 0.5 / self.T1_q
        self.gamma_m = 1.0 / self.T1_m
        
        # Thermal occupancy
        T = 20e-3  # 20 mK
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        self.n_th = 1.0 / (np.exp(hbar * self.wm / (kB * T)) - 1)
        
        # Hilbert space dimensions
        self.N_q = 2
        self.N_m = 15
        
        # Build operators
        self._build_operators()
        
        # Time parameters
        self.dt = 1e-9
        self.time_total = time_total_us * 1e-6
        self.n_steps = int(self.time_total / self.dt)
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32)
        )
        
        # Observation space: [10 photocurrent, n_q, n_m, time]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # Solver options - FIXED: store states
        self.solver_options = Options(store_states=True, store_final_state=True)
        
        # State
        self.psi = None
        self.t = 0.0
        self.step_idx = 0
        self.photo_history = np.zeros(10, dtype=np.float32)
        self.trajectory_data = []
        
        print(f"\n🎯 QUANTUM Environment initialized:")
        print(f"  ω_q/2π = {self.hw_params['qubit']['frequency_ghz']} GHz")
        print(f"  ω_m/2π = {self.hw_params['mechanical']['frequency_mhz']} MHz")
        print(f"  g₀/2π = {self.hw_params['couplings']['g0_qubit_mech_mhz']} MHz")
        print(f"  n_th = {self.n_th:.3f}")
        print(f"  Hilbert space: {self.N_q} x {self.N_m} = {self.N_q*self.N_m}")
        print(f"  Episode steps: {self.n_steps}")
    
    def _build_operators(self):
        """Build quantum operators"""
        self.a = tensor(destroy(self.N_q), qeye(self.N_m))
        self.a_dag = self.a.dag()
        self.n_q = self.a_dag * self.a
        
        self.b = tensor(qeye(self.N_q), destroy(self.N_m))
        self.b_dag = self.b.dag()
        self.n_m = self.b_dag * self.b
        
        # Identity
        self.identity = tensor(qeye(self.N_q), qeye(self.N_m))
    
    def _build_hamiltonian(self, action):
        """Build Hamiltonian with control"""
        delta_norm, alpha_norm = action
        delta = delta_norm * self.wm
        alpha = alpha_norm * 1e6
        
        H0 = self.wq * self.n_q + self.wm * self.n_m + self.g0 * (self.a_dag * self.b + self.a * self.b_dag)
        H_drive = alpha * (self.a + self.a_dag) * np.cos(delta * self.t)
        
        return H0 + H_drive
    
    def _get_collapse_operators(self):
        """Get Lindblad collapse operators"""
        c_ops = []
        c_ops.append(np.sqrt(self.gamma_q) * self.a)
        c_ops.append(np.sqrt(2 * self.gamma_phi) * self.n_q)
        c_ops.append(np.sqrt(self.gamma_m * (self.n_th + 1)) * self.b)
        c_ops.append(np.sqrt(self.gamma_m * self.n_th) * self.b_dag)
        return c_ops
    
    def _compute_entanglement(self, psi):
        """Compute logarithmic negativity E_N"""
        # Convert to density matrix if needed
        if psi.isket:
            rho = qt.ket2dm(psi)
        else:
            rho = psi
        
        # Partial transpose of mechanical mode
        rho_pt = qt.partial_transpose(rho, [1, 0])
        
        # Compute negativity
        evals = rho_pt.eigenenergies()
        negativity = (np.sum(np.abs(evals[evals < 0])) + 1) / 2
        
        return np.log2(2 * negativity + 1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial state: qubit ground, mechanical ground
        psi_q = basis(self.N_q, 0)
        psi_m = basis(self.N_m, 0)
        self.psi = tensor(psi_q, psi_m)
        
        self.t = 0.0
        self.step_idx = 0
        self.photo_history = np.zeros(10, dtype=np.float32)
        self.trajectory_data = []
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Construct observation"""
        obs = np.concatenate([
            self.photo_history,
            [float(qt.expect(self.n_q, self.psi))],
            [float(qt.expect(self.n_m, self.psi))],
            [self.t * 1e6]
        ]).astype(np.float32)
        return obs
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        try:
            H = self._build_hamiltonian(action)
            c_ops = self._get_collapse_operators()
            tlist = [self.t, self.t + self.dt]
            
            # FIXED: Added options to store states
            result = qt.mesolve(
                H, self.psi, tlist,
                c_ops=c_ops,
                e_ops=[self.n_q, self.n_m],
                options=self.solver_options
            )
            
            # FIXED: Check if states were stored
            if len(result.states) > 0:
                self.psi = result.states[-1]
            else:
                # Fallback: just use the last expectation (shouldn't happen with options)
                print("Warning: No states stored, using expectation only")
            
            self.t += self.dt
            self.step_idx += 1
            
            # Get expectation values
            n_q_val = float(result.expect[0][-1]) if result.expect else 0.0
            n_m_val = float(result.expect[1][-1]) if result.expect else 0.0
            
            # Compute entanglement
            E_N = self._compute_entanglement(self.psi)
            
            # Generate photocurrent
            dW = np.random.randn() * np.sqrt(self.dt)
            photocurrent = float(np.sqrt(0.9) * dW)
            
            # Update photocurrent history
            self.photo_history = np.roll(self.photo_history, 1)
            self.photo_history[0] = photocurrent
            
            # Reward: maximize entanglement, minimize photons
            reward = E_N - 0.1 * n_q_val
            
            terminated = self.step_idx >= self.n_steps
            truncated = False
            
            # Log data
            self.trajectory_data.append({
                'step': self.step_idx,
                'time_us': self.t * 1e6,
                'n_q': n_q_val,
                'n_m': n_m_val,
                'E_N': float(E_N),
                'photocurrent': photocurrent,
                'reward': reward,
                'action': action.tolist()
            })
            
            info = {'E_N': float(E_N), 'n_q': n_q_val, 'n_m': n_m_val}
            
            return self._get_obs(), reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Step error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_obs(), 0.0, True, False, {}
    
    def save_trajectory(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2)
        print(f"✅ Saved {len(self.trajectory_data)} steps to {filename}")

if __name__ == "__main__":
    env = QuantumEnv(seed=42)
    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: E_N={info['E_N']:.4f}, reward={reward:.4f}")
    
    env.save_trajectory("test_quantum.json")
    print("Test complete!")
