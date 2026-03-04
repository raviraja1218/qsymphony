#!/usr/bin/env python
"""
Step 2.5: RL Environment Interface for Phase 3
Gym-compatible environment for reinforcement learning control
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pickle
import json
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# QuTiP imports
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mesolve, mcsolve
    from qutip.solver import Options as SolverOptions
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    print("Please install QuTiP: pip install qutip")
    raise

# Load hardware parameters
hw_params_file = Path(__file__).parent / 'hardware_params.json'
if not hw_params_file.exists():
    raise FileNotFoundError(f"Hardware parameters not found at {hw_params_file}")

with open(hw_params_file, 'r') as f:
    hw_params = json.load(f)

class QSymphonyEnv(gym.Env):
    """
    Quantum Symphony Environment for Reinforcement Learning
    
    Action Space:
        - Laser detuning Δ ∈ [-2ω_m, 2ω_m] (MHz)
        - Drive amplitude α_L ∈ [0, 10^6] (photons/s)
    
    Observation Space:
        - Last 10 photocurrent samples
        - Filtered photon number ⟨n_q⟩
        - Filtered phonon number ⟨n_m⟩
        - Time since start (μs)
    
    Reward:
        - Logarithmic negativity E_N (entanglement measure)
        - Penalty for photon number: R = E_N - λ⟨n_q⟩
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, render_mode=None, seed=None, time_total_us=50.0):
        super().__init__()
        
        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        # Parameters from Phase 1
        self.wq = 2 * np.pi * hw_params['qubit']['frequency_ghz'] * 1e9  # rad/s
        self.wm = 2 * np.pi * hw_params['mechanical']['frequency_mhz'] * 1e6  # rad/s
        self.g0 = 2 * np.pi * hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6  # rad/s
        
        # Decay rates
        self.T1_q = hw_params['losses']['t1_qubit_us'] * 1e-6  # s
        self.T2_q = hw_params['losses']['t2_qubit_us'] * 1e-6  # s
        self.T1_m = hw_params['losses']['t1_mech_us'] * 1e-6  # s
        
        self.gamma_q = 1.0 / self.T1_q
        self.gamma_phi = 1.0 / self.T2_q - 0.5 / self.T1_q
        self.gamma_m = 1.0 / self.T1_m
        
        # Thermal occupancy
        T = 20e-3  # 20 mK
        hbar = 1.0545718e-34
        kB = 1.380649e-23
        self.n_th = 1.0 / (np.exp(hbar * self.wm / (kB * T)) - 1)
        
        # Measurement parameters
        self.kappa = 2 * np.pi * 50.0 * 1e6  # 50 MHz linewidth
        self.eta = 0.9  # detection efficiency
        
        # Hilbert space dimensions
        self.N_q = 2  # transmon levels
        self.N_m = 15  # mechanical levels
        
        # Build operators
        self._build_operators()
        
        # Time parameters
        self.dt = 1e-9  # 1 ns time step
        self.time_total = time_total_us * 1e-6  # convert to seconds
        self.n_steps = int(self.time_total / self.dt)
        
        # Action space: [Δ, α_L]
        # Δ in units of ω_m, α_L in 10^6 photons/s
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0]),
            high=np.array([2.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: 13 dimensions
        # [10 photocurrent samples, n_q, n_m, t]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # History buffer for photocurrent
        self.photo_history = np.zeros(10)
        
        # Current state
        self.psi = None
        self.t = 0
        self.step_idx = 0
        self.n_q_current = 0
        self.n_m_current = 0
        
        # Reward parameter
        self.lambda_penalty = 0.1
        
        # For logging
        self.trajectory_data = []
        
        print(f"\n🎯 RL Environment initialized:")
        print(f"  ω_q/2π = {hw_params['qubit']['frequency_ghz']} GHz")
        print(f"  ω_m/2π = {hw_params['mechanical']['frequency_mhz']} MHz")
        print(f"  g₀/2π = {hw_params['couplings']['g0_qubit_mech_mhz']} MHz")
        print(f"  n_th = {self.n_th:.3f}")
        print(f"  Action space: Δ/ω_m ∈ [-2,2], α_L ∈ [0,1e6]")
        print(f"  Observation space: {self.observation_space.shape[0]} dim")
        print(f"  Episode steps: {self.n_steps}")
    
    def _build_operators(self):
        """Build quantum operators"""
        # Qubit operators
        self.a = tensor(destroy(self.N_q), qeye(self.N_m))
        self.a_dag = tensor(destroy(self.N_q).dag(), qeye(self.N_m))
        self.n_q = self.a_dag * self.a
        
        # Mechanical operators
        self.b = tensor(qeye(self.N_q), destroy(self.N_m))
        self.b_dag = tensor(qeye(self.N_q), destroy(self.N_m).dag())
        self.n_m = self.b_dag * self.b
        
        # Identity
        self.identity = tensor(qeye(self.N_q), qeye(self.N_m))
    
    def _build_hamiltonian(self, action):
        """Build Hamiltonian with current control parameters"""
        # Parse action
        delta_norm, alpha_norm = action
        
        # Convert to physical units
        delta = delta_norm * self.wm  # detuning in rad/s
        alpha = alpha_norm * 1e6  # drive amplitude in photons/s
        
        # Base Hamiltonian
        H0 = self.wq * self.n_q + self.wm * self.n_m + self.g0 * (self.a_dag * self.b + self.a * self.b_dag)
        
        # Drive Hamiltonian (simplified - will be refined in Phase 3)
        H_drive = alpha * (self.a + self.a_dag) * np.cos(delta * self.t)
        
        return H0 + H_drive
    
    def _get_collapse_operators(self):
        """Get Lindblad collapse operators"""
        c_ops = []
        c_ops.append(np.sqrt(self.gamma_q) * self.a)                    # Qubit relaxation
        c_ops.append(np.sqrt(2 * self.gamma_phi) * self.n_q)            # Qubit dephasing
        c_ops.append(np.sqrt(self.gamma_m * (self.n_th + 1)) * self.b)  # Mechanical emission
        c_ops.append(np.sqrt(self.gamma_m * self.n_th) * self.b_dag)    # Mechanical absorption
        return c_ops
    
    def _compute_entanglement(self, psi):
        """Compute logarithmic negativity E_N (entanglement measure)"""
        # Simplified for now - will be implemented fully in Phase 3
        # For a product state |0,0⟩, E_N = 0
        # For a Bell state, E_N = log 2 ≈ 0.693
        
        # Convert state vector to density matrix if needed
        if psi.isket:
            rho = qt.ket2dm(psi)
        else:
            rho = psi
        
        # Partial transpose of mechanical mode
        rho_pt = qt.partial_transpose(rho, [1, 0], method='dense')
        
        # Compute negativity
        evals = rho_pt.eigenenergies()
        negativity = (np.sum(np.abs(evals[evals < 0])) + 1) / 2
        
        return np.log2(2 * negativity + 1)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initial state: qubit ground, mechanical ground
        psi_q = basis(self.N_q, 0)
        psi_m = basis(self.N_m, 0)
        self.psi = tensor(psi_q, psi_m)
        
        # Reset time
        self.t = 0
        self.step_idx = 0
        self.photo_history = np.zeros(10)
        self.n_q_current = 0
        self.n_m_current = 0
        
        # Clear trajectory data
        self.trajectory_data = []
        
        # Get initial observation
        obs = self._get_obs()
        
        return obs, {}
    
    def _get_obs(self):
        """Construct observation vector"""
        obs = np.concatenate([
            self.photo_history,
            [self.n_q_current],
            [self.n_m_current],
            [self.t * 1e6]  # time in μs
        ]).astype(np.float32)
        return obs
    
    def step(self, action):
        """Take a step in the environment"""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Build Hamiltonian for this step
        H = self._build_hamiltonian(action)
        
        # Get collapse operators
        c_ops = self._get_collapse_operators()
        
        # Measurement operator (for photocurrent)
        L = np.sqrt(self.kappa) * self.a
        
        # Time list for this step (just two points: now and now+dt)
        tlist = [self.t, self.t + self.dt]
        
        try:
            # Evolve state for one time step
            # Using deterministic evolution for speed (will add stochastic in Phase 3)
            result = qt.mesolve(
                H, self.psi, tlist,
                c_ops=c_ops,
                e_ops=[self.n_q, self.n_m],
                progress_bar=False
            )
            
            # Update state
            self.psi = result.states[-1]
            self.t += self.dt
            self.step_idx += 1
            
            # Update observables
            self.n_q_current = result.expect[0][-1]
            self.n_m_current = result.expect[1][-1]
            
            # Generate photocurrent (simulated)
            dW = np.random.randn() * np.sqrt(self.dt)
            photocurrent = np.sqrt(self.eta) * dW
            
            # Update history
            self.photo_history = np.roll(self.photo_history, -1)
            self.photo_history[-1] = photocurrent
            
            # Compute reward
            E_N = self._compute_entanglement(self.psi)
            reward = E_N - self.lambda_penalty * self.n_q_current
            
            # Check if episode is done
            terminated = self.step_idx >= self.n_steps
            truncated = False
            
            # Get observation
            obs = self._get_obs()
            
            # Log data
            self.trajectory_data.append({
                'step': self.step_idx,
                'time_us': self.t * 1e6,
                'action': action.tolist(),
                'n_q': float(self.n_q_current),
                'n_m': float(self.n_m_current),
                'photocurrent': float(photocurrent),
                'entanglement': float(E_N),
                'reward': float(reward)
            })
            
            return obs, reward, terminated, truncated, {}
            
        except Exception as e:
            print(f"❌ Step failed: {e}")
            # Return zero reward and terminate
            return self._get_obs(), 0.0, True, False, {}
    
    def render(self):
        """Render the environment state"""
        print(f"\nStep {self.step_idx}: t = {self.t*1e6:.2f} μs")
        print(f"  ⟨n_q⟩ = {self.n_q_current:.4f}")
        print(f"  ⟨n_m⟩ = {self.n_m_current:.4f}")
        print(f"  E_N = {self._compute_entanglement(self.psi):.4f}")
    
    def close(self):
        """Clean up"""
        pass
    
    def save_trajectory(self, filename):
        """Save trajectory data to file"""
        with open(filename, 'w') as f:
            json.dump(self.trajectory_data, f, indent=2)
        print(f"✅ Trajectory saved to {filename}")

# Simple test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing QSymphony Environment")
    print("="*60)
    
    # Create environment
    env = QSymphonyEnv(seed=42)
    
    # Test reset
    obs, _ = env.reset()
    print(f"\n✅ Reset successful")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test random actions
    print("\n🎲 Testing random actions...")
    n_test_steps = 10
    total_reward = 0
    
    for i in range(n_test_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: action={action}, reward={reward:.4f}")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ {i+1} steps completed successfully")
    print(f"Total reward: {total_reward:.4f}")
    
    # Test render
    env.render()
    
    # Save test trajectory
    env.save_trajectory("test_trajectory.json")
    
    print("\n" + "="*60)
    print("✅ Environment test complete - Ready for Phase 3!")
    print("="*60)
