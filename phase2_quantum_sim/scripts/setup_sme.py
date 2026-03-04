#!/usr/bin/env python
"""
Step 2.1: Implement Stochastic Master Equation (SME)
Target: Working SME solver with all noise channels
"""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# QuTiP imports
try:
    import qutip as qt
    from qutip import basis, tensor, destroy, qeye, mesolve, mcsolve
    from qutip.ui.progressbar import TextProgressBar
    print(f"✅ QuTiP version: {qt.__version__}")
except ImportError as e:
    print(f"❌ QuTiP import failed: {e}")
    print("Please install QuTiP: pip install qutip")
    sys.exit(1)

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase2_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load Phase 1 hardware parameters
hw_params_file = Path(config['hardware_params']).expanduser()
with open(hw_params_file, 'r') as f:
    hw_params = json.load(f)

print("="*60)
print("STEP 2.1: Stochastic Master Equation Implementation")
print("="*60)

print(f"\n📋 Loaded hardware parameters from Phase 1:")
print(f"  Qubit frequency: {hw_params['qubit']['frequency_ghz']} GHz")
print(f"  Mechanical frequency: {hw_params['mechanical']['frequency_mhz']} MHz")
print(f"  Coupling g0: {hw_params['couplings']['g0_qubit_mech_mhz']} MHz")
print(f"  Qubit T1: {hw_params['losses']['t1_qubit_us']} μs")
print(f"  Qubit T2*: {hw_params['losses']['t2_qubit_us']} μs")
print(f"  Mechanical T1: {hw_params['losses']['t1_mech_us']} μs")

class SME_Solver:
    """Stochastic Master Equation Solver for Q-SYMPHONY"""
    
    def __init__(self, hw_params, config):
        self.hw_params = hw_params
        self.config = config
        
        # Hilbert space dimensions
        self.N_q = config['hilbert']['transmon_levels']
        self.N_m = config['hilbert']['mechanical_levels']
        
        # Convert frequencies to angular frequencies (rad/s)
        self.wq = 2 * np.pi * hw_params['qubit']['frequency_ghz'] * 1e9  # rad/s
        self.wm = 2 * np.pi * hw_params['mechanical']['frequency_mhz'] * 1e6  # rad/s
        self.g0 = 2 * np.pi * hw_params['couplings']['g0_qubit_mech_mhz'] * 1e6  # rad/s
        
        # Convert decay rates to Lindblad rates
        self.T1_q = hw_params['losses']['t1_qubit_us'] * 1e-6  # s
        self.T2_q = hw_params['losses']['t2_qubit_us'] * 1e-6  # s
        self.T1_m = hw_params['losses']['t1_mech_us'] * 1e-6  # s
        
        # Calculate rates
        self.gamma_q = 1.0 / self.T1_q  # qubit relaxation rate
        # Dephasing rate from T2 and T1
        self.gamma_phi = 1.0 / self.T2_q - 0.5 / self.T1_q
        self.gamma_m = 1.0 / self.T1_m  # mechanical relaxation rate
        
        # Thermal occupancy at 20 mK
        T = config['constants']['temperature_mK'] * 1e-3  # K
        hbar = config['constants']['hbar']
        kB = config['constants']['kBoltzmann']
        
        # Bose-Einstein distribution
        self.n_th = 1.0 / (np.exp(hbar * self.wm / (kB * T)) - 1)
        print(f"\n🌡️  Thermal occupancy at {T*1000:.1f} mK: n_th = {self.n_th:.3f}")
        
        # Build operators
        self._build_operators()
        
        # Build Hamiltonian
        self._build_hamiltonian()
        
        # Build collapse operators
        self._build_collapse_operators()
        
        # Measurement parameters
        self.kappa = 2 * np.pi * config['measurement']['kappa_MHz'] * 1e6  # rad/s
        self.eta = config['measurement']['efficiency']
        
        print(f"\n🔬 Measurement:")
        print(f"  Cavity linewidth κ/2π: {config['measurement']['kappa_MHz']} MHz")
        print(f"  Detection efficiency η: {self.eta}")
    
    def _build_operators(self):
        """Build basic operators"""
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
    
    def _build_hamiltonian(self):
        """Build system Hamiltonian"""
        # H = ℏω_q a†a + ℏω_m b†b + ℏg0 (a†b + a b†)
        H_q = self.wq * self.n_q
        H_m = self.wm * self.n_m
        H_int = self.g0 * (self.a_dag * self.b + self.a * self.b_dag)
        
        self.H = H_q + H_m + H_int
        
        print(f"\n🎯 Hamiltonian built:")
        print(f"  H_q/ℏ = {self.wq/2/np.pi/1e9:.3f} GHz")
        print(f"  H_m/ℏ = {self.wm/2/np.pi/1e6:.1f} MHz")
        print(f"  H_int/ℏ = {self.g0/2/np.pi/1e6:.2f} MHz")
    
    def _build_collapse_operators(self):
        """Build Lindblad collapse operators"""
        self.c_ops = []
        
        # Qubit energy relaxation
        c_q_relax = np.sqrt(self.gamma_q) * self.a
        self.c_ops.append(c_q_relax)
        
        # Qubit dephasing
        c_q_dephase = np.sqrt(2 * self.gamma_phi) * self.n_q
        self.c_ops.append(c_q_dephase)
        
        # Mechanical relaxation (emission)
        c_m_emit = np.sqrt(self.gamma_m * (self.n_th + 1)) * self.b
        self.c_ops.append(c_m_emit)
        
        # Mechanical excitation (absorption)
        c_m_absorb = np.sqrt(self.gamma_m * self.n_th) * self.b_dag
        self.c_ops.append(c_m_absorb)
        
        print(f"\n💫 Collapse operators:")
        print(f"  Qubit relaxation: γ_q = {self.gamma_q:.2e} s⁻¹")
        print(f"  Qubit dephasing: γ_φ = {self.gamma_phi:.2e} s⁻¹")
        print(f"  Mechanical emission: γ_m↓ = {self.gamma_m*(self.n_th+1):.2e} s⁻¹")
        print(f"  Mechanical absorption: γ_m↑ = {self.gamma_m*self.n_th:.2e} s⁻¹")
    
    def initial_state(self):
        """Create initial thermal state for mechanical mode"""
        # Qubit starts in ground state
        rho_q = basis(self.N_q, 0) * basis(self.N_q, 0).dag()
        
        # Mechanical mode in thermal state
        rho_m = qt.thermal_dm(self.N_m, self.n_th)
        
        # Combined state
        rho0 = tensor(rho_q, rho_m)
        
        return rho0
    
    def run_single_trajectory(self, tlist=None, store_states=True):
        """Run a single SME trajectory"""
        if tlist is None:
            # Create time list
            dt = self.config['simulation']['time_step_ns'] * 1e-9  # s
            T = self.config['simulation']['time_total_us'] * 1e-6  # s
            tlist = np.arange(0, T + dt, dt)
        
        # Initial state
        rho0 = self.initial_state()
        
        # Measurement operator for homodyne detection
        # L = √κ a
        L = np.sqrt(self.kappa) * self.a
        
        print(f"\n🚀 Running single trajectory...")
        print(f"  Time steps: {len(tlist)}")
        print(f"  Total time: {T*1e6:.1f} μs")
        print(f"  Time step: {dt*1e9:.1f} ns")
        
        try:
            # Run stochastic master equation
            result = qt.smesolve(
                self.H, rho0, tlist, 
                c_ops=self.c_ops,
                sc_ops=[L],  # measured operator
                e_ops=[self.n_q, self.n_m, self.a_dag * self.b + self.a * self.b_dag],
                ntraj=1,
                store_states=store_states,
                progress_bar=True,
                options={'store_measurement': True, 'method': 'euler'}
            )
            
            print(f"✅ Trajectory completed successfully")
            
            return result
            
        except Exception as e:
            print(f"❌ SME solver failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def verify_physical(self, result):
        """Verify physical behavior of trajectory"""
        checks = {}
        
        # Check trace conservation
        if result.states:
            traces = [rho.tr() for rho in result.states]
            max_trace_dev = np.max(np.abs(np.array(traces) - 1))
            checks['trace_conservation'] = max_trace_dev < 1e-6
            print(f"  Max trace deviation: {max_trace_dev:.2e}")
        
        # Check positivity (eigenvalues)
        if result.states:
            rho_final = result.states[-1]
            evals = rho_final.eigenenergies()
            min_eval = np.min(evals)
            checks['positivity'] = min_eval > -1e-10
            print(f"  Minimum eigenvalue: {min_eval:.2e}")
        
        # Check population decay
        if result.expect:
            n_q = np.array(result.expect[0])
            n_m = np.array(result.expect[1])
            
            # Fit decay
            t = result.times * 1e6  # μs
            
            # Qubit decay should be exponential
            n_q0 = n_q[0]
            n_q_theory = n_q0 * np.exp(-t / (self.T1_q * 1e6))
            
            # Calculate MSE
            mse = np.mean((n_q - n_q_theory)**2)
            checks['decay_match'] = mse < 0.01
            
            print(f"  Qubit population decay MSE: {mse:.2e}")
        
        return checks
    
    def plot_verification(self, result, save_path=None):
        """Create verification plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        t_us = result.times * 1e6  # Convert to μs
        
        # Plot 1: Qubit population
        if result.expect:
            axes[0, 0].plot(t_us, result.expect[0], 'b-', label='Simulation')
            # Theoretical decay
            n_q0 = result.expect[0][0]
            theory = n_q0 * np.exp(-t_us / (self.T1_q * 1e6))
            axes[0, 0].plot(t_us, theory, 'r--', label=f'Theory T₁={self.T1_q*1e6:.1f}μs')
            axes[0, 0].set_xlabel('Time (μs)')
            axes[0, 0].set_ylabel('⟨n_q⟩')
            axes[0, 0].set_title('Qubit Population Decay')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Mechanical population
        if result.expect and len(result.expect) > 1:
            axes[0, 1].plot(t_us, result.expect[1], 'g-', label='Simulation')
            # Thermal equilibrium
            axes[0, 1].axhline(y=self.n_th, color='r', linestyle='--', 
                               label=f'n_th={self.n_th:.3f}')
            axes[0, 1].set_xlabel('Time (μs)')
            axes[0, 1].set_ylabel('⟨n_m⟩')
            axes[0, 1].set_title('Mechanical Mode Thermalization')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Correlation
        if result.expect and len(result.expect) > 2:
            axes[0, 2].plot(t_us, result.expect[2], 'm-')
            axes[0, 2].set_xlabel('Time (μs)')
            axes[0, 2].set_ylabel('⟨a†b + a b†⟩')
            axes[0, 2].set_title('Qubit-Mechanical Correlation')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Photocurrent (if available)
        if hasattr(result, 'measurement') and result.measurement:
            axes[1, 0].plot(t_us[1:], result.measurement[0].real, 'k-', alpha=0.7)
            axes[1, 0].set_xlabel('Time (μs)')
            axes[1, 0].set_ylabel('I(t)')
            axes[1, 0].set_title('Homodyne Photocurrent')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Trace conservation
        if result.states:
            traces = [rho.tr() for rho in result.states]
            axes[1, 1].plot(t_us, np.array(traces)-1, 'r-')
            axes[1, 1].set_xlabel('Time (μs)')
            axes[1, 1].set_ylabel('Tr(ρ) - 1')
            axes[1, 1].set_title('Trace Conservation')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([-1e-6, 1e-6])
        
        # Plot 6: Purity
        if result.states:
            purities = [rho.purity() for rho in result.states]
            axes[1, 2].plot(t_us, purities, 'b-')
            axes[1, 2].set_xlabel('Time (μs)')
            axes[1, 2].set_ylabel('Tr(ρ²)')
            axes[1, 2].set_title('Purity Decay')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('SME Solver Verification - Single Trajectory', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Verification plot saved to: {save_path}")
        
        plt.show()
        
        return fig

def main():
    """Main execution for Step 2.1"""
    
    print("\n🚀 Initializing SME Solver...")
    solver = SME_Solver(hw_params, config)
    
    # Run single trajectory
    result = solver.run_single_trajectory(store_states=True)
    
    if result is None:
        print("\n❌ SME solver failed - check errors above")
        return
    
    # Verify physical behavior
    print("\n🔍 Verifying physical behavior:")
    checks = solver.verify_physical(result)
    
    all_passed = all(checks.values())
    if all_passed:
        print("\n✅ All physical checks passed!")
    else:
        print("\n⚠️ Some checks failed:")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
    
    # Save verification plot
    save_dir = Path(config['paths']['validation']).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / 'sme_verification.png'
    solver.plot_verification(result, save_path=plot_path)
    
    # Save solver state for next steps
    import pickle
    solver_path = save_dir / 'sme_solver.pkl'
    with open(solver_path, 'wb') as f:
        pickle.dump(solver, f)
    print(f"✅ Solver saved to: {solver_path}")
    
    # Summary
    print("\n" + "="*60)
    print("STEP 2.1 SUMMARY")
    print("="*60)
    print(f"SME Solver: {'✅ VALIDATED' if all_passed else '⚠️ NEEDS WORK'}")
    print(f"Verification plot: {plot_path}")
    print(f"Solver object: {solver_path}")
    print("\nNext: Step 2.2 - Generate 1000 Baseline Trajectories")
    print("="*60)

if __name__ == "__main__":
    main()
