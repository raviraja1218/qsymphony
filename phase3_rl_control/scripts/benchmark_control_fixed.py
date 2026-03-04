#!/usr/bin/env python
"""
Step 3.7: Benchmark Against Analytical Control - FIXED imports
Compare RL agent with Kummer-Floquet optimal control
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
from utils.environment_wrapper_fixed import QuantumControlEnv

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase3_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths
figures_dir = Path(config['paths']['figures']).expanduser()
data_dir = Path(config['paths']['data']).expanduser()
traj_dir = Path(config['paths']['trajectories']).expanduser()

# Create directories
figures_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

class KummerFloquetControl:
    """Kummer-Floquet theoretical optimal control"""
    
    def __init__(self):
        # Parameters from system
        self.wm = 2 * np.pi * 492.4e6  # mechanical frequency (rad/s)
        self.g0 = 2 * np.pi * 11.19e6  # coupling (rad/s)
        
        # Floquet parameters (theoretical optimal)
        self.modulation_freq = self.wm
        self.modulation_amplitude = 0.5 * self.g0
        self.detuning_offset = 0
        
    def get_action(self, t):
        """
        Get control action at time t (seconds)
        Returns: [Δ_norm, α_norm]
        """
        # Kummer-Floquet optimal modulation
        delta = self.modulation_amplitude * np.cos(self.modulation_freq * t)
        alpha = 0.5 * self.g0 * np.ones_like(t) if hasattr(t, '__len__') else 0.5 * self.g0
        
        # Normalize to action space
        delta_norm = delta / self.wm  # Δ in units of ω_m
        alpha_norm = alpha / 1e6  # α in units of 10^6 photons/s
        
        return np.array([delta_norm, alpha_norm])

class Benchmark:
    """Compare RL agent with theoretical control"""
    
    def __init__(self):
        self.env = QuantumControlEnv(
            mode='measurement',
            golden_path_file=str(Path(config['paths']['golden_path']).expanduser())
        )
        self.theoretical = KummerFloquetControl()
        
    def run_theoretical_episode(self):
        """Run episode with Kummer-Floquet control"""
        
        obs, _ = self.env.reset()
        done = False
        
        times = []
        actions = []
        observations = []
        rewards = []
        step = 0
        
        while not done:
            t = step * 1e-9  # time in seconds
            action = self.theoretical.get_action(t)
            
            obs_np, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            time_us = step * 1e-3  # μs
            times.append(time_us)
            actions.append(action)
            observations.append(obs_np)
            rewards.append(reward)
            step += 1
        
        return {
            'times': np.array(times),
            'actions': np.array(actions),
            'observations': np.array(observations),
            'rewards': np.array(rewards),
            'total_reward': np.sum(rewards),
            'n_steps': step
        }
    
    def load_rl_episode(self):
        """Load best RL episode from saved trajectory"""
        traj_files = sorted(traj_dir.glob('best_episode_*.json'))
        if not traj_files:
            print("⚠️ No RL trajectory found. Please run evaluation first.")
            return None
        
        latest = traj_files[-1]
        with open(latest, 'r') as f:
            data = json.load(f)
        
        return {
            'times': np.array(data['times_us']),
            'actions': np.array(data['actions']),
            'observations': np.array(data['observations']),
            'rewards': np.array(data['rewards']),
            'total_reward': data['total_reward'],
            'n_steps': data['n_steps']
        }
    
    def compute_metrics(self, episode_data):
        """Compute performance metrics from episode"""
        obs = episode_data['observations']
        
        # Metrics
        avg_photon = np.mean(obs[:, 10])  # average ⟨n_q⟩
        max_photon = np.max(obs[:, 10])
        
        # Entanglement proxy (negative photon number)
        entanglement = -obs[:, 10]
        peak_entanglement = np.max(entanglement)
        avg_entanglement_last = np.mean(entanglement[-1000:])  # last 1 μs
        
        return {
            'avg_photon': float(avg_photon),
            'max_photon': float(max_photon),
            'peak_entanglement': float(peak_entanglement),
            'avg_entanglement_last': float(avg_entanglement_last),
            'total_reward': float(episode_data['total_reward'])
        }
    
    def calculate_improvement(self, rl_metrics, theory_metrics):
        """Calculate improvement percentages"""
        improvements = {}
        
        for metric in rl_metrics:
            if metric in theory_metrics:
                rl_val = rl_metrics[metric]
                theory_val = theory_metrics[metric]
                
                if theory_val != 0:
                    # For metrics where lower is better (photons)
                    if 'photon' in metric:
                        impr = (theory_val - rl_val) / theory_val * 100
                    else:
                        # For metrics where higher is better (entanglement, reward)
                        impr = (rl_val - theory_val) / theory_val * 100
                    
                    improvements[metric] = float(impr)
        
        return improvements
    
    def generate_comparison_plots(self, rl_data, theory_data):
        """Generate comparison plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Photon number comparison
        ax = axes[0, 0]
        ax.plot(rl_data['times'], rl_data['observations'][:, 10], 
                'b-', linewidth=1.5, label='RL Agent')
        ax.plot(theory_data['times'], theory_data['observations'][:, 10], 
                'r--', linewidth=1.5, label='Kummer-Floquet')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('⟨n_q⟩')
        ax.set_title('Photon Number')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Entanglement proxy comparison
        ax = axes[0, 1]
        rl_ent = -rl_data['observations'][:, 10]
        theory_ent = -theory_data['observations'][:, 10]
        ax.plot(rl_data['times'], rl_ent, 'b-', linewidth=1.5, label='RL Agent')
        ax.plot(theory_data['times'], theory_ent, 'r--', linewidth=1.5, label='Kummer-Floquet')
        ax.axhline(y=0.693, color='g', linestyle=':', alpha=0.7, label='log 2 (max)')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('E_N (proxy)')
        ax.set_title('Entanglement Proxy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Control signals comparison - Detuning
        ax = axes[1, 0]
        ax.plot(rl_data['times'][::100], rl_data['actions'][::100, 0], 
                'b.', markersize=2, alpha=0.5, label='RL')
        ax.plot(theory_data['times'][::100], theory_data['actions'][::100, 0], 
                'r-', linewidth=1, label='Theory')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Δ / ω_m')
        ax.set_title('Detuning Control')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Control signals comparison - Amplitude
        ax = axes[1, 1]
        ax.plot(rl_data['times'][::100], rl_data['actions'][::100, 1] * 1e6, 
                'b.', markersize=2, alpha=0.5, label='RL')
        ax.plot(theory_data['times'][::100], theory_data['actions'][::100, 1] * 1e6, 
                'r-', linewidth=1, label='Theory')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('α_L (photons/s)')
        ax.set_title('Drive Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('RL vs Kummer-Floquet Control Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        png_file = figures_dir / 'benchmark_comparison.png'
        eps_file = figures_dir / 'benchmark_comparison.eps'
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(eps_file, format='eps', bbox_inches='tight')
        print(f"✅ Comparison plot saved: {png_file}")
        plt.close()
    
    def generate_report(self, rl_metrics, theory_metrics, improvements):
        """Generate benchmark report"""
        
        report = f"""
============================================================
BENCHMARK COMPARISON: RL vs Kummer-Floquet Control
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================

1. Photon Number Metrics:
   - RL Average: {rl_metrics['avg_photon']:.4f}
   - Theory Average: {theory_metrics['avg_photon']:.4f}
   - Improvement: {improvements.get('avg_photon', 0):+.1f}%
   
   - RL Maximum: {rl_metrics['max_photon']:.4f}
   - Theory Maximum: {theory_metrics['max_photon']:.4f}
   - Improvement: {improvements.get('max_photon', 0):+.1f}%

2. Entanglement Metrics (proxy):
   - RL Peak: {rl_metrics['peak_entanglement']:.4f}
   - Theory Peak: {theory_metrics['peak_entanglement']:.4f}
   - Improvement: {improvements.get('peak_entanglement', 0):+.1f}%
   
   - RL Last 1μs Avg: {rl_metrics['avg_entanglement_last']:.4f}
   - Theory Last 1μs Avg: {theory_metrics['avg_entanglement_last']:.4f}
   - Improvement: {improvements.get('avg_entanglement_last', 0):+.1f}%

3. Total Reward:
   - RL: {rl_metrics['total_reward']:.2f}
   - Theory: {theory_metrics['total_reward']:.2f}
   - Improvement: {improvements.get('total_reward', 0):+.1f}%

============================================================
OVERALL ASSESSMENT:
"""
        
        # Determine if RL outperforms theory
        avg_impr = np.mean(list(improvements.values()))
        if avg_impr > 10:
            report += "✅ RL significantly outperforms theoretical control"
        elif avg_impr > 0:
            report += "✅ RL marginally outperforms theoretical control"
        else:
            report += "⚠️ RL underperforms theoretical control - check implementation"
        
        report += f"\n(Average improvement: {avg_impr:.1f}%)\n"
        report += "============================================================"
        
        return report

def main():
    print("="*60)
    print("STEP 3.7: Benchmark Against Analytical Control")
    print("="*60)
    
    benchmark = Benchmark()
    
    # Run theoretical control
    print("\n📐 Running Kummer-Floquet theoretical control...")
    theory_data = benchmark.run_theoretical_episode()
    print(f"  Episode length: {theory_data['n_steps']} steps")
    print(f"  Total reward: {theory_data['total_reward']:.2f}")
    
    # Load RL results
    print("\n🤖 Loading RL agent results...")
    rl_data = benchmark.load_rl_episode()
    if rl_data is None:
        print("❌ No RL trajectory found. Please run evaluation first.")
        return
    print(f"  Episode length: {rl_data['n_steps']} steps")
    print(f"  Total reward: {rl_data['total_reward']:.2f}")
    
    # Compute metrics
    print("\n📊 Computing performance metrics...")
    rl_metrics = benchmark.compute_metrics(rl_data)
    theory_metrics = benchmark.compute_metrics(theory_data)
    
    # Calculate improvements
    improvements = benchmark.calculate_improvement(rl_metrics, theory_metrics)
    
    # Generate comparison plots
    print("\n🖼️  Generating comparison plots...")
    benchmark.generate_comparison_plots(rl_data, theory_data)
    
    # Generate report
    print("\n📝 Generating benchmark report...")
    report = benchmark.generate_report(rl_metrics, theory_metrics, improvements)
    
    # Save report
    report_file = data_dir / 'benchmark_comparison.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Benchmark report saved to: {report_file}")
    
    # Save metrics as JSON
    metrics_data = {
        'rl': rl_metrics,
        'theory': theory_metrics,
        'improvements': improvements,
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_file = data_dir / 'benchmark_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✅ Metrics saved to: {metrics_file}")
    
    print("\n" + "="*60)
    print("✅ STEP 3.7 COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
