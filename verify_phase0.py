#!/usr/bin/env python
"""
Phase 0 Verification Script
Run this to confirm everything is set up correctly
"""

import sys
import subprocess
import importlib

def check_package(package_name):
    try:
        pkg = importlib.import_module(package_name)
        version = getattr(pkg, '__version__', 'unknown')
        print(f"✅ {package_name:20} version: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20} NOT FOUND - {str(e)}")
        return False

def main():
    print("="*60)
    print("PHASE 0 VERIFICATION - Project Q-SYMPHONY")
    print("="*60)
    
    # Check Python
    print(f"\n🐍 Python: {sys.version.split()[0]}")
    
    # Check Conda environment
    env = sys.prefix.split('/')[-1]
    print(f"📦 Conda env: {env}")
    
    # Check GPU
    try:
        import torch
        print(f"🎮 PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("🎮 PyTorch CUDA: Failed to import torch")
    
    print("\n📚 Critical Packages:")
    critical = [
        'numpy', 'scipy', 'pandas', 'matplotlib',
        'torch', 'tensorflow', 'qutip', 'qiskit',
        'qiskit_metal', 'stable_baselines3', 'torch_geometric'
    ]
    
    for pkg in critical:
        check_package(pkg)
    
    print("\n📂 Directory Structure:")
    dirs = [
        'phase1_hardware', 'phase2_quantum_sim', 'phase3_rl_control',
        'phase4_error_mitigation', 'phase5_manuscript',
        'results/figures', 'results/tables', 'results/models',
        'datasets'
    ]
    
    import os
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ {d:25} exists")
        else:
            print(f"❌ {d:25} MISSING")
    
    print("\n" + "="*60)
    print("Phase 0 Complete! Ready for Phase 1.")
    print("="*60)

if __name__ == "__main__":
    main()
