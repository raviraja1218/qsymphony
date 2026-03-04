#!/usr/bin/env python
"""Simple verification of Q-SYMPHONY packages (no pkg_resources needed)"""

import importlib
import sys
import os
import subprocess

# Define critical packages we absolutely need
CRITICAL_PACKAGES = [
    'numpy',
    'scipy', 
    'matplotlib',
    'torch',
    'tensorflow',
    'qiskit',
    'qiskit_metal',
    'qutip',
    'gym',
    'stable_baselines3',
    'torch_geometric',
]

# Define important packages for each phase
PHASE1_PACKAGES = ['qiskit_metal', 'torch_geometric', 'networkx']
PHASE2_PACKAGES = ['qutip', 'scipy', 'numpy']
PHASE3_PACKAGES = ['gym', 'stable_baselines3', 'torch']
PHASE4_PACKAGES = ['tensorflow', 'sklearn', 'cvxpy']

def check_package(package_name):
    """Simple check if package can be imported"""
    try:
        module = importlib.import_module(package_name)
        # Try to get version
        version = None
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        return True, version
    except ImportError:
        return False, None
    except Exception as e:
        return False, str(e)

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            return True, torch.cuda.get_device_name(0)
        return False, "No CUDA device"
    except:
        return False, "PyTorch not installed"

def main():
    print("\n" + "="*60)
    print("Q-SYMPHONY SIMPLE VERIFICATION")
    print("="*60)
    
    # System info
    print(f"\n📋 System:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Path: {sys.executable}")
    
    # CUDA check
    cuda_ok, cuda_info = check_cuda()
    print(f"\n🎮 CUDA: {'✅' if cuda_ok else '❌'} {cuda_info}")
    
    # Check critical packages
    print(f"\n📦 CRITICAL PACKAGES:")
    all_critical_ok = True
    for pkg in CRITICAL_PACKAGES:
        ok, version = check_package(pkg)
        status = f"✅ v{version}" if ok and version else "✅" if ok else f"❌"
        if not ok:
            all_critical_ok = False
        print(f"  {pkg:<20} {status}")
    
    # Check phase-specific packages
    print(f"\n📦 PHASE 1 (Hardware Design):")
    for pkg in PHASE1_PACKAGES:
        ok, version = check_package(pkg)
        status = f"✅ v{version}" if ok and version else "✅" if ok else f"❌"
        print(f"  {pkg:<20} {status}")
    
    print(f"\n📦 PHASE 2 (Quantum Simulation):")
    for pkg in PHASE2_PACKAGES:
        ok, version = check_package(pkg)
        status = f"✅ v{version}" if ok and version else "✅" if ok else f"❌"
        print(f"  {pkg:<20} {status}")
    
    print(f"\n📦 PHASE 3 (RL Control):")
    for pkg in PHASE3_PACKAGES:
        ok, version = check_package(pkg)
        status = f"✅ v{version}" if ok and version else "✅" if ok else f"❌"
        print(f"  {pkg:<20} {status}")
    
    print(f"\n📦 PHASE 4 (Error Mitigation):")
    for pkg in PHASE4_PACKAGES:
        ok, version = check_package(pkg)
        status = f"✅ v{version}" if ok and version else "✅" if ok else f"❌"
        print(f"  {pkg:<20} {status}")
    
    # Directory check
    print(f"\n📁 Directory Structure:")
    dirs = [
        '~/projects/qsymphony',
        '~/projects/qsymphony/phase1_hardware',
        '~/projects/qsymphony/results',
        '~/Research/Datasets/qsymphony',
    ]
    for d in dirs:
        expanded = os.path.expanduser(d)
        exists = os.path.exists(expanded)
        print(f"  {'✅' if exists else '❌'} {expanded}")
    
    # Summary
    print("\n" + "="*60)
    if all_critical_ok and cuda_ok:
        print("✅✅✅ READY FOR PHASE 1! ✅✅✅")
        print("\nAll critical packages installed and CUDA working.")
    else:
        print("⚠️⚠️⚠️ SOME ISSUES DETECTED ⚠️⚠️⚠️")
        print("\nSee above for missing packages.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
