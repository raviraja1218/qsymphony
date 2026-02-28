#!/usr/bin/env python3
"""
Comprehensive verification for Q-SYMPHONY
Tests each package individually and reports status
"""

import importlib
import subprocess
import sys
from datetime import datetime

PACKAGES = {
    'Core': [
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'jupyter',
        'ipykernel',
        'tqdm',
        'h5py',
        'yaml'
    ],
    'Quantum': [
        'qutip',
        'qiskit',
        'qiskit_metal',
        'qiskit_dynamics',
        'qiskit_experiments',
        'qiskit_optimization',
        'qiskit_machine_learning',
        'qiskit_nature'
    ],
    'PyEPR': [
        'pymongo',
        'PySpice',
        'gdspy',
        'pyEPR'
    ],
    'Deep Learning': [
        'torch',
        'torch_geometric',
        'tensorflow'
    ],
    'RL': [
        'stable_baselines3',
        'gymnasium'
    ],
    'Visualization': [
        'plotly',
        'seaborn'
    ],
    'Utilities': [
        'sklearn',
        'optuna',
        'wandb'
    ],
    'Physics': [
        'deepxde',
        'sympy'
    ],
    'Data': [
        'dask',
        'zarr'
    ]
}

def check_package(module_name):
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = 'installed (no version)'
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"

def check_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"GPU: {gpu_name}, CUDA: {cuda_version}"
        else:
            return False, "CUDA not available"
    except:
        return False, "PyTorch not installed"

def main():
    print("="*60)
    print(f" Q-SYMPHONY Final Verification - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check CUDA first
    print("\n🔍 CUDA CHECK")
    print("-"*40)
    cuda_ok, cuda_info = check_cuda()
    if cuda_ok:
        print(f"✅ {cuda_info}")
    else:
        print(f"❌ {cuda_info}")
    
    # Check all packages
    all_passed = True
    for category, packages in PACKAGES.items():
        print(f"\n📦 {category} PACKAGES")
        print("-"*40)
        for pkg in packages:
            ok, info = check_package(pkg)
            if ok:
                print(f"✅ {pkg:25} {info}")
            else:
                print(f"❌ {pkg:25} {info}")
                all_passed = False
    
    # Check if pyEPR specifically worked
    print("\n🔧 SPECIAL CHECKS")
    print("-"*40)
    
    # Try alternate import for pyEPR
    try:
        import pyEPR
        print(f"✅ pyEPR (as pyEPR) {pyEPR.__version__ if hasattr(pyEPR, '__version__') else ''}")
    except:
        try:
            import pyepr
            print(f"✅ pyepr (lowercase) works")
        except:
            print("❌ pyEPR still not importable")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY")
        print("   Phase 0 COMPLETE - Ready for Phase 1")
    else:
        print("❌ SOME PACKAGES FAILED")
        print("   See above for missing packages")
    print("="*60)

if __name__ == "__main__":
    main()
