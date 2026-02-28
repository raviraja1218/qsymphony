#!/usr/bin/env python3
"""
Clean environment verification for Q-SYMPHONY
Tests each package individually
"""

import importlib
import sys
from datetime import datetime

def check_pkg(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'installed')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("="*70)
    print(f" CLEAN ENVIRONMENT VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check CUDA
    try:
        import torch
        print(f"\n✅ CUDA: {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
    except:
        print("\n❌ CUDA check failed")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('jupyter', 'jupyter'),
        ('ipykernel', 'ipykernel'),
        ('qutip', 'qutip'),
        ('qiskit', 'qiskit'),
        ('qiskit_metal', 'qiskit_metal'),
        ('qiskit_dynamics', 'qiskit_dynamics'),
        ('qiskit_experiments', 'qiskit_experiments'),
        ('qiskit_optimization', 'qiskit_optimization'),
        ('qiskit_machine_learning', 'qiskit_machine_learning'),
        ('qiskit_nature', 'qiskit_nature'),
        ('pyEPR', 'pyEPR'),
        ('torch_geometric', 'torch_geometric'),
        ('tensorflow', 'tensorflow'),
        ('stable_baselines3', 'stable_baselines3'),
        ('gymnasium', 'gymnasium'),
        ('sklearn', 'sklearn'),
        ('deepxde', 'deepxde'),
    ]
    
    print("\n📦 PACKAGE CHECK")
    print("-"*70)
    
    all_passed = True
    for name, import_name in packages:
        ok, info = check_pkg(name, import_name)
        status = "✅" if ok else "❌"
        print(f"{status} {name:25} {info}")
        if not ok:
            all_passed = False
    
    # Check pkg_resources
    try:
        import pkg_resources
        print(f"\n✅ pkg_resources: available")
    except:
        print(f"\n❌ pkg_resources: NOT available")
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 SUCCESS! All packages installed correctly")
        print("Phase 0 COMPLETE - Ready for Phase 1")
    else:
        print("⚠️ Some packages failed - see above")
    print("="*70)

if __name__ == "__main__":
    main()
