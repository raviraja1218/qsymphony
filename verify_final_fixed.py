#!/usr/bin/env python3
"""
Final verification for Q-SYMPHONY - Fixed version
Tests each package individually with proper error handling
"""

import importlib
import sys
from datetime import datetime

def check_pkg(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        # Special handling for problematic imports
        if name == 'qiskit_metal':
            # Qiskit Metal might have Qt issues, but we only need to know it's installed
            import qiskit_metal
            return True, getattr(qiskit_metal, '__version__', 'installed')
        else:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'installed')
            return True, version
    except ImportError as e:
        return False, str(e).split('\n')[0]  # First line only
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"

def main():
    print("="*70)
    print(f" FINAL FIXED VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"\n✅ CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   PyTorch: {torch.__version__}")
        else:
            print("\n❌ CUDA not available")
    except:
        print("\n❌ PyTorch not installed")
    
    # Check NumPy version (critical)
    try:
        import numpy
        print(f"\n✅ NumPy: {numpy.__version__} (should be 1.x)")
        if numpy.__version__.startswith('2.'):
            print("   ⚠️  WARNING: NumPy 2.x may cause issues!")
    except:
        print("\n❌ NumPy not installed")
    
    # Package groups
    groups = {
        "CORE": ['numpy', 'scipy', 'matplotlib', 'pandas', 'h5py', 'pyyaml'],
        "QUANTUM": ['qiskit', 'qiskit_metal', 'qiskit_dynamics', 'qiskit_experiments', 
                    'qiskit_optimization', 'qiskit_machine_learning', 'qiskit_nature', 'qutip'],
        "PyEPR": ['pymongo', 'PySpice', 'gdspy', 'pyEPR'],
        "DEEP LEARNING": ['torch', 'torch_geometric', 'tensorflow'],
        "RL": ['stable_baselines3', 'gymnasium'],
        "PHYSICS": ['deepxde', 'sympy'],
        "VISUALIZATION": ['plotly', 'seaborn'],
        "UTILITIES": ['sklearn', 'optuna', 'wandb', 'dask', 'zarr'],
        "JUPYTER": ['jupyter', 'ipykernel']
    }
    
    all_passed = True
    failed = []
    
    for group_name, packages in groups.items():
        print(f"\n📦 {group_name}")
        print("-"*50)
        for pkg in packages:
            ok, info = check_pkg(pkg)
            if ok:
                print(f"✅ {pkg:20} {info}")
            else:
                print(f"❌ {pkg:20} {info}")
                all_passed = False
                failed.append(pkg)
    
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
        print("⚠️ Some packages failed:")
        for f in failed:
            print(f"   • {f}")
        print("\nRun the fix commands below:")
    print("="*70)

if __name__ == "__main__":
    main()
