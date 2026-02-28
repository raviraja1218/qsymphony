#!/usr/bin/env python3
"""
Final verification for Q-SYMPHONY - Handles all edge cases
"""

import importlib
import sys
from datetime import datetime

def safe_import(module_name):
    """Safely import module and return version"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__'):
            return True, module.__version__
        elif module_name == 'sklearn':
            return True, module.__version__
        else:
            return True, 'installed'
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"

def check_pkg_resources():
    """Special check for pkg_resources"""
    try:
        import pkg_resources
        return True, f"setuptools {pkg_resources.__version__ if hasattr(pkg_resources, '__version__') else 'installed'}"
    except ImportError:
        return False, "pkg_resources not available"

def main():
    print("="*70)
    print(f" Q-SYMPHONY FINAL VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check CUDA
    print("\n🔍 CUDA CHECK")
    print("-"*50)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"   PyTorch Version: {torch.__version__}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("❌ CUDA not available")
    except:
        print("❌ PyTorch not installed")
    
    # Check pkg_resources (critical)
    print("\n📦 PKG_RESOURCES CHECK")
    print("-"*50)
    ok, info = check_pkg_resources()
    if ok:
        print(f"✅ {info}")
    else:
        print(f"❌ {info}")
    
    # Package categories
    packages = {
        'Core': [
            ('numpy', 'numpy'),
            ('scipy', 'scipy'),
            ('matplotlib', 'matplotlib'),
            ('pandas', 'pandas'),
            ('jupyter', 'jupyter'),
            ('ipykernel', 'ipykernel'),
            ('tqdm', 'tqdm'),
            ('h5py', 'h5py'),
            ('yaml', 'yaml')
        ],
        'Quantum': [
            ('qutip', 'qutip'),
            ('qiskit', 'qiskit'),
            ('qiskit_metal', 'qiskit_metal'),
            ('qiskit_dynamics', 'qiskit_dynamics'),
            ('qiskit_experiments', 'qiskit_experiments'),
            ('qiskit_optimization', 'qiskit_optimization'),
            ('qiskit_machine_learning', 'qiskit_machine_learning'),
            ('qiskit_nature', 'qiskit_nature')
        ],
        'PyEPR': [
            ('pymongo', 'pymongo'),
            ('PySpice', 'PySpice'),
            ('gdspy', 'gdspy'),
            ('pyEPR', 'pyEPR')
        ],
        'Deep Learning': [
            ('torch', 'torch'),
            ('torch_geometric', 'torch_geometric'),
            ('tensorflow', 'tensorflow')
        ],
        'RL': [
            ('stable_baselines3', 'stable_baselines3'),
            ('gymnasium', 'gymnasium')
        ],
        'Visualization': [
            ('plotly', 'plotly'),
            ('seaborn', 'seaborn')
        ],
        'Utilities': [
            ('sklearn', 'sklearn'),
            ('optuna', 'optuna'),
            ('wandb', 'wandb')
        ],
        'Physics': [
            ('deepxde', 'deepxde'),
            ('sympy', 'sympy')
        ],
        'Data': [
            ('dask', 'dask'),
            ('zarr', 'zarr')
        ]
    }
    
    all_passed = True
    failed_packages = []
    
    for category, pkg_list in packages.items():
        print(f"\n📦 {category} PACKAGES")
        print("-"*50)
        for display_name, module_name in pkg_list:
            ok, info = safe_import(module_name)
            if ok:
                print(f"✅ {display_name:25} {info}")
            else:
                print(f"❌ {display_name:25} {info}")
                all_passed = False
                failed_packages.append(display_name)
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("🎉 SUCCESS! ALL PACKAGES INSTALLED CORRECTLY")
        print("   Phase 0 COMPLETE - Ready for Phase 1")
    else:
        print("⚠️  PARTIAL SUCCESS - Some packages failed")
        print("\nFailed packages:")
        for pkg in failed_packages:
            print(f"   • {pkg}")
        print("\nRun the fix commands below:")
    print("="*70)
    
    if not all_passed:
        print("\n🔧 FIX COMMANDS:")
        if 'qiskit_experiments' in failed_packages:
            print("   pip install --upgrade qiskit-experiments")
        if 'wandb' in failed_packages:
            print("   pip install --upgrade wandb")
        if 'jupyter' in failed_packages or 'ipykernel' in failed_packages:
            print("   pip install jupyter ipykernel")
        if any(pkg in failed_packages for pkg in ['qiskit_experiments', 'wandb']):
            print("   pip install --upgrade setuptools wheel")

if __name__ == "__main__":
    main()
