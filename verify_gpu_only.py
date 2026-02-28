#!/usr/bin/env python3
"""
Verification for GPU-only packages
Tests each package and reports GPU status
"""

import importlib
import subprocess
import sys
from datetime import datetime

def check_gpu():
    """Detailed GPU check"""
    print("\n🎮 GPU STATUS")
    print("-"*50)
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver.version', '--format=csv,noheader'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ NVIDIA Driver: {result.stdout.strip()}")
        else:
            print("❌ nvidia-smi failed")
    except:
        print("❌ nvidia-smi not available")
    
    # Check PyTorch CUDA
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    except:
        print("❌ PyTorch not installed")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPU Available: {len(gpus) > 0}")
        if gpus:
            print(f"   GPU: {gpus[0].name}")
    except:
        print("❌ TensorFlow not installed")

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'installed')
        return True, version
    except ImportError as e:
        return False, str(e).split('\n')[0]
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"

def main():
    print("="*70)
    print(f" GPU-ENABLED VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # GPU Check
    check_gpu()
    
    # Package groups with GPU relevance
    groups = {
        "CORE": ['numpy', 'scipy', 'matplotlib', 'pandas', 'pyyaml'],
        "QUANTUM": ['qiskit', 'qiskit_metal', 'qiskit_dynamics', 'qiskit_experiments', 
                    'qiskit_optimization', 'qiskit_machine_learning', 'qiskit_nature', 'qutip'],
        "GPU DEEP LEARNING": ['torch', 'torch_geometric', 'tensorflow'],
        "RL": ['stable_baselines3', 'gymnasium'],
        "PHYSICS": ['deepxde', 'sympy'],
        "PyEPR": ['pymongo', 'PySpice', 'gdspy', 'pyEPR'],
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
            ok, info = check_package(pkg)
            if ok:
                print(f"✅ {pkg:25} {info}")
            else:
                print(f"❌ {pkg:25} {info}")
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
        print("🎉 SUCCESS! All GPU-enabled packages installed")
        print("Phase 0 COMPLETE - Ready for Phase 1")
    else:
        print("⚠️ Some packages failed:")
        for f in failed[:10]:  # Show first 10 failures
            print(f"   • {f}")
        if len(failed) > 10:
            print(f"   ... and {len(failed)-10} more")
    print("="*70)

if __name__ == "__main__":
    main()
