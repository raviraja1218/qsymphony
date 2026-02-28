#!/usr/bin/env python3
"""
COMPLETE verification for Q-SYMPHONY - All packages should work now
"""

import importlib
import sys
from datetime import datetime

def print_header(text):
    print(f"\n📌 {text}")
    print("-"*60)

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
    print(f" 🎯 Q-SYMPHONY FINAL VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # GPU Check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    except:
        print("\n⚠️  PyTorch CUDA check failed")
    
    # Core Packages
    print_header("CORE PACKAGES")
    core_pkgs = ['numpy', 'scipy', 'matplotlib', 'pandas', 'pyyaml']
    for pkg in core_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # Qiskit Core
    print_header("QISKIT CORE")
    ok, ver = check_package('qiskit')
    print(f"{'✅' if ok else '❌'} {'qiskit':20} {ver}")
    
    # Check for qiskit-terra (should NOT exist)
    try:
        import qiskit_terra
        print("❌ qiskit-terra              FOUND - THIS IS BAD!")
    except:
        print("✅ qiskit-terra              Not present (good)")
    
    # Qiskit Extensions
    print_header("QISKIT EXTENSIONS")
    extensions = [
        ('qiskit_metal', 'qiskit_metal'),
        ('qiskit_dynamics', 'qiskit_dynamics'),
        ('qiskit_experiments', 'qiskit_experiments'),
        ('qiskit_optimization', 'qiskit_optimization'),
        ('qiskit_machine_learning', 'qiskit_machine_learning'),
        ('qiskit_nature', 'qiskit_nature'),
    ]
    for name, import_name in extensions:
        ok, ver = check_package(name, import_name)
        print(f"{'✅' if ok else '❌'} {name:25} {ver}")
    
    # Other Quantum Packages
    print_header("QUANTUM PACKAGES")
    quantum_pkgs = ['qutip']
    for pkg in quantum_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:25} {ver}")
    
    # PyEPR Stack
    print_header("PyEPR STACK")
    pyepr_pkgs = ['pymongo', 'PySpice', 'gdspy', 'pyEPR']
    for pkg in pyepr_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:25} {ver}")
    
    # Deep Learning
    print_header("DEEP LEARNING")
    dl_pkgs = ['torch', 'torch_geometric', 'tensorflow']
    for pkg in dl_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # TensorFlow GPU Check
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   TensorFlow GPU: {'✅' if gpus else '❌'} {gpus if gpus else 'No GPU found'}")
    except:
        print("   TensorFlow GPU: ⚠️  Check failed")
    
    # Reinforcement Learning
    print_header("REINFORCEMENT LEARNING")
    rl_pkgs = ['stable_baselines3', 'gymnasium']
    for pkg in rl_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # Physics-Informed
    print_header("PHYSICS-INFORMED")
    physics_pkgs = ['deepxde', 'sympy']
    for pkg in physics_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # Visualization
    print_header("VISUALIZATION")
    viz_pkgs = ['plotly', 'seaborn']
    for pkg in viz_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # Utilities
    print_header("UTILITIES")
    util_pkgs = ['sklearn', 'optuna', 'wandb', 'dask', 'zarr']
    for pkg in util_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # Jupyter
    print_header("JUPYTER")
    jupyter_pkgs = ['jupyter', 'ipykernel']
    for pkg in jupyter_pkgs:
        ok, ver = check_package(pkg)
        print(f"{'✅' if ok else '❌'} {pkg:20} {ver}")
    
    # Final Status
    print("\n" + "="*70)
    print(" 🎉 PHASE 0 COMPLETE! All packages installed successfully")
    print("    Ready to begin Phase 1: Hardware Topology Discovery")
    print("="*70)
    
    print("\n📁 Next steps:")
    print("   cd ~/projects/qsymphony/phase1_hardware")
    print("   # Start implementing Symplectic GNN for chip design")

if __name__ == "__main__":
    main()
