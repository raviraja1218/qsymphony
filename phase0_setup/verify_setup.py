#!/usr/bin/env python
"""Verify Q-SYMPHONY setup is complete and working."""

import sys
import subprocess
import importlib

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return f"✓ {module_name}"
    except ImportError as e:
        return f"✗ {module_name}: {e}"

def main():
    print("\n" + "="*50)
    print("Q-SYMPHONY Setup Verification")
    print("="*50 + "\n")

    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check critical packages
    critical_packages = [
        'numpy', 'scipy', 'torch', 'tensorflow',
        'qiskit', 'qutip', 'gym', 'stable_baselines3'
    ]
    
    print("\nCritical Packages:")
    for pkg in critical_packages:
        print(f"  {check_import(pkg)}")
    
    # Check GPU
    print("\nGPU Status:")
    try:
        import torch
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
    except:
        print("  Could not check GPU")
    
    # Check paths
    import os
    print("\nDirectory Structure:")
    paths = [
        '~/projects/qsymphony',
        '~/projects/qsymphony/phase1_hardware',
        '~/projects/qsymphony/results',
        '~/Research/Datasets/qsymphony'
    ]
    for path in paths:
        expanded = os.path.expanduser(path)
        exists = os.path.exists(expanded)
        print(f"  {'✓' if exists else '✗'} {expanded}")
    
    print("\n" + "="*50)
    print("Verification Complete")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
