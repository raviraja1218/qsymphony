#!/usr/bin/env python3
"""
Q-SYMPHONY Setup Verification Script
Run this to confirm all components are properly installed
"""

import sys
import platform
import subprocess

def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

print_header("Q-SYMPHONY Environment Verification")
print(f"Python: {platform.python_version()}")
print(f"Platform: {platform.platform()}")

print_header("CUDA & PyTorch")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
except ImportError:
    print("❌ PyTorch not installed")

print_header("Core Scientific Stack")
check_import("numpy")
check_import("scipy")
check_import("matplotlib")
check_import("pandas")

print_header("Quantum Packages")
check_import("qutip")
check_import("qiskit")
check_import("qiskit_metal")
check_import("pyEPR")

print_header("Deep Learning")
check_import("torch_geometric")
check_import("tensorflow")

print_header("Reinforcement Learning")
check_import("stable_baselines3")
check_import("gymnasium")

print_header("Visualization")
check_import("plotly")
check_import("seaborn")

print_header("Utilities")
check_import("sklearn")
check_import("optuna")

print_header("Dataset Directories")
import os
paths = [
    "~/projects/qsymphony",
    "~/Research/Datasets/qsymphony",
    "~/projects/qsymphony/datasets"
]
for path in paths:
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        print(f"✅ {path}")
        if os.path.islink(expanded):
            print(f"   → symlink to {os.readlink(expanded)}")
    else:
        print(f"❌ {path} (not found)")

print_header("Disk Space")
try:
    result = subprocess.run(['df', '-h', os.path.expanduser('~')], 
                          capture_output=True, text=True)
    print(result.stdout)
except:
    pass

print_header("Verification Complete")
print("If all ✅, proceed to Phase 1")
print("If any ❌, check installation")
