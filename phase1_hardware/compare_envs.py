#!/usr/bin/env python
"""Compare tnf and qsymphony_hardware environments"""

import subprocess
import sys

def get_packages(env_name):
    """Get package list from conda environment"""
    try:
        result = subprocess.run(
            f"conda list -n {env_name}",
            shell=True, capture_output=True, text=True
        )
        return result.stdout
    except:
        return f"Error getting packages for {env_name}"

def check_qiskit_metal(env_name):
    """Check if qiskit-metal is installed in environment"""
    try:
        result = subprocess.run(
            f"conda run -n {env_name} python -c \"import qiskit_metal; print(qiskit_metal.__version__)\"",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"✅ Qiskit Metal {result.stdout.strip()}"
        else:
            return f"❌ Qiskit Metal not working: {result.stderr[:100]}"
    except:
        return "❌ Error checking"

print("="*60)
print("Environment Comparison")
print("="*60)

envs = ['tnf', 'qsymphony_hardware']

for env in envs:
    print(f"\n📦 Environment: {env}")
    print("-"*40)
    print(check_qiskit_metal(env))
    
    # Check key packages
    key_packages = ['torch', 'tensorflow', 'qutip', 'qiskit']
    for pkg in key_packages:
        try:
            result = subprocess.run(
                f"conda run -n {env} python -c \"import {pkg}; print({pkg}.__version__)\"",
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✅ {pkg}: {result.stdout.strip()}")
            else:
                print(f"  ❌ {pkg}: Not found")
        except:
            print(f"  ❌ {pkg}: Error checking")

print("\n" + "="*60)
print("Which environment should we use?")
print("="*60)
print("""
qsymphony_hardware: Created specifically for hardware design
- Has Qiskit Metal (we verified it works partially)
- Clean environment with minimal packages

tnf: Unknown contents (could have conflicting packages)
- Need to check what's installed
- Might have version conflicts
""")
