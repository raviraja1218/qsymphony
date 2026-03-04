#!/usr/bin/env python
"""Explore Qiskit Metal structure to find available components"""

import qiskit_metal
import inspect
from pathlib import Path

print("="*60)
print("QISKIT METAL STRUCTURE EXPLORATION")
print("="*60)
print(f"Version: {qiskit_metal.__version__}")
print(f"Location: {qiskit_metal.__file__}")
print()

# Explore top-level attributes
print("TOP-LEVEL ATTRIBUTES:")
for attr in dir(qiskit_metal):
    if not attr.startswith('__'):
        print(f"  - {attr}")
print()

# Explore qlibrary
print("QLIBRARY CONTENTS:")
if hasattr(qiskit_metal, 'qlibrary'):
    qlib = qiskit_metal.qlibrary
    for attr in dir(qlib):
        if not attr.startswith('__'):
            print(f"  - {attr}")
            
            # Check if it's a module with submodules
            try:
                sub = getattr(qlib, attr)
                if hasattr(sub, '__path__'):
                    print(f"      (package with submodules)")
                    for subattr in dir(sub):
                        if not subattr.startswith('__') and not subattr.startswith('_'):
                            print(f"        - {subattr}")
            except:
                pass
print()

# Look for transmon-like components
print("\nSEARCHING FOR TRANSMON COMPONENTS:")
for attr in dir(qiskit_metal):
    if 'transmon' in attr.lower() and not attr.startswith('__'):
        print(f"  Found: {attr}")

# Search through all modules
print("\nDETAILED MODULE SCAN:")
modules_to_check = ['qlibrary', 'designs', 'components', 'elements']

for module_name in modules_to_check:
    if hasattr(qiskit_metal, module_name):
        module = getattr(qiskit_metal, module_name)
        print(f"\n{module_name.upper()}:")
        for item in dir(module):
            if not item.startswith('__'):
                print(f"  - {item}")
                
                # Check for classes
                try:
                    cls = getattr(module, item)
                    if inspect.isclass(cls):
                        print(f"      (class)")
                except:
                    pass
