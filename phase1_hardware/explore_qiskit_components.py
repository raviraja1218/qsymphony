#!/usr/bin/env python
"""Explore available Qiskit Metal components"""

import pkgutil
import importlib
import qiskit_metal
from qiskit_metal import qlibrary

print("="*60)
print("Qiskit Metal Component Explorer")
print("="*60)

print(f"\nQiskit Metal version: {qiskit_metal.__version__}")
print(f"Installation path: {qiskit_metal.__file__}")

print("\n" + "="*60)
print("Available QLibrary Components:")
print("="*60)

def explore_qlibrary():
    """List all available components in qlibrary"""
    
    # Get all modules in qlibrary
    qlibrary_path = qlibrary.__path__[0]
    
    print(f"\nQLibrary path: {qlibrary_path}")
    
    # List all subdirectories
    import os
    components = []
    
    for item in os.listdir(qlibrary_path):
        item_path = os.path.join(qlibrary_path, item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            components.append(item)
            print(f"  📁 {item}/")
            
            # List Python files in this component
            for file in os.listdir(item_path):
                if file.endswith('.py') and not file.startswith('__'):
                    print(f"      📄 {file}")
    
    return components

components = explore_qlibrary()

print("\n" + "="*60)
print("Testing Common Capacitor Imports:")
print("="*60)

# Try different possible import paths
import_paths = [
    "qiskit_metal.qlibrary.terminations.lumped_capacitor",
    "qiskit_metal.qlibrary.terminations.capacitor",
    "qiskit_metal.qlibrary.terminations.LumpedCapacitor",
    "qiskit_metal.qlibrary.terminations.lumped_capacitor.LumpedCapacitor",
    "qiskit_metal.qlibrary.lumped_capacitor",
    "qiskit_metal.qlibrary.capacitors.lumped_capacitor",
    "qiskit_metal.qlibrary.terminations.lumped_element",
]

for import_path in import_paths:
    try:
        module = importlib.import_module(import_path)
        print(f"✅ {import_path} - SUCCESS")
        
        # List classes in module
        classes = [c for c in dir(module) if not c.startswith('_')]
        print(f"   Classes: {classes}")
        
    except ImportError as e:
        print(f"❌ {import_path} - {str(e)}")

print("\n" + "="*60)
print("Alternative: Using TransmonPocket Only")
print("="*60)
print("""
We can create layouts using just TransmonPocket and RouteMeander,
using built-in coupling instead of explicit capacitor components.

The transmon pocket itself has built-in coupling capacitors via its
connection pads. This might be sufficient for our layout generation.
""")
