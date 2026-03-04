#!/usr/bin/env python
"""Find the correct ground plane method"""

from qiskit_metal import designs
import inspect

print("="*60)
print("Finding Ground Plane Method")
print("="*60)

# Create design
design = designs.DesignPlanar(metadata={'name': 'Test'})
print(f"✅ Design created: {design.name}")

# List all methods that might be ground-related
print("\n🔍 Searching for ground-related methods:")
ground_methods = []
for method_name in dir(design):
    if 'ground' in method_name.lower() and not method_name.startswith('_'):
        ground_methods.append(method_name)
        method = getattr(design, method_name)
        if callable(method):
            print(f"  ✅ {method_name}() - callable")
        else:
            print(f"  📄 {method_name} - attribute")

# If no ground methods, check for chip-related methods
if not ground_methods:
    print("\n🔍 Checking chip-related methods:")
    chip_methods = []
    for method_name in dir(design):
        if 'chip' in method_name.lower() and not method_name.startswith('_'):
            chip_methods.append(method_name)
            method = getattr(design, method_name)
            if callable(method):
                print(f"  ✅ {method_name}()")
            else:
                print(f"  📄 {method_name}")

# Check if ground plane exists by default
print("\n🔍 Checking default ground plane:")
if hasattr(design, 'ground_plane'):
    print(f"  ✅ ground_plane exists: {design.ground_plane}")
else:
    print("  ❌ No ground_plane attribute")

# Check chips attribute
if hasattr(design, 'chips'):
    print(f"\n✅ chips attribute exists: {type(design.chips)}")
    print(f"  Chips keys: {list(design.chips.keys())}")
    
    # Check if chips have ground info
    for chip_name, chip in design.chips.items():
        print(f"\n  Chip '{chip_name}' attributes:")
        chip_attrs = [a for a in dir(chip) if not a.startswith('_')][:10]
        print(f"    {chip_attrs}")

print("\n" + "="*60)
