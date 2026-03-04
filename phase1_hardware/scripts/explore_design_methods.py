#!/usr/bin/env python
"""Comprehensive exploration of DesignPlanar methods"""

from qiskit_metal import designs
import inspect

print("="*60)
print("COMPREHENSIVE DESIGNPLANAR METHOD EXPLORATION")
print("="*60)

# Create design
design = designs.DesignPlanar(metadata={'name': 'TestDesign'})
print(f"✅ Design created: {design.name}")
print(f"Design type: {type(design)}")

# Get all methods and attributes
all_attrs = dir(design)
methods = []
attributes = []

for attr in all_attrs:
    if not attr.startswith('_'):
        try:
            obj = getattr(design, attr)
            if callable(obj):
                methods.append(attr)
            else:
                attributes.append(attr)
        except:
            pass

print(f"\n📋 ATTRIBUTES ({len(attributes)}):")
for attr in sorted(attributes)[:20]:  # Show first 20
    print(f"  - {attr}")
if len(attributes) > 20:
    print(f"  ... and {len(attributes)-20} more")

print(f"\n📋 METHODS ({len(methods)}):")
for method in sorted(methods):
    print(f"  ✅ {method}()")

print("\n🔍 Looking for connection-related methods:")
connection_methods = [m for m in methods if 'connect' in m.lower() or 'link' in m.lower()]
for method in connection_methods:
    print(f"  🔗 {method}()")

print("\n🔍 Looking for component addition methods:")
component_methods = [m for m in methods if 'component' in m.lower() or 'add' in m.lower()]
for method in component_methods[:10]:  # First 10
    print(f"  ➕ {method}()")

print("\n🔍 Looking for pin-related methods:")
pin_methods = [m for m in methods if 'pin' in m.lower()]
for method in pin_methods:
    print(f"  📌 {method}()")

print("\n" + "="*60)
