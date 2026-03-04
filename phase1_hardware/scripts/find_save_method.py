#!/usr/bin/env python
"""Find how to save/export designs in Qiskit Metal"""

from qiskit_metal import designs
import inspect

print("="*60)
print("FINDING SAVE/EXPORT METHODS")
print("="*60)

# Create design
design = designs.DesignPlanar(metadata={'name': 'TestDesign'})
print(f"✅ Design created: {design.name}")

# Look for save/export methods
print("\n🔍 Searching for save/export methods:")
save_methods = []
for method_name in dir(design):
    if 'save' in method_name.lower() or 'export' in method_name.lower() or 'to_' in method_name.lower():
        if not method_name.startswith('_'):
            save_methods.append(method_name)
            method = getattr(design, method_name)
            if callable(method):
                print(f"  ✅ {method_name}()")
            else:
                print(f"  📄 {method_name}")

# Also check for serialization
print("\n🔍 Searching for serialization methods:")
serial_methods = []
for method_name in dir(design):
    if 'json' in method_name.lower() or 'serialize' in method_name.lower() or 'dump' in method_name.lower():
        if not method_name.startswith('_'):
            serial_methods.append(method_name)
            method = getattr(design, method_name)
            if callable(method):
                print(f"  ✅ {method_name}()")
            else:
                print(f"  📄 {method_name}")

print("\n" + "="*60)
