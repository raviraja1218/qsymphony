#!/usr/bin/env python
"""Simple test to verify Qiskit Metal basics"""

print("="*60)
print("Testing Qiskit Metal Basic Functionality")
print("="*60)

# Test imports
try:
    from qiskit_metal import designs
    print("✅ designs imported")
except ImportError as e:
    print(f"❌ designs import failed: {e}")

try:
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    print("✅ TransmonPocket imported")
except ImportError as e:
    print(f"❌ TransmonPocket import failed: {e}")

try:
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
    print("✅ RouteMeander imported")
except ImportError as e:
    print(f"❌ RouteMeander import failed: {e}")

try:
    from qiskit_metal.qlibrary.lumped.cap_n_interdigital import CapNInterdigital
    print("✅ CapNInterdigital imported")
except ImportError as e:
    print(f"❌ CapNInterdigital import failed: {e}")

# Try to create a minimal design
print("\nAttempting to create minimal design...")
try:
    design = designs.DesignPlanar("TestDesign")
    print("✅ Design created")
    
    # Add ground plane
    design.planar.add_gnd_polygon(layers=[1])
    print("✅ Ground plane added")
    
    # Try to create a simple component
    from qiskit_metal.qlibrary.sample_shapes.rectangle import Rectangle
    rect = Rectangle(design, 'test_rect', options={'pos_x': '0mm', 'pos_y': '0mm'})
    print("✅ Rectangle created")
    
    print("\n🎉 Qiskit Metal is working!")
    
except Exception as e:
    print(f"❌ Design creation failed: {e}")
    import traceback
    traceback.print_exc()
