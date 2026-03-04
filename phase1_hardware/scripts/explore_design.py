#!/usr/bin/env python
"""Explore DesignPlanar methods"""

from qiskit_metal import designs

print("="*60)
print("Exploring DesignPlanar methods")
print("="*60)

# Create design
design = designs.DesignPlanar(metadata={'name': 'TestDesign'})
print(f"✅ Design created: {design.name}")
print(f"Design type: {type(design)}")

# List all methods
print("\n📋 Available methods:")
methods = [m for m in dir(design) if not m.startswith('_')]
for method in sorted(methods)[:20]:  # Show first 20
    print(f"  - {method}")

# Look for ground-related methods
print("\n🔍 Looking for ground-related methods:")
ground_methods = [m for m in dir(design) if 'ground' in m.lower()]
for method in ground_methods:
    print(f"  - {method}")

# Try to add ground plane using different approaches
print("\n🧪 Testing ground plane addition:")

# Method 1: Try add_ground_plane (what we tried)
try:
    design.add_ground_plane()
    print("✅ add_ground_plane() works")
except AttributeError:
    print("❌ add_ground_plane() does not exist")

# Method 2: Try make_ground_plane
try:
    design.make_ground_plane()
    print("✅ make_ground_plane() works")
except AttributeError:
    print("❌ make_ground_plane() does not exist")

# Method 3: Try add_ground
try:
    design.add_ground()
    print("✅ add_ground() works")
except AttributeError:
    print("❌ add_ground() does not exist")

# Method 4: Try create_ground_plane
try:
    design.create_ground_plane()
    print("✅ create_ground_plane() works")
except AttributeError:
    print("❌ create_ground_plane() does not exist")

# Method 5: Check if ground plane is added by default
print("\nℹ️  Checking if ground plane exists by default...")
if hasattr(design, 'ground_plane'):
    print(f"  ground_plane attribute: {design.ground_plane}")
else:
    print("  No ground_plane attribute")

# Method 6: Look at qgeometry
try:
    from qiskit_metal import qgeometry
    print("\n📦 qgeometry module available")
    print("qgeometry methods:", [m for m in dir(qgeometry) if not m.startswith('_')][:5])
except ImportError:
    print("\n❌ qgeometry not available")

print("\n" + "="*60)
