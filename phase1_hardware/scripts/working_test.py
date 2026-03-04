#!/usr/bin/env python
"""Test with correct metadata format"""

from qiskit_metal import designs

print("="*60)
print("Testing correct DesignPlanar initialization")
print("="*60)

# Method 1: With name as string (CORRECT)
try:
    design1 = designs.DesignPlanar("MyDesign")
    print("✅ Method 1: DesignPlanar('MyDesign') works")
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: With metadata dict (ALSO CORRECT)
try:
    design2 = designs.DesignPlanar(metadata={"name": "MyDesign2"})
    print("✅ Method 2: DesignPlanar(metadata={'name': 'MyDesign2'}) works")
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

print("\n" + "="*60)
