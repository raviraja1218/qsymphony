#!/usr/bin/env python
"""Ultra minimal test to isolate the error"""

print("1. Importing qiskit_metal...")
from qiskit_metal import designs

print("2. Attempting to create design...")
try:
    # Try with empty metadata
    design = designs.DesignPlanar()
    print("✅ Design created with no arguments")
except Exception as e:
    print(f"❌ Error with no args: {e}")
    
try:
    # Try with explicit metadata
    design = designs.DesignPlanar(metadata={})
    print("✅ Design created with empty metadata")
except Exception as e:
    print(f"❌ Error with empty metadata: {e}")
    
try:
    # Try with name as metadata
    design = designs.DesignPlanar(metadata={"name": "Test"})
    print("✅ Design created with name metadata")
except Exception as e:
    print(f"❌ Error with name metadata: {e}")
    
print("\n3. Checking addict installation...")
try:
    import addict
    print(f"addict version: {addict.__version__}")
    print(f"addict location: {addict.__file__}")
    
    # Test addict directly
    d = addict.Dict()
    d.update({"test": 1})
    print("✅ addict works")
except Exception as e:
    print(f"❌ addict error: {e}")
