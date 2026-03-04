#!/usr/bin/env python
"""Fix pyEPR imports and test connection"""

import sys
import subprocess

print("="*60)
print("Fixing pyEPR Configuration")
print("="*60)

# Check pyEPR version
try:
    import pyEPR
    print(f"✅ pyEPR installed: version {pyEPR.__version__}")
    
    # List available modules
    import inspect
    print("\n📋 Available pyEPR modules:")
    for name, obj in inspect.getmembers(pyEPR):
        if not name.startswith('_'):
            print(f"  - {name}")
    
    # Check core module contents
    print("\n📋 pyEPR.core contents:")
    from pyEPR import core
    for name, obj in inspect.getmembers(core):
        if not name.startswith('_') and inspect.isclass(obj):
            print(f"  - {name} (class)")
        elif not name.startswith('_'):
            print(f"  - {name}")
            
except ImportError as e:
    print(f"❌ pyEPR import failed: {e}")

print("\n" + "="*60)
print("For real Ansys HFSS connection:")
print("1. Install Ansys HFSS on Windows")
print("2. Enable COM automation in Ansys")
print("3. Use the correct import: from pyEPR import AnsysAPD, ProjectInfo")
print("="*60)
