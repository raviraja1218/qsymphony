#!/usr/bin/env python
"""
Fix for Qiskit Metal addict bug
The issue: DesignPlanar("name") fails, but DesignPlanar(metadata={"name": "name"}) works
"""

from qiskit_metal import designs
import os

print("="*60)
print("Testing Qiskit Metal with proper metadata")
print("="*60)

# CORRECT WAY: Pass metadata as dictionary, not string
try:
    # This works - metadata as dict with 'name' key
    design = designs.DesignPlanar(metadata={"name": "TestDesign"})
    print("✅ Design created with metadata dict")
    
    # Add ground plane
    design.planar.add_gnd_polygon(layers=[1])
    print("✅ Ground plane added")
    
    # Create transmon
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    transmon_options = {
        'pos_x': '0mm',
        'pos_y': '0mm',
        'pocket_width': '200um',
        'pocket_height': '150um'
    }
    q1 = TransmonPocket(design, 'Q1', options=transmon_options)
    print(f"✅ Transmon created: {q1.name}")
    
    # Save design
    os.makedirs('./fix_test', exist_ok=True)
    design.save('./fix_test/fixed_design.json')
    print("✅ Design saved")
    
    print("\n🎉 FIX WORKED! Qiskit Metal is now working correctly")
    
except Exception as e:
    print(f"❌ Still failing: {e}")
    import traceback
    traceback.print_exc()
