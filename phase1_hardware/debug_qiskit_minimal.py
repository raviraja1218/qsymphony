#!/usr/bin/env python
"""
Minimal debug script to isolate the Qiskit Metal dictionary error
"""

import traceback
from qiskit_metal import designs
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

print("="*60)
print("QISKIT METAL DEBUG - MINIMAL TEST")
print("="*60)

try:
    # Create design
    design = designs.DesignPlanar("Debug_Design")
    print("✅ Design created")
    
    # Simplest possible transmon options
    transmon_options = {
        'pos_x': '0mm',
        'pos_y': '0mm',
        'pocket_width': '200um',
        'pocket_height': '150um'
    }
    
    print(f"Options type: {type(transmon_options)}")
    print(f"Options: {transmon_options}")
    
    # Try to create transmon
    q1 = TransmonPocket(design, 'Q_debug', options=transmon_options)
    print("✅ Transmon created")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTraceback:")
    traceback.print_exc()

print("\n" + "="*60)
