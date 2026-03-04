#!/usr/bin/env python
"""Minimal test to find the exact error"""

from qiskit_metal import designs
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

print("1. Creating design...")
design = designs.DesignPlanar("Test")
design.overwrite_enabled = True
design.add_ground_plane()

print("2. Testing transmon creation...")
try:
    # Simplest possible options
    transmon_options = dict(
        pos_x='0mm',
        pos_y='0mm'
    )
    
    q1 = TransmonPocket(design, 'Q1', options=transmon_options)
    print("✅ Transmon created successfully")
    
    # Check the options format
    print(f"Options type: {type(transmon_options)}")
    print(f"Options: {transmon_options}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e)}")
    
    # Try to see what options are expected
    import inspect
    print("\nTransmonPocket signature:")
    print(inspect.signature(TransmonPocket.__init__))
