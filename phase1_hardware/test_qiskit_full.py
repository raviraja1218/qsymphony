#!/usr/bin/env python
"""Test full Qiskit Metal functionality"""
print("Testing Qiskit Metal imports...")

try:
    from qiskit_metal import designs, draw
    print("✅ Base imports OK")
    
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    print("✅ TransmonPocket OK")
    
    from qiskit_metal.qlibrary.terminations.lumped_capacitor import LumpedCapacitor
    print("✅ LumpedCapacitor OK")
    
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
    print("✅ RouteMeander OK")
    
    # Create a test design
    design = designs.DesignPlanar()
    print("✅ Design creation OK")
    
    print("\n🎉 All Qiskit Metal components working!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
