#!/usr/bin/env python
"""Explore pin structure of components"""

from qiskit_metal import designs
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

print("="*60)
print("EXPLORING COMPONENT PINS")
print("="*60)

# Create design
design = designs.DesignPlanar(metadata={'name': 'PinTest'})

# Create a transmon
transmon_options = {
    'pos_x': '0mm',
    'pos_y': '0mm',
    'junction_width': '200nm',
    'junction_length': '150nm',
    'pad_width': '10um',
    'pad_height': '10um',
    'gap': '20um',
}
q1 = TransmonPocket(design, 'Q1', options=transmon_options)

# Create a coupler
cap_options = {
    'pos_x': '200um',
    'pos_y': '0um',
    'finger_length': '50um',
    'finger_width': '5um',
    'finger_count': 10,
    'finger_gap': '3um',
}
cap = CoupledLineTee(design, 'C1', options=cap_options)

# Create a resonator
res_options = {
    'pos_x': '400um',
    'pos_y': '0um',
    'length': '200um',
    'width': '15um',
}
res = RouteMeander(design, 'R1', options=res_options)

print("\n📋 Transmon pins:")
if hasattr(q1, 'pins'):
    for pin_name, pin_info in q1.pins.items():
        print(f"  - {pin_name}: {pin_info}")
else:
    print("  No pins attribute")

print("\n📋 Coupler pins:")
if hasattr(cap, 'pins'):
    for pin_name, pin_info in cap.pins.items():
        print(f"  - {pin_name}: {pin_info}")
else:
    print("  No pins attribute")

print("\n📋 Resonator pins:")
if hasattr(res, 'pins'):
    for pin_name, pin_info in res.pins.items():
        print(f"  - {pin_name}: {pin_info}")
else:
    print("  No pins attribute")

print("\n" + "="*60)
