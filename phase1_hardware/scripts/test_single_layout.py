#!/usr/bin/env python
"""Test generating a single layout with available components"""

from generate_layouts_fixed import create_single_layout
import os
import tempfile

# Create a test parameter dict
test_params = {
    'layout_id': 'TEST001',
    'transmon_width_um': 200,
    'transmon_height_um': 150,
    'coupling_gap_um': 80,
    'resonator_length_um': 1200,
    'junction_area_nm2': 125,
    'substrate_thickness_um': 350
}

# Create temp output directory
output_dir = './test_output'
os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("Testing single layout generation")
print("="*60)

# Generate layout
result = create_single_layout(test_params, output_dir)

print(f"\nResult: {result['status']}")
if result['status'] == 'success':
    print(f"✅ JSON: {result['json_file']}")
    print(f"✅ GDS: {result['gds_status']}")
    print(f"📁 Directory: {result['directory']}")
else:
    print(f"❌ Error: {result.get('error', 'Unknown error')}")

print("\nCheck output directory:")
os.system(f"ls -la {output_dir}/TEST001/")
