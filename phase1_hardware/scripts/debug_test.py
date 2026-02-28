#!/usr/bin/env python
"""Debug the test layout generation with detailed output"""

import sys
import os
import traceback

# Add the current directory to path
sys.path.insert(0, os.getcwd())

print("="*60)
print("DEBUG: Testing layout generation")
print("="*60)

# Import the function
try:
    from scripts.generate_layouts_fixed_v2 import create_single_layout
    print("✅ Successfully imported create_single_layout")
except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Create test parameters
test_params = {
    'layout_id': 'TEST001',
    'transmon_width_um': 200.0,
    'transmon_height_um': 150.0,
    'coupling_gap_um': 80.0,
    'resonator_length_um': 1200.0,
    'junction_area_nm2': 125.0,
    'substrate_thickness_um': 350.0
}

print("\nTest parameters:")
for k, v in test_params.items():
    print(f"  {k}: {v} (type: {type(v)})")

# Create output directory
output_dir = "./debug_output"
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# Try to generate layout
print("\nAttempting to generate layout...")
try:
    result = create_single_layout(test_params, output_dir)
    print(f"\nResult: {result['status']}")
    if result['status'] == 'success':
        print(f"  JSON: {result.get('json_file')}")
        print(f"  GDS: {result.get('gds_status')}")
        print(f"  Directory: {result.get('directory')}")
    else:
        print(f"  Error: {result.get('error')}")
        if 'trace' in result:
            print("\nTraceback:")
            print(result['trace'])
except Exception as e:
    print(f"❌ Exception during generation: {e}")
    traceback.print_exc()

# Check what was created
print("\n" + "="*60)
print("Files created:")
print("="*60)
os.system(f"ls -la {output_dir}/")
if os.path.exists(f"{output_dir}/TEST001"):
    print(f"\nContents of TEST001 directory:")
    os.system(f"ls -la {output_dir}/TEST001/")
else:
    print(f"\n❌ TEST001 directory not found")
