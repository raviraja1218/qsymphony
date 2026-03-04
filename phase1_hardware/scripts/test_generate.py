#!/usr/bin/env python
"""Test version - generate 10 layouts to verify everything works"""

from generate_layouts import generate_parameter_grid, create_chip_layout, validate_layout, save_layout
from generate_layouts import designs, OUTPUT_DIR, INDEX_FILE, config
import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / 'config' / 'phase1_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("="*60)
print("TEST MODE: Generating 10 layouts")
print("="*60)

# Generate only 10 samples
samples = generate_parameter_grid(config)[:10]

valid_count = 0

for i, params in enumerate(samples):
    layout_id = i + 1
    print(f"\nTesting layout {layout_id}/10...")
    
    try:
        design = designs.DesignPlanar(f"Test_Layout_{layout_id}")
        design = create_chip_layout(design, params, layout_id)
        is_valid, message = validate_layout(design)
        
        if is_valid:
            filename = save_layout(design, params, layout_id)
            print(f"  ✅ Layout {layout_id} valid - saved to {filename}")
            valid_count += 1
        else:
            print(f"  ❌ Layout {layout_id} invalid: {message}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n✅ Test complete: {valid_count}/10 layouts generated successfully")
