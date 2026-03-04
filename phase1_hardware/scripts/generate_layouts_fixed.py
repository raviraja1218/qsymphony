#!/usr/bin/env python
"""
Step 1.1: Generate 10,000 candidate chip layouts using Qiskit Metal
Fixed options format
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import random
from pathlib import Path

# Qiskit Metal imports
try:
    from qiskit_metal import designs, draw
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase1_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Expand user paths
def expand_path(path):
    return str(Path(os.path.expanduser(path)).expanduser())

OUTPUT_DIR = Path(expand_path(config['paths']['layouts_raw']))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = OUTPUT_DIR.parent / 'layouts_index.csv'

# Set random seed
np.random.seed(config['simulation']['seed'])
random.seed(config['simulation']['seed'])

def generate_parameter_grid(config):
    """Generate parameter combinations"""
    
    transmon_params = config['transmon']
    cap_params = config['capacitor']
    res_params = config['resonator']
    
    # Generate 10,000 random samples
    print(f"Generating 10,000 random samples...")
    samples = []
    for _ in range(10000):
        sample = {
            'junction_width_nm': random.randint(transmon_params['junction_width_nm']['min'], 
                                               transmon_params['junction_width_nm']['max']),
            'junction_length_nm': random.randint(transmon_params['junction_length_nm']['min'], 
                                                transmon_params['junction_length_nm']['max']),
            'pad_area_um2': random.randint(transmon_params['pad_area_um2']['min'], 
                                          transmon_params['pad_area_um2']['max']),
            'gap_to_ground_um': random.randint(transmon_params['gap_to_ground_um']['min'], 
                                              transmon_params['gap_to_ground_um']['max']),
            'finger_length_um': random.randint(cap_params['finger_length_um']['min'], 
                                              cap_params['finger_length_um']['max']),
            'finger_width_um': random.randint(cap_params['finger_width_um']['min'], 
                                             cap_params['finger_width_um']['max']),
            'finger_count': random.randint(cap_params['finger_count']['min'], 
                                          cap_params['finger_count']['max']),
            'finger_gap_um': random.randint(cap_params['finger_gap_um']['min'], 
                                           cap_params['finger_gap_um']['max']),
            'hbar_thickness_um': random.randint(res_params['hbar_thickness_um']['min'], 
                                               res_params['hbar_thickness_um']['max']),
            'beam_length_um': random.randint(res_params['beam_length_um']['min'], 
                                            res_params['beam_length_um']['max']),
            'beam_width_um': random.randint(res_params['beam_width_um']['min'], 
                                           res_params['beam_width_um']['max']),
        }
        samples.append(sample)
    
    return samples

def create_chip_layout(design, params, layout_id):
    """Create a single chip layout"""
    
    design.delete_all_components()
    design.overwrite_enabled = True
    design.add_ground_plane()
    
    # Calculate pad size
    pad_size_um = np.sqrt(params['pad_area_um2'])
    pad_size_nm = pad_size_um * 1000
    
    # Transmon options - using proper format
    transmon_options = {
        'pos_x': '0mm',
        'pos_y': '0mm',
        'junction_width': f"{params['junction_width_nm']}nm",
        'junction_length': f"{params['junction_length_nm']}nm",
        'pad_width': f"{pad_size_nm:.0f}nm",
        'pad_height': f"{pad_size_nm:.0f}nm",
        'gap': f"{params['gap_to_ground_um']}um",
        'connection_pads': {
            'readout': {'loc_W': 1, 'pad_width': '175um'},
            'drive': {'loc_W': 0, 'pad_width': '100um'}
        }
    }
    
    print(f"Debug - Transmon options: {transmon_options}")  # Debug
    
    q1 = TransmonPocket(design, f'Q{layout_id}', options=transmon_options)
    
    # Capacitor options
    cap_options = {
        'pos_x': '200um',
        'pos_y': '0um',
        'finger_length': f"{params['finger_length_um']}um",
        'finger_width': f"{params['finger_width_um']}um",
        'finger_count': params['finger_count'],
        'finger_gap': f"{params['finger_gap_um']}um",
        'orientation': '0'
    }
    
    cap = CoupledLineTee(design, f'C{layout_id}', options=cap_options)
    
    # Resonator options
    res_options = {
        'pos_x': '400um',
        'pos_y': '0um',
        'length': f"{params['beam_length_um']}um",
        'width': f"{params['beam_width_um']}um",
        'orientation': '0'
    }
    
    res = RouteMeander(design, f'R{layout_id}', options=res_options)
    
    # Connect
    design.connect_components(f'Q{layout_id}', f'C{layout_id}')
    design.connect_components(f'C{layout_id}', f'R{layout_id}')
    
    return design

def validate_layout(design):
    """Validate layout"""
    try:
        if design.check_overlaps():
            return False, "Overlapping components"
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def save_layout(design, params, layout_id):
    """Save layout"""
    json_str = design.save_to_string()
    layout_data = json.loads(json_str)
    
    layout_data['metadata'] = {
        'layout_id': f'layout_{layout_id:06d}',
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'version': '1.0'
    }
    
    filename = OUTPUT_DIR / f'layout_{layout_id:06d}.json'
    with open(filename, 'w') as f:
        json.dump(layout_data, f, indent=2)
    
    return filename

def test_single_layout():
    """Test creating one layout"""
    print("Testing single layout creation...")
    
    # Simple test params
    params = {
        'junction_width_nm': 200,
        'junction_length_nm': 150,
        'pad_area_um2': 100,
        'gap_to_ground_um': 20,
        'finger_length_um': 50,
        'finger_width_um': 5,
        'finger_count': 10,
        'finger_gap_um': 3,
        'hbar_thickness_um': 10,
        'beam_length_um': 200,
        'beam_width_um': 15,
    }
    
    try:
        design = designs.DesignPlanar("Test")
        design = create_chip_layout(design, params, 1)
        print("✅ Layout created successfully")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("STEP 1.1: Generate Layouts")
    print("="*60)
    
    # First test single layout
    if not test_single_layout():
        print("Test failed - aborting")
        return
    
    # Generate full batch
    print("\n📊 Generating parameter combinations...")
    samples = generate_parameter_grid(config)
    
    print("\n🏗️  Generating layouts...")
    index_data = []
    valid_count = 0
    
    for i, params in enumerate(tqdm(samples[:10])):  # Start with 10
        layout_id = i + 1
        
        try:
            design = designs.DesignPlanar(f"Layout_{layout_id}")
            design = create_chip_layout(design, params, layout_id)
            is_valid, message = validate_layout(design)
            
            if is_valid:
                filename = save_layout(design, params, layout_id)
                index_data.append({'layout_id': f'layout_{layout_id:06d}', 'filename': str(filename), **params, 'valid': True})
                valid_count += 1
        except Exception as e:
            print(f"Error on layout {layout_id}: {e}")
    
    print(f"\n✅ Generated {valid_count} valid layouts")

if __name__ == "__main__":
    main()
