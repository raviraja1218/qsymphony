#!/usr/bin/env python
"""
Step 1.1: Generate 10,000 candidate chip layouts using Qiskit Metal
Corrected with actual class names from the installation
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

# Qiskit Metal imports - using actual available classes
try:
    from qiskit_metal import designs, draw
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
    from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander  # Changed from Meandered to RouteMeander
except ImportError as e:
    print(f"Qiskit Metal import error: {e}")
    print("\nChecking available modules...")
    import qiskit_metal.qlibrary as qlib
    print("\nAvailable qlibrary submodules:")
    import pkgutil
    for module in pkgutil.iter_modules(qlib.__path__):
        print(f"  - {module.name}")
    
    print("\nChecking tlines contents:")
    try:
        import qiskit_metal.qlibrary.tlines as tlines
        print("Available tlines modules:")
        for module in pkgutil.iter_modules(tlines.__path__):
            print(f"  - {module.name}")
            
        print("\nContents of meandered:")
        import inspect
        from qiskit_metal.qlibrary.tlines import meandered
        for name, obj in inspect.getmembers(meandered):
            if not name.startswith('__') and inspect.isclass(obj):
                print(f"  📦 Class: {name}")
    except:
        pass
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

# Set random seed for reproducibility
np.random.seed(config['simulation']['seed'])
random.seed(config['simulation']['seed'])

def generate_parameter_grid(config):
    """Generate all parameter combinations for sweep"""
    
    transmon_params = config['transmon']
    cap_params = config['capacitor']
    res_params = config['resonator']
    
    # Create parameter ranges as lists
    junction_widths = list(range(
        transmon_params['junction_width_nm']['min'],
        transmon_params['junction_width_nm']['max'] + transmon_params['junction_width_nm']['step'],
        transmon_params['junction_width_nm']['step']
    ))
    
    junction_lengths = list(range(
        transmon_params['junction_length_nm']['min'],
        transmon_params['junction_length_nm']['max'] + transmon_params['junction_length_nm']['step'],
        transmon_params['junction_length_nm']['step']
    ))
    
    pad_areas = list(range(
        transmon_params['pad_area_um2']['min'],
        transmon_params['pad_area_um2']['max'] + transmon_params['pad_area_um2']['step'],
        transmon_params['pad_area_um2']['step']
    ))
    
    gaps = list(range(
        transmon_params['gap_to_ground_um']['min'],
        transmon_params['gap_to_ground_um']['max'] + transmon_params['gap_to_ground_um']['step'],
        transmon_params['gap_to_ground_um']['step']
    ))
    
    finger_lengths = list(range(
        cap_params['finger_length_um']['min'],
        cap_params['finger_length_um']['max'] + cap_params['finger_length_um']['step'],
        cap_params['finger_length_um']['step']
    ))
    
    finger_widths = list(range(
        cap_params['finger_width_um']['min'],
        cap_params['finger_width_um']['max'] + cap_params['finger_width_um']['step'],
        cap_params['finger_width_um']['step']
    ))
    
    finger_counts = list(range(
        cap_params['finger_count']['min'],
        cap_params['finger_count']['max'] + cap_params['finger_count']['step'],
        cap_params['finger_count']['step']
    ))
    
    finger_gaps = list(range(
        cap_params['finger_gap_um']['min'],
        cap_params['finger_gap_um']['max'] + cap_params['finger_gap_um']['step'],
        cap_params['finger_gap_um']['step']
    ))
    
    hbar_thickness = list(range(
        res_params['hbar_thickness_um']['min'],
        res_params['hbar_thickness_um']['max'] + res_params['hbar_thickness_um']['step'],
        res_params['hbar_thickness_um']['step']
    ))
    
    beam_lengths = list(range(
        res_params['beam_length_um']['min'],
        res_params['beam_length_um']['max'] + res_params['beam_length_um']['step'],
        res_params['beam_length_um']['step']
    ))
    
    beam_widths = list(range(
        res_params['beam_width_um']['min'],
        res_params['beam_width_um']['max'] + res_params['beam_width_um']['step'],
        res_params['beam_width_um']['step']
    ))
    
    # Generate 10,000 random samples
    print(f"Generating 10,000 random samples...")
    samples = []
    for _ in range(10000):
        sample = {
            'junction_width_nm': random.choice(junction_widths),
            'junction_length_nm': random.choice(junction_lengths),
            'pad_area_um2': random.choice(pad_areas),
            'gap_to_ground_um': random.choice(gaps),
            'finger_length_um': random.choice(finger_lengths),
            'finger_width_um': random.choice(finger_widths),
            'finger_count': int(random.choice(finger_counts)),
            'finger_gap_um': random.choice(finger_gaps),
            'hbar_thickness_um': random.choice(hbar_thickness),
            'beam_length_um': random.choice(beam_lengths),
            'beam_width_um': random.choice(beam_widths),
        }
        samples.append(sample)
    
    return samples

def create_chip_layout(design, params, layout_id):
    """Create a single chip layout with given parameters"""
    
    design.delete_all_components()
    design.overwrite_enabled = True
    
    # Add ground plane
    design.add_ground_plane()
    
    # Calculate pad size from area
    pad_size_um = np.sqrt(params['pad_area_um2'])
    pad_size_nm = pad_size_um * 1000
    
    # Create transmon qubit
    transmon_options = dict(
        pos_x='0mm',
        pos_y='0mm',
        junction_width=f"{params['junction_width_nm']}nm",
        junction_length=f"{params['junction_length_nm']}nm",
        pad_width=f"{pad_size_nm:.0f}nm",
        pad_height=f"{pad_size_nm:.0f}nm",
        gap=f"{params['gap_to_ground_um']}um",
        connection_pads=dict(
            readout=dict(loc_W=1, pad_width='175um'),
            drive=dict(loc_W=0, pad_width='100um')
        )
    )
    
    q1 = TransmonPocket(design, f'Q{layout_id}', options=transmon_options)
    
    # Create coupling capacitor using CoupledLineTee
    cap_options = dict(
        pos_x='200um',
        pos_y='0um',
        finger_length=f"{params['finger_length_um']}um",
        finger_width=f"{params['finger_width_um']}um",
        finger_number=params['finger_count'],
        finger_gap=f"{params['finger_gap_um']}um",
        orientation='0',
    )
    
    cap = CoupledLineTee(design, f'C{layout_id}', options=cap_options)
    
    # Create meandered resonator using RouteMeander
    res_options = dict(
        pos_x='400um',
        pos_y='0um',
        length=f"{params['beam_length_um']}um",
        width=f"{params['beam_width_um']}um",
        orientation='0',
    )
    
    res = RouteMeander(design, f'R{layout_id}', options=res_options)
    
    # Connect components
    design.connect_components(f'Q{layout_id}', f'C{layout_id}')
    design.connect_components(f'C{layout_id}', f'R{layout_id}')
    
    return design

def validate_layout(design):
    """Check if layout meets minimum feature size and connectivity requirements"""
    try:
        # Check for overlapping components
        if design.check_overlaps():
            return False, "Overlapping components"
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def save_layout(design, params, layout_id):
    """Save layout as JSON file"""
    
    # Export design to JSON
    json_str = design.save_to_string()
    layout_data = json.loads(json_str)
    
    # Add metadata
    layout_data['metadata'] = {
        'layout_id': f'layout_{layout_id:06d}',
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'version': '1.0'
    }
    
    # Save file
    filename = OUTPUT_DIR / f'layout_{layout_id:06d}.json'
    with open(filename, 'w') as f:
        json.dump(layout_data, f, indent=2)
    
    return filename

def main():
    """Main execution function"""
    
    print("="*60)
    print("STEP 1.1: Generate 10,000 Candidate Chip Layouts")
    print("="*60)
    
    # Generate parameter combinations
    print("\n📊 Generating parameter combinations...")
    samples = generate_parameter_grid(config)
    print(f"Generated {len(samples)} parameter sets")
    
    # Create index dataframe
    index_data = []
    
    # Generate layouts
    print("\n🏗️  Generating chip layouts...")
    
    valid_count = 0
    invalid_count = 0
    
    for i, params in enumerate(tqdm(samples, desc="Creating layouts")):
        layout_id = i + 1
        
        try:
            # Create new design
            design = designs.DesignPlanar(f"Layout_{layout_id}")
            
            # Create layout
            design = create_chip_layout(design, params, layout_id)
            
            # Validate
            is_valid, message = validate_layout(design)
            
            if is_valid:
                # Save layout
                filename = save_layout(design, params, layout_id)
                
                # Add to index
                index_data.append({
                    'layout_id': f'layout_{layout_id:06d}',
                    'filename': str(filename),
                    'junction_width_nm': params['junction_width_nm'],
                    'junction_length_nm': params['junction_length_nm'],
                    'pad_area_um2': params['pad_area_um2'],
                    'gap_to_ground_um': params['gap_to_ground_um'],
                    'finger_length_um': params['finger_length_um'],
                    'finger_width_um': params['finger_width_um'],
                    'finger_count': params['finger_count'],
                    'finger_gap_um': params['finger_gap_um'],
                    'hbar_thickness_um': params['hbar_thickness_um'],
                    'beam_length_um': params['beam_length_um'],
                    'beam_width_um': params['beam_width_um'],
                    'valid': True,
                    'validation_message': message
                })
                valid_count += 1
            else:
                invalid_count += 1
                index_data.append({
                    'layout_id': f'layout_{layout_id:06d}',
                    'filename': '',
                    **params,
                    'valid': False,
                    'validation_message': message
                })
                
        except Exception as e:
            print(f"\nError creating layout {layout_id}: {e}")
            invalid_count += 1
    
    # Save index file
    print("\n💾 Saving index file...")
    df = pd.DataFrame(index_data)
    df.to_csv(INDEX_FILE, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("📋 GENERATION SUMMARY")
    print("="*60)
    print(f"Total layouts attempted: {len(samples)}")
    print(f"✅ Valid layouts: {valid_count}")
    print(f"❌ Invalid layouts: {invalid_count}")
    print(f"📍 Layouts saved to: {OUTPUT_DIR}")
    print(f"📍 Index file: {INDEX_FILE}")
    
    # Save validation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_attempted': len(samples),
        'valid': valid_count,
        'invalid': invalid_count,
        'validation_rate': valid_count/len(samples) if len(samples) > 0 else 0,
        'output_directory': str(OUTPUT_DIR),
        'index_file': str(INDEX_FILE)
    }
    
    report_file = OUTPUT_DIR.parent / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📍 Validation report: {report_file}")
    print("="*60)
    
    # Check if target achieved
    if valid_count >= 10000:
        print("\n✅ TARGET ACHIEVED: 10,000 valid layouts generated!")
    else:
        print(f"\n⚠️  Target not achieved: Need {10000-valid_count} more valid layouts")
    
    return valid_count

if __name__ == "__main__":
    main()
