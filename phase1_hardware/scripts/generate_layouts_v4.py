#!/usr/bin/env python
"""
Step 1.1: Generate layouts with correct export method
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
from qiskit_metal import designs
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

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

def generate_parameter_samples(n_samples=10000):
    """Generate random parameter samples"""
    transmon_params = config['transmon']
    cap_params = config['capacitor']
    res_params = config['resonator']
    
    samples = []
    for _ in range(n_samples):
        sample = {
            'junction_width_nm': random.randint(
                transmon_params['junction_width_nm']['min'],
                transmon_params['junction_width_nm']['max']
            ),
            'junction_length_nm': random.randint(
                transmon_params['junction_length_nm']['min'],
                transmon_params['junction_length_nm']['max']
            ),
            'pad_area_um2': random.randint(
                transmon_params['pad_area_um2']['min'],
                transmon_params['pad_area_um2']['max']
            ),
            'gap_to_ground_um': random.randint(
                transmon_params['gap_to_ground_um']['min'],
                transmon_params['gap_to_ground_um']['max']
            ),
            'finger_length_um': random.randint(
                cap_params['finger_length_um']['min'],
                cap_params['finger_length_um']['max']
            ),
            'finger_width_um': random.randint(
                cap_params['finger_width_um']['min'],
                cap_params['finger_width_um']['max']
            ),
            'finger_count': random.randint(
                cap_params['finger_count']['min'],
                cap_params['finger_count']['max']
            ),
            'finger_gap_um': random.randint(
                cap_params['finger_gap_um']['min'],
                cap_params['finger_gap_um']['max']
            ),
            'hbar_thickness_um': random.randint(
                res_params['hbar_thickness_um']['min'],
                res_params['hbar_thickness_um']['max']
            ),
            'beam_length_um': random.randint(
                res_params['beam_length_um']['min'],
                res_params['beam_length_um']['max']
            ),
            'beam_width_um': random.randint(
                res_params['beam_width_um']['min'],
                res_params['beam_width_um']['max']
            ),
        }
        samples.append(sample)
    
    return samples

def save_layout(design, params, layout_id):
    """Save layout - use whatever method exists"""
    
    # Try different save methods
    try:
        # Method 1: save_design()
        if hasattr(design, 'save_design'):
            filename = OUTPUT_DIR / f'layout_{layout_id:06d}.json'
            design.save_design(str(filename))
            return filename
            
        # Method 2: save_to_string() doesn't exist, so skip
        # Method 3: Just save the design object as JSON string
        else:
            # Create minimal representation
            layout_data = {
                'metadata': {
                    'layout_id': f'layout_{layout_id:06d}',
                    'timestamp': datetime.now().isoformat(),
                    'parameters': params,
                    'version': '1.0'
                },
                'components': {name: comp.options for name, comp in design.components.items()},
                'connections': []  # We'll need to add connection info
            }
            
            filename = OUTPUT_DIR / f'layout_{layout_id:06d}.json'
            with open(filename, 'w') as f:
                json.dump(layout_data, f, indent=2)
            return filename
            
    except Exception as e:
        print(f"Save error: {e}")
        return None

def test_single_layout():
    """Test creating one layout"""
    print("Testing single layout creation...")
    
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
        # Create design
        design = designs.DesignPlanar(metadata={'name': 'TestLayout'})
        design.overwrite_enabled = True
        
        # Calculate pad size
        pad_size_um = np.sqrt(params['pad_area_um2'])
        pad_size_nm = pad_size_um * 1000
        
        # Add transmon
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
        q1 = TransmonPocket(design, 'Q1', options=transmon_options)
        
        # Add capacitor
        cap_options = {
            'pos_x': '200um',
            'pos_y': '0um',
            'finger_length': f"{params['finger_length_um']}um",
            'finger_width': f"{params['finger_width_um']}um",
            'finger_count': params['finger_count'],
            'finger_gap': f"{params['finger_gap_um']}um",
            'orientation': '0'
        }
        cap = CoupledLineTee(design, 'C1', options=cap_options)
        
        # Add resonator
        res_options = {
            'pos_x': '400um',
            'pos_y': '0um',
            'length': f"{params['beam_length_um']}um",
            'width': f"{params['beam_width_um']}um",
            'orientation': '0'
        }
        res = RouteMeander(design, 'R1', options=res_options)
        
        # Save the design
        filename = save_layout(design, params, 1)
        
        if filename and filename.exists():
            print(f"✅ Layout saved to {filename}")
            return True
        else:
            print("❌ Failed to save layout")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("STEP 1.1: Generate 10,000 Layouts")
    print("="*60)
    
    # Test single layout first
    if not test_single_layout():
        print("\n❌ Test failed - aborting")
        return
    
    print("\n✅ Test passed - proceeding with full generation...")
    
    # Generate full batch
    print("\n📊 Generating parameter samples...")
    samples = generate_parameter_samples(10000)
    print(f"Generated {len(samples)} parameter sets")
    
    print("\n🏗️  Generating layouts...")
    index_data = []
    valid_count = 0
    
    for i, params in enumerate(tqdm(samples, desc="Creating layouts")):
        layout_id = i + 1
        
        try:
            # Create design
            design = designs.DesignPlanar(metadata={'name': f"Layout_{layout_id}"})
            design.overwrite_enabled = True
            
            # Calculate pad size
            pad_size_um = np.sqrt(params['pad_area_um2'])
            pad_size_nm = pad_size_um * 1000
            
            # Add transmon
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
            q1 = TransmonPocket(design, f'Q{layout_id}', options=transmon_options)
            
            # Add capacitor
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
            
            # Add resonator
            res_options = {
                'pos_x': '400um',
                'pos_y': '0um',
                'length': f"{params['beam_length_um']}um",
                'width': f"{params['beam_width_um']}um",
                'orientation': '0'
            }
            res = RouteMeander(design, f'R{layout_id}', options=res_options)
            
            # Save the design
            filename = save_layout(design, params, layout_id)
            
            if filename:
                index_data.append({
                    'layout_id': f'layout_{layout_id:06d}',
                    'filename': str(filename),
                    **params,
                    'valid': True
                })
                valid_count += 1
            else:
                index_data.append({
                    'layout_id': f'layout_{layout_id:06d}',
                    'filename': '',
                    **params,
                    'valid': False
                })
                
        except Exception as e:
            print(f"\nError on layout {layout_id}: {e}")
            index_data.append({
                'layout_id': f'layout_{layout_id:06d}',
                'filename': '',
                **params,
                'valid': False
            })
        
        # Save index periodically
        if (i + 1) % 1000 == 0:
            df_temp = pd.DataFrame(index_data[-1000:])
            temp_file = INDEX_FILE.parent / f'index_checkpoint_{i+1}.csv'
            df_temp.to_csv(temp_file, index=False)
    
    # Save final index
    print("\n💾 Saving final index file...")
    df = pd.DataFrame(index_data)
    df.to_csv(INDEX_FILE, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("📋 GENERATION SUMMARY")
    print("="*60)
    print(f"Total layouts: {len(samples)}")
    print(f"✅ Valid: {valid_count}")
    print(f"❌ Invalid: {len(samples) - valid_count}")
    print(f"📍 Output directory: {OUTPUT_DIR}")
    print(f"📍 Index file: {INDEX_FILE}")
    
    return valid_count

if __name__ == "__main__":
    main()
