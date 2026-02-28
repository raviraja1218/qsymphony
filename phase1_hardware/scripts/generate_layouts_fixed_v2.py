#!/usr/bin/env python
"""
Generate Qiskit Metal layouts using available components
FIXED VERSION - Proper dictionary handling
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import time
import traceback

# Import available Qiskit Metal components
from qiskit_metal import designs, draw
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

# Try to import interdigital capacitor
try:
    from qiskit_metal.qlibrary.lumped.cap_n_interdigital import CapNInterdigital
    HAS_INTERDIGITAL = True
    print("✅ Using CapNInterdigital")
except ImportError:
    try:
        from qiskit_metal.qlibrary.lumped.cap_3_interdigital import Cap3Interdigital
        HAS_INTERDIGITAL = True
        print("✅ Using Cap3Interdigital")
    except ImportError:
        HAS_INTERDIGITAL = False
        print("⚠️ No interdigital capacitor found, using connection pads only")

def create_single_layout(params_dict, output_dir):
    """Create one chip layout from parameters"""
    
    layout_id = params_dict['layout_id']
    
    try:
        # Create design
        design = designs.DesignPlanar(f"Design_{layout_id}")
        
        # Extract parameters safely
        try:
            w = float(params_dict.get('transmon_width_um', 200))
            h = float(params_dict.get('transmon_height_um', 150))
            gap = float(params_dict.get('coupling_gap_um', 80))
            length = float(params_dict.get('resonator_length_um', 1000))
        except (ValueError, TypeError) as e:
            # Use defaults if conversion fails
            w, h, gap, length = 200, 150, 80, 1000
            print(f"  ⚠️ Using defaults for {layout_id}: {e}")
        
        # Add ground plane
        design.planar.add_gnd_polygon(layers=[1])
        
        # Create transmon qubit with built-in coupling pads
        transmon_options = {
            'pos_x': '0mm',
            'pos_y': '0mm',
            'pocket_width': f'{w}um',
            'pocket_height': f'{h}um',
            'pocket_type': 'rectangular',
            'inductor_width': '10um',
            'inductor_gap': '10um',
            'connection_pads': {
                'readout': {'loc': 'right', 'pad_width': '50um', 'pad_height': '50um'},
                'drive': {'loc': 'left', 'pad_width': '50um', 'pad_height': '50um'},
                'flux': {'loc': 'bottom', 'pad_width': '50um', 'pad_height': '50um'}
            }
        }
        
        q1 = TransmonPocket(design, f'Q_{layout_id}', options=transmon_options)
        
        # Create meandered resonator
        meander_options = {
            'pos_x': f'{gap + 100}um',
            'pos_y': '0mm',
            'total_length': f'{length}um',
            'meander': 'sinusoidal',
            'amplitude': '100um',
            'period': '200um',
            'width': '10um',
            'layer': '1'
        }
        
        res = RouteMeander(design, f'R_{layout_id}', options=meander_options)
        
        # Connect transmon to resonator
        design.connect(f'Q_{layout_id}.readout', f'R_{layout_id}.tline')
        
        # Create coupling capacitor if available
        if HAS_INTERDIGITAL:
            try:
                cap_options = {
                    'pos_x': f'{gap}um',
                    'pos_y': '0mm',
                    'finger_length': '50um',
                    'finger_width': '5um',
                    'finger_gap': '5um',
                    'n_fingers': 5,
                    'orientation': '0'
                }
                cap = CapNInterdigital(design, f'C_{layout_id}', options=cap_options)
                # Re-route through capacitor
                design.delete_connection(f'Q_{layout_id}.readout', f'R_{layout_id}.tline')
                design.connect(f'Q_{layout_id}.readout', f'C_{layout_id}.tline')
                design.connect(f'C_{layout_id}.tline', f'R_{layout_id}.tline')
                has_cap = True
            except Exception as e:
                has_cap = False
                print(f"  ⚠️ Capacitor failed for {layout_id}, using direct connection")
        else:
            has_cap = False
        
        # Create layout directory
        layout_dir = os.path.join(output_dir, layout_id)
        os.makedirs(layout_dir, exist_ok=True)
        
        # Save design as JSON
        json_file = os.path.join(layout_dir, f"{layout_id}.json")
        design.save(json_file)
        
        # Export GDS
        gds_file = None
        gds_status = "failed"
        try:
            gds_file = os.path.join(layout_dir, f"{layout_id}.gds")
            design.export_gds(gds_file)
            gds_status = "exported"
        except Exception as e:
            gds_status = f"failed: {str(e)}"
        
        # Save parameters
        params_file = os.path.join(layout_dir, "parameters.json")
        with open(params_file, 'w') as f:
            # Convert any numpy types to Python native
            clean_params = {}
            for k, v in params_dict.items():
                if hasattr(v, 'item'):  # Check if numpy type
                    clean_params[k] = v.item()
                else:
                    clean_params[k] = v
            json.dump(clean_params, f, indent=2)
        
        # Extract and save basic metrics
        metrics = {
            'layout_id': layout_id,
            'transmon_area_um2': float(w * h),
            'coupling_gap_um': float(gap),
            'resonator_length_um': float(length),
            'has_capacitor': has_cap,
            'gds_generated': gds_status == 'exported',
            'date_generated': datetime.now().isoformat()
        }
        
        metrics_file = os.path.join(layout_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return {
            'layout_id': layout_id,
            'status': 'success',
            'json_file': json_file,
            'gds_status': gds_status,
            'gds_file': gds_file,
            'directory': layout_dir,
            'has_capacitor': has_cap
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"  ❌ Error for {layout_id}: {str(e)}")
        return {
            'layout_id': layout_id,
            'status': 'failed',
            'error': str(e),
            'trace': error_trace
        }

def process_batch(batch_num, start_idx=0, num_layouts=None):
    """Process a batch of layouts"""
    
    # Setup paths
    params_file = f"../datasets/raw_simulations/layouts/batch_{str(batch_num).zfill(3)}_params.csv"
    output_base = f"../datasets/raw_simulations/layouts/real_batch_{str(batch_num).zfill(3)}"
    os.makedirs(output_base, exist_ok=True)
    
    # Load parameters
    df = pd.read_csv(params_file)
    
    # Slice if requested
    if num_layouts:
        df = df.iloc[start_idx:start_idx + num_layouts]
    
    print(f"\n📦 Batch {batch_num}: Processing {len(df)} layouts")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Batch {batch_num}"):
        params = row.to_dict()
        result = create_single_layout(params, output_base)
        results.append(result)
        
        # Small delay to prevent overwhelming
        time.sleep(0.1)
    
    # Save batch results
    results_file = os.path.join(output_base, "generation_results.json")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    cap_count = sum(1 for r in results if r.get('has_capacitor', False))
    
    summary = {
        'batch': batch_num,
        'total': len(results),
        'success': success_count,
        'failed': fail_count,
        'with_capacitor': cap_count,
        'success_rate': f"{(success_count/len(results))*100:.1f}%",
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Batch {batch_num} complete: {success_count}/{len(df)} successful")
    print(f"   Capacitors added: {cap_count}")
    return batch_num, summary

def main():
    print("="*60)
    print("QISKIT METAL LAYOUT GENERATOR v2 (Fixed)")
    print("="*60)
    print(f"Interdigital capacitor available: {HAS_INTERDIGITAL}")
    print("="*60)
    
    # For testing, process just first 10 layouts from batch 0
    print("\n🔬 TEST MODE: Processing first 10 layouts from batch 0")
    batch_num, summary = process_batch(0, start_idx=0, num_layouts=10)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Results: {summary['success']}/10 successful")
    
    if summary['success'] == 10:
        print("\n✅ All tests passed! Ready for full generation.")
        print("\nTo run full generation, use:")
        print("python -c 'from generate_layouts_fixed_v2 import process_batch; [process_batch(b) for b in range(10)]'")
    else:
        print("\n❌ Some layouts failed. Check errors above.")

if __name__ == "__main__":
    main()
