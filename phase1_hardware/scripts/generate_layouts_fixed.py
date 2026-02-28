#!/usr/bin/env python
"""
Generate Qiskit Metal layouts using available components:
- TransmonPocket from qubits/
- Interdigital capacitors from lumped/
- Meandered resonators from tlines/
- Launchpad terminations from terminations/
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
        
        # Extract parameters
        w = float(params_dict['transmon_width_um'])
        h = float(params_dict['transmon_height_um'])
        gap = float(params_dict['coupling_gap_um'])
        length = float(params_dict['resonator_length_um'])
        
        # Add ground plane
        design.planar.add_gnd_polygon(layers=[1])
        
        # Create transmon qubit with built-in coupling pads
        transmon_options = dict(
            pos_x='0mm',
            pos_y='0mm',
            pocket_width=f'{w}um',
            pocket_height=f'{h}um',
            pocket_type='rectangular',
            inductor_width='10um',
            inductor_gap='10um',
            connection_pads=dict(
                readout=dict(loc='right', pad_width='50um', pad_height='50um'),
                drive=dict(loc='left', pad_width='50um', pad_height='50um'),
                flux=dict(loc='bottom', pad_width='50um', pad_height='50um')
            )
        )
        q1 = TransmonPocket(design, f'Q_{layout_id}', options=transmon_options)
        
        # Create coupling element (interdigital capacitor if available)
        if HAS_INTERDIGITAL:
            try:
                cap_options = dict(
                    pos_x=f'{gap}um',
                    pos_y='0mm',
                    finger_length='50um',
                    finger_width='5um',
                    finger_gap='5um',
                    n_fingers=5
                )
                cap = CapNInterdigital(design, f'C_{layout_id}', options=cap_options)
                design.connect(f'Q_{layout_id}.readout', f'C_{layout_id}.tline')
            except:
                # Fall back to direct connection
                design.connect(f'Q_{layout_id}.readout', f'R_{layout_id}.tline')
        else:
            # Direct connection from transmon to resonator
            design.connect(f'Q_{layout_id}.readout', f'R_{layout_id}.tline')
        
        # Create meandered resonator
        meander_options = dict(
            pos_x=f'{gap + 100}um',
            pos_y='0mm',
            total_length=f'{length}um',
            meander='sinusoidal',
            amplitude='100um',
            period='200um',
            width='10um',
            layer='1'
        )
        res = RouteMeander(design, f'R_{layout_id}', options=meander_options)
        
        # Create layout directory
        layout_dir = os.path.join(output_dir, layout_id)
        os.makedirs(layout_dir, exist_ok=True)
        
        # Save design as JSON
        json_file = os.path.join(layout_dir, f"{layout_id}.json")
        design.save(json_file)
        
        # Export GDS
        try:
            gds_file = os.path.join(layout_dir, f"{layout_id}.gds")
            design.export_gds(gds_file)
            gds_status = "exported"
        except Exception as e:
            gds_status = f"failed: {str(e)}"
            gds_file = None
        
        # Save parameters
        params_file = os.path.join(layout_dir, "parameters.json")
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        return {
            'layout_id': layout_id,
            'status': 'success',
            'json_file': json_file,
            'gds_status': gds_status,
            'gds_file': gds_file,
            'directory': layout_dir
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            'layout_id': layout_id,
            'status': 'failed',
            'error': str(e),
            'trace': error_trace
        }

def process_batch(batch_num):
    """Process one batch of layouts"""
    
    # Setup paths
    params_file = f"../datasets/raw_simulations/layouts/batch_{str(batch_num).zfill(3)}_params.csv"
    output_base = f"../datasets/raw_simulations/layouts/real_batch_{str(batch_num).zfill(3)}"
    os.makedirs(output_base, exist_ok=True)
    
    # Load parameters
    df = pd.read_csv(params_file)
    print(f"\n📦 Batch {batch_num}: Processing {len(df)} layouts")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Batch {batch_num}"):
        params = row.to_dict()
        result = create_single_layout(params, output_base)
        results.append(result)
        
        # Small delay
        time.sleep(0.05)
    
    # Save batch results
    results_file = os.path.join(output_base, "generation_results.json")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = sum(1 for r in results if r['status'] == 'failed')
    
    summary = {
        'batch': batch_num,
        'total': len(results),
        'success': success_count,
        'failed': fail_count,
        'success_rate': f"{(success_count/len(results))*100:.1f}%",
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Batch {batch_num} complete: {success_count}/{len(results)} successful")
    return batch_num, summary

def main():
    print("="*60)
    print("QISKIT METAL LAYOUT GENERATOR (Fixed Version)")
    print("="*60)
    print(f"Interdigital capacitor available: {HAS_INTERDIGITAL}")
    print("="*60)
    
    # Process all batches
    batches = range(10)
    all_results = []
    
    start_time = time.time()
    
    for batch in batches:
        batch_num, summary = process_batch(batch)
        all_results.append(summary)
        
        # Brief pause between batches
        if batch < 9:
            print("⏳ Cooling down for 10 seconds...")
            time.sleep(10)
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    # Overall summary
    total_success = sum(r['success'] for r in all_results)
    total_layouts = sum(r['total'] for r in all_results)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total layouts processed: {total_layouts}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_layouts - total_success}")
    print(f"Success rate: {(total_success/total_layouts)*100:.1f}%")
    print(f"Total time: {minutes}m {seconds}s")
    print("="*60)
    
    # Save master summary
    master_summary = {
        'date': datetime.now().isoformat(),
        'total_layouts': total_layouts,
        'successful': total_success,
        'failed': total_layouts - total_success,
        'batches': all_results
    }
    
    with open('../datasets/raw_simulations/layouts/master_summary.json', 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    print("\n📊 Master summary saved to: ../datasets/raw_simulations/layouts/master_summary.json")

if __name__ == "__main__":
    main()
