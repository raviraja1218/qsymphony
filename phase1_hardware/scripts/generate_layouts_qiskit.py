#!/usr/bin/env python
"""
Generate actual Qiskit Metal layouts from parameter files
Processes batches in parallel
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import time

# Import Qiskit Metal
try:
    from qiskit_metal import designs, draw
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
    from qiskit_metal.qlibrary.terminations.lumped_capacitor import LumpedCapacitor
    from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
    from qiskit_metal.qlibrary.tlines.rectangular_waveguide import RectangularWaveguide
    QISKIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Qiskit Metal not fully available: {e}")
    QISKIT_AVAILABLE = False

def create_single_layout(params_dict, output_dir):
    """Create one chip layout from parameters"""
    
    if not QISKIT_AVAILABLE:
        # Mock mode for testing without Qiskit
        return mock_layout(params_dict, output_dir)
    
    try:
        # Create design
        design = designs.DesignPlanar()
        
        # Extract parameters
        layout_id = params_dict['layout_id']
        w = float(params_dict['transmon_width_um'])
        h = float(params_dict['transmon_height_um'])
        gap = float(params_dict['coupling_gap_um'])
        length = float(params_dict['resonator_length_um'])
        
        # Add ground plane
        design.planar.add_gnd_polygon(layers=[1])
        
        # Create transmon qubit
        transmon_options = dict(
            pos_x='0mm',
            pos_y='0mm',
            pocket_width=f'{w}um',
            pocket_height=f'{h}um',
            pocket_type='rectangular',
            inductor_width='10um',
            inductor_gap='10um',
            connection_pads=dict(
                readout=dict(loc='right', pad_width='50um', pad_height='50um')
            )
        )
        q1 = TransmonPocket(design, f'Q_{layout_id}', options=transmon_options)
        
        # Create coupling capacitor
        cap_options = dict(
            pos_x=f'{gap}um',
            pos_y='0mm',
            width='20um',
            gap='5um',
            orientation='0'
        )
        cap = LumpedCapacitor(design, f'C_{layout_id}', options=cap_options)
        
        # Create meandered resonator (acoustic mode)
        meander_options = dict(
            pos_x=f'{gap + 50}um',
            pos_y='0mm',
            total_length=f'{length}um',
            meander='sinusoidal',
            amplitude='100um',
            period='200um',
            width='10um',
            layer='1'
        )
        res = RouteMeander(design, f'R_{layout_id}', options=meander_options)
        
        # Connect components
        design.connect(f'Q_{layout_id}.tline', f'C_{layout_id}.tline')
        design.connect(f'C_{layout_id}.tline', f'R_{layout_id}.tline')
        
        # Save design
        layout_file = os.path.join(output_dir, f"{layout_id}.json")
        design.save(layout_file)
        
        # Also save GDS if possible
        try:
            gds_file = os.path.join(output_dir, f"{layout_id}.gds")
            design.export_gds(gds_file)
        except:
            pass
        
        return {
            'layout_id': layout_id,
            'status': 'success',
            'file': layout_file,
            'components': ['transmon', 'capacitor', 'resonator']
        }
        
    except Exception as e:
        return {
            'layout_id': layout_id,
            'status': 'failed',
            'error': str(e)
        }

def mock_layout(params_dict, output_dir):
    """Mock layout generator for testing"""
    layout_id = params_dict['layout_id']
    
    # Create mock JSON
    mock_data = {
        'layout_id': layout_id,
        'parameters': params_dict,
        'mock': True,
        'components': [
            {'type': 'transmon', 'width_um': params_dict['transmon_width_um']},
            {'type': 'capacitor', 'gap_um': params_dict['coupling_gap_um']},
            {'type': 'resonator', 'length_um': params_dict['resonator_length_um']}
        ]
    }
    
    mock_file = os.path.join(output_dir, f"{layout_id}_mock.json")
    with open(mock_file, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    return {
        'layout_id': layout_id,
        'status': 'mock',
        'file': mock_file
    }

def process_batch(batch_num):
    """Process one batch of layouts"""
    
    # Setup paths
    params_file = f"../datasets/raw_simulations/layouts/batch_{str(batch_num).zfill(3)}_params.csv"
    output_dir = f"../datasets/raw_simulations/layouts/batch_{str(batch_num).zfill(3)}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters
    df = pd.read_csv(params_file)
    print(f"Batch {batch_num}: Processing {len(df)} layouts")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Batch {batch_num}"):
        params = row.to_dict()
        result = create_single_layout(params, output_dir)
        results.append(result)
        
        # Small delay to prevent overwhelming
        time.sleep(0.01)
    
    # Save batch results
    results_file = os.path.join(output_dir, "generation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'batch': batch_num,
            'total': len(results),
            'success': sum(1 for r in results if r['status'] in ['success', 'mock']),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        }, f, indent=2)
    
    print(f"Batch {batch_num} complete: {len(results)} layouts")
    return batch_num, results

def main():
    print("="*60)
    print("Qiskit Metal Layout Generator")
    print("="*60)
    
    print(f"Qiskit Metal available: {QISKIT_AVAILABLE}")
    if not QISKIT_AVAILABLE:
        print("Running in MOCK mode - generating placeholder JSON files")
    
    # Process all batches
    batches = range(10)
    results = []
    
    # Use multiprocessing for parallel batch generation
    with mp.Pool(processes=4) as pool:
        for batch_result in tqdm(pool.imap(process_batch, batches), total=len(batches)):
            results.append(batch_result)
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    
    total_layouts = 0
    total_success = 0
    
    for batch_num, batch_results in results:
        total_layouts += len(batch_results)
        success = sum(1 for r in batch_results if r['status'] in ['success', 'mock'])
        total_success += success
        print(f"Batch {batch_num}: {success}/{len(batch_results)} successful")
    
    print("="*60)
    print(f"TOTAL: {total_success}/{total_layouts} layouts generated")
    print(f"Output directory: ~/Research/Datasets/qsymphony/raw_simulations/layouts/")
    print("="*60)

if __name__ == "__main__":
    main()
