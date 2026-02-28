#!/usr/bin/env python
"""Final verification of all generated layouts"""

import os
import json
import glob

def verify_all():
    base_dir = "../datasets/raw_simulations/layouts/"
    
    # Find all real batch directories
    real_batches = glob.glob(os.path.join(base_dir, "real_batch_*"))
    
    print("="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    
    total_layouts = 0
    total_json = 0
    total_gds = 0
    total_capacitors = 0
    
    for batch_dir in sorted(real_batches):
        batch_name = os.path.basename(batch_dir)
        
        # Find all layout directories
        layouts = glob.glob(os.path.join(batch_dir, "L*"))
        
        batch_json = 0
        batch_gds = 0
        batch_cap = 0
        
        for layout_dir in layouts:
            layout_id = os.path.basename(layout_dir)
            
            # Check JSON
            if os.path.exists(os.path.join(layout_dir, f"{layout_id}.json")):
                batch_json += 1
            
            # Check GDS
            if os.path.exists(os.path.join(layout_dir, f"{layout_id}.gds")):
                batch_gds += 1
            
            # Check metrics for capacitor
            metrics_file = os.path.join(layout_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if metrics.get('has_capacitor', False):
                        batch_cap += 1
        
        print(f"\n{batch_name}:")
        print(f"  Layouts: {len(layouts)}")
        print(f"  JSON: {batch_json}")
        print(f"  GDS: {batch_gds}")
        print(f"  With Capacitors: {batch_cap}")
        
        total_layouts += len(layouts)
        total_json += batch_json
        total_gds += batch_gds
        total_capacitors += batch_cap
    
    print("\n" + "="*60)
    print(f"TOTAL LAYOUTS: {total_layouts}")
    print(f"TOTAL JSON: {total_json}")
    print(f"TOTAL GDS: {total_gds}")
    print(f"GDS RATE: {(total_gds/total_layouts)*100:.1f}%")
    print(f"CAPACITORS: {total_capacitors}")
    print("="*60)
    
    # Check master summary
    summary_file = os.path.join(base_dir, "master_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"\nMaster summary reports: {summary.get('successful', 0)} successful")

if __name__ == "__main__":
    verify_all()
