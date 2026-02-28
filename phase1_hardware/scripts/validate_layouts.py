#!/usr/bin/env python
"""
Validate all generated layouts
Count files, check completeness, generate summary
"""

import os
import json
import pandas as pd
from collections import Counter

def validate_batch(batch_num):
    """Validate a single batch"""
    
    batch_dir = f"../datasets/raw_simulations/layouts/batch_{str(batch_num).zfill(3)}"
    
    if not os.path.exists(batch_dir):
        return {'batch': batch_num, 'exists': False}
    
    # Count JSON files
    json_files = [f for f in os.listdir(batch_dir) if f.endswith('.json') and 'L' in f]
    gds_files = [f for f in os.listdir(batch_dir) if f.endswith('.gds')]
    mock_files = [f for f in os.listdir(batch_dir) if f.endswith('_mock.json')]
    
    # Check results file
    results_file = os.path.join(batch_dir, 'generation_results.json')
    results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    
    return {
        'batch': batch_num,
        'exists': True,
        'json_files': len(json_files),
        'gds_files': len(gds_files),
        'mock_files': len(mock_files),
        'results': results.get('success', 0),
        'failed': results.get('failed', 0)
    }

def main():
    print("="*60)
    print("Validating All Layout Batches")
    print("="*60)
    
    all_results = []
    total_jsons = 0
    total_gds = 0
    
    for batch in range(10):
        result = validate_batch(batch)
        all_results.append(result)
        
        if result['exists']:
            print(f"Batch {batch}:")
            print(f"  JSON files: {result['json_files']}")
            print(f"  GDS files: {result['gds_files']}")
            print(f"  Mock files: {result['mock_files']}")
            print(f"  Reported success: {result['results']}")
            print(f"  Reported failed: {result['failed']}")
            print()
            
            total_jsons += result['json_files']
            total_gds += result['gds_files']
    
    print("="*60)
    print(f"TOTAL JSON layouts: {total_jsons}/10000")
    print(f"TOTAL GDS layouts: {total_gds}/10000")
    print("="*60)
    
    # Save validation report
    report = {
        'total_expected': 10000,
        'total_json': total_jsons,
        'total_gds': total_gds,
        'completeness': total_jsons / 10000 * 100,
        'batches': all_results,
        'validation_date': pd.Timestamp.now().isoformat()
    }
    
    with open('logs/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Validation report saved to: logs/validation_report.json")

if __name__ == "__main__":
    main()
