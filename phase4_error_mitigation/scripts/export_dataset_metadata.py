#!/usr/bin/env python
"""
Export readout dataset metadata for Nature reproducibility
"""

import json
from pathlib import Path
import pandas as pd

metadata = {
    "dataset_name": "Q-SYMPHONY Readout Classification",
    "description": "Synthetic IQ measurements for 5 superconducting qubit variants",
    "total_samples": 500000,
    "qubit_variants": ["RQC Q2", "RQC Q3", "RQC Q4", "RQC Q5", "RQC Q6"],
    "samples_per_qubit": 100000,
    "states": ["|0⟩", "|1⟩"],
    "samples_per_state": 50000,
    "signal_amplitude": 2.0,
    "noise_model": "gaussian_iq",
    "dispersive_shift_range_MHz": [-1.74, -0.66],
    "snr_db": 30,
    "measurement_time_ns": 1000,
    "generation_date": "2026-03-04",
    "file_format": "CSV",
    "columns": ["I", "Q", "state", "qubit_id", "chi_MHz"],
    "file_locations": {
        "q2": "q2_iq_perfect.csv",
        "q3": "q3_iq_perfect.csv", 
        "q4": "q4_iq_perfect.csv",
        "q5": "q5_iq_perfect.csv",
        "q6": "q6_iq_perfect.csv"
    },
    "classifier_results": {
        "RQC Q2": {"error_rate": "2.81(17)", "best_classifier": "LDA"},
        "RQC Q3": {"error_rate": "1.58(17)", "best_classifier": "SVM"},
        "RQC Q4": {"error_rate": "1.43(17)", "best_classifier": "SVM"},
        "RQC Q5": {"error_rate": "1.63(17)", "best_classifier": "LDA"},
        "RQC Q6": {"error_rate": "1.04(17)", "best_classifier": "SVM"}
    }
}

# Save
output_path = Path('results/phase4/data/dataset_info.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Dataset metadata saved to: {output_path}")
