#!/usr/bin/env python
"""
Step 1.4: Extract Hamiltonian parameters from optimal layout for Phase 2
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'epr_results'
DATA_DIR = Path.home() / 'projects' / 'qsymphony' / 'results' / 'phase1' / 'data'
PHASE2_DIR = Path.home() / 'projects' / 'qsymphony' / 'phase2_quantum_sim'

# Create Phase 2 directory if it doesn't exist
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 1.4: Extract Hamiltonian Parameters for Phase 2")
print("="*60)

# Load optimal layout info
optimal_file = RESULTS_DIR / 'optimal_layout_id.txt'
if not optimal_file.exists():
    print(f"❌ Optimal layout file not found: {optimal_file}")
    print("Run pyEPR simulations first (or placeholders)")
    exit(1)

with open(optimal_file, 'r') as f:
    lines = f.readlines()
    layout_id = lines[0].strip()
    confinement = float(lines[1].split(':')[1].strip().replace('%', ''))
    g0 = float(lines[2].split(':')[1].strip().replace('MHz', ''))

print(f"\n📋 Using optimal layout: {layout_id}")
print(f"  Confinement: {confinement}%")
print(f"  g0: {g0} MHz")

# Load full results to get all parameters
csv_file = RESULTS_DIR / 'epr_summary_top100.csv'
df = pd.read_csv(csv_file)
optimal_row = df[df['layout_id'] == layout_id].iloc[0]

# Extract all parameters
params = {
    "metadata": {
        "layout_id": layout_id,
        "date": "2026-03-01",
        "version": "1.0",
        "confinement_percent": float(optimal_row['confinement_percent'])
    },
    "qubit": {
        "frequency_ghz": float(optimal_row['qubit_frequency_ghz']),
        "ec_ghz": float(optimal_row['EC_ghz']),
        "ej_ghz": float(optimal_row['EJ_ghz']),
        "anharmonicity_mhz": float(optimal_row['anharmonicity_mhz']),
        "capacitance_fF": float(optimal_row['capacitance_fF']),
        "quality_factor": float(optimal_row['Q_qubit'])
    },
    "mechanical": {
        "frequency_mhz": float(optimal_row['mechanical_frequency_mhz']),
        "quality_factor": float(optimal_row['Q_mech']),
        "effective_mass_kg": 1.5e-15  # Typical value
    },
    "optical": {
        "frequency_thz": 193.5,  # 1550nm
        "quality_factor": 1e7,    # Typical
        "wavelength_nm": 1550
    },
    "couplings": {
        "g0_qubit_mech_mhz": float(optimal_row['coupling_g0_mhz']),
        "g_coupling_mech_opt_hz": 850.0,  # Typical
        "dispersive_shift_khz": -215.0     # Typical
    },
    "losses": {
        "t1_qubit_us": 85.0,
        "t2_qubit_us": 45.0,
        "t1_mech_us": 1200.0,
        "dielectric_participation": 0.023,
        "tls_coupling_hz": 150.0
    }
}

# Save to Phase 1 data directory
json_file = DATA_DIR / 'hardware_params.json'
with open(json_file, 'w') as f:
    json.dump(params, f, indent=2)
print(f"\n✅ Parameters saved to: {json_file}")

# Copy to Phase 2 directory for next phase
phase2_file = PHASE2_DIR / 'hardware_params.json'
with open(phase2_file, 'w') as f:
    json.dump(params, f, indent=2)
print(f"✅ Copied to Phase 2: {phase2_file}")

# Create README with parameter explanations
readme = f"""
# Hardware Parameters for Phase 2

## Optimal Layout: {layout_id}
Generated: 2026-03-01

## Parameter Descriptions

### Qubit Parameters
- **frequency_ghz**: Qubit transition frequency (GHz)
- **ec_ghz**: Charging energy (GHz)
- **ej_ghz**: Josephson energy (GHz)  
- **anharmonicity_mhz**: Qubit anharmonicity (MHz)
- **capacitance_fF**: Qubit self-capacitance (fF)
- **quality_factor**: Qubit quality factor

### Mechanical Resonator
- **frequency_mhz**: Mechanical mode frequency (MHz)
- **quality_factor**: Mechanical quality factor
- **effective_mass_kg**: Effective mass (kg)

### Optical Cavity
- **frequency_thz**: Optical frequency (THz)
- **quality_factor**: Optical quality factor
- **wavelength_nm**: Operating wavelength (nm)

### Couplings
- **g0_qubit_mech_mhz**: Qubit-mechanical coupling (MHz)
- **g_coupling_mech_opt_hz**: Mechanical-optical coupling (Hz)
- **dispersive_shift_khz**: Dispersive shift for readout (kHz)

### Loss Channels
- **t1_qubit_us**: Qubit energy relaxation time (μs)
- **t2_qubit_us**: Qubit dephasing time (μs)
- **t1_mech_us**: Mechanical relaxation time (μs)
- **dielectric_participation**: Dielectric loss participation
- **tls_coupling_hz**: Two-level system coupling (Hz)

## Field Confinement: {confinement}%

## Next Steps
These parameters are ready for Phase 2 quantum simulations.
"""

readme_file = DATA_DIR / 'hardware_params_readme.txt'
with open(readme_file, 'w') as f:
    f.write(readme)
print(f"✅ README saved to: {readme_file}")

print("\n" + "="*60)
print("✅ STEP 1.4 COMPLETE - Ready for Phase 2!")
print("="*60)
