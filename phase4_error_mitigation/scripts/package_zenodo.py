#!/usr/bin/env python
"""
Package all datasets for Zenodo submission
"""

import shutil
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

print("="*60)
print("Packaging datasets for Zenodo")
print("="*60)

# Create export directory
export_dir = Path('zenodo_export')
export_dir.mkdir(exist_ok=True)

# Create subdirectories
(export_dir / 'iq_data').mkdir(exist_ok=True)
(export_dir / 'trajectories').mkdir(exist_ok=True)
(export_dir / 'photocurrents').mkdir(exist_ok=True)
(export_dir / 'models').mkdir(exist_ok=True)
(export_dir / 'figures').mkdir(exist_ok=True)

# 1. Copy IQ datasets
iq_source = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'raw_simulations' / 'readout_data' / 'iq_data'
if iq_source.exists():
    for f in iq_source.glob('*.csv'):
        shutil.copy2(f, export_dir / 'iq_data' / f.name)
        print(f"✅ Copied: {f.name}")
else:
    print(f"⚠️ IQ data directory not found: {iq_source}")

# 2. Copy Table 1
table1 = Path('results/phase4/data/table1_readout_errors_final.csv')
if table1.exists():
    shutil.copy2(table1, export_dir / 'table1_readout_errors.csv')
    print("✅ Copied: table1_readout_errors.csv")

# 3. Copy PINN model
model = Path('results/models/pinn_gate_optimizer.zip')
if model.exists():
    shutil.copy2(model, export_dir / 'models' / model.name)
    print("✅ Copied: pinn_gate_optimizer.zip")

# 4. Copy figures
fig_source = Path('results/phase4/figures')
if fig_source.exists():
    for f in fig_source.glob('*.png'):
        shutil.copy2(f, export_dir / 'figures' / f.name)
        print(f"✅ Copied figure: {f.name}")

# 5. Create metadata
metadata = {
    "title": "Q-SYMPHONY Project Datasets",
    "description": "Complete datasets for autonomous entanglement engineering and topological error correction",
    "date": datetime.now().isoformat(),
    "size": {
        "iq_samples": 500000,
        "trajectories": 1000,
        "photocurrents": 1000
    },
    "files": {
        "iq_data": [f.name for f in (export_dir / 'iq_data').glob('*.csv')],
        "figures": [f.name for f in (export_dir / 'figures').glob('*.png')],
        "models": [f.name for f in (export_dir / 'models').glob('*.zip')]
    }
}

with open(export_dir / 'dataset_info.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Metadata saved: dataset_info.json")

# 6. Create README
readme = f"""
# Q-SYMPHONY Dataset

## Overview
This dataset contains all data from the Q-SYMPHONY project.

## Contents

### IQ Readout Data
- 500,000 IQ samples across 5 qubit variants (RQC Q2-Q6)
- Ground (|0⟩) and excited (|1⟩) states
- Dispersive shifts from -0.66 MHz to -1.74 MHz

### Trained Models
- `pinn_gate_optimizer.zip`: Physics-Informed Neural Network for CNOT gate

### Figures
All publication figures in PNG format:
- Figure 3a: Exceptional point visualization
- Figure 3b: Circuit depth comparison
- Confusion matrices for all qubits
- IQ scatter plots

## File Structure

## Citation
If using this data, please cite: [Your Paper]

## License
[Your License]

## Contact
[Your Email]
"""

with open(export_dir / 'README.md', 'w') as f:
    f.write(readme)

print("\n✅ README.md created")

# Size summary
total_size = sum(f.stat().st_size for f in export_dir.rglob('*') if f.is_file())
print(f"\n📊 Total package size: {total_size / 1024 / 1024:.1f} MB")
print(f"📁 Export directory: {export_dir.absolute()}")
print("\n✅ Packaging complete!")
