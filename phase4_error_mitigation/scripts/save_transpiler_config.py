#!/usr/bin/env python
"""
Save transpiler configuration for reproducibility
"""

import json
from pathlib import Path

config = {
    "circuit_name": "parity_check_3qubit",
    "description": "3-qubit parity measurement circuit",
    "qiskit_version": "0.44.1",
    "transpiler_settings": {
        "basis_gates": ["cx", "u1", "u2", "u3"],
        "optimization_level": 3,
        "coupling_map": "linear_3qubit",
        "layout_method": "trivial",
        "routing_method": "stochastic",
        "seed_transpiler": 42
    },
    "results": {
        "qiskit_depth": 37,
        "ai_optimized_depth": 21,
        "improvement_percent": 43.2
    },
    "date": "2026-03-04"
}

# Save
output_path = Path('results/phase4/data/transpiler_config.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Transpiler config saved to: {output_path}")
