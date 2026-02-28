#!/usr/bin/env python
"""
Prepare parameter data for GNN training
(Without requiring Qiskit Metal)
"""

import pandas as pd
import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

print("="*60)
print("Preparing Parameter Data for GNN Training")
print("="*60)

# Load all parameters
df = pd.read_csv('../datasets/raw_simulations/layouts/all_parameters.csv')
print(f"Loaded {len(df)} parameter sets")

# Add derived features for GNN
df['aspect_ratio'] = df['transmon_width_um'] / df['transmon_height_um']
df['area_um2'] = df['transmon_width_um'] * df['transmon_height_um']
df['capacitance_estimate'] = 10.0 / df['coupling_gap_um']  # Rough estimate
df['frequency_estimate_ghz'] = 5.0 + (df['transmon_width_um'] - 200) / 100  # Rough estimate

# Normalize features for GNN
feature_columns = [
    'transmon_width_um', 'transmon_height_um', 'coupling_gap_um',
    'resonator_length_um', 'junction_area_nm2', 'substrate_thickness_um',
    'aspect_ratio', 'area_um2', 'capacitance_estimate', 'frequency_estimate_ghz'
]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[feature_columns])

# Save normalized features
np.save('../datasets/raw_simulations/layouts/gnn_features.npy', df_scaled)
np.save('../datasets/raw_simulations/layouts/gnn_labels.npy', df_scaled[:, :3])  # Use first 3 as labels for now

# Save feature names
with open('../datasets/raw_simulations/layouts/gnn_feature_names.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)

# Create graph edges (nearest neighbors in parameter space)
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(df_scaled)
distances, indices = nbrs.kneighbors(df_scaled)

# Save edge indices for GNN
edge_index = []
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:  # Skip self
        edge_index.append([i, j])

edge_index = np.array(edge_index).T
np.save('../datasets/raw_simulations/layouts/gnn_edge_index.npy', edge_index)

print(f"\n✅ GNN Data Prepared:")
print(f"   Features shape: {df_scaled.shape}")
print(f"   Edge index shape: {edge_index.shape}")
print(f"   Number of edges: {edge_index.shape[1]}")

# Save summary
summary = {
    'num_nodes': len(df),
    'num_features': len(feature_columns),
    'num_edges': edge_index.shape[1],
    'feature_names': feature_columns,
    'parameters': df.to_dict(orient='records')[0]  # Sample
}

with open('../datasets/raw_simulations/layouts/gnn_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n📁 Files saved:")
print("   - gnn_features.npy")
print("   - gnn_labels.npy")
print("   - gnn_edge_index.npy")
print("   - gnn_feature_names.json")
print("   - gnn_summary.json")
print("="*60)
