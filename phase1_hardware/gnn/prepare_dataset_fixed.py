#!/usr/bin/env python
"""
Prepare layout dataset for GNN training
Convert JSON layouts to graph representations
Fixed: All nodes have same feature dimension
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
from pathlib import Path
import yaml
from tqdm import tqdm
import pickle

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase1_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def expand_path(path):
    return str(Path(os.path.expanduser(path)).expanduser())

# Paths
LAYOUTS_DIR = Path(expand_path(config['paths']['layouts_raw']))
INDEX_FILE = LAYOUTS_DIR.parent / 'layouts_index.csv'
PROCESSED_DIR = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class LayoutDataset(Dataset):
    """PyTorch Geometric dataset from layout JSONs"""
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = []
        
    @property
    def raw_file_names(self):
        return ['layouts_index.csv']
    
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(10000)]
    
    def download(self):
        pass
    
    def process(self):
        """Convert layouts to graph data"""
        
        # Load index
        df = pd.read_csv(INDEX_FILE)
        df_valid = df[df['valid'] == True]
        
        print(f"Processing {len(df_valid)} layouts...")
        
        for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
            # Create node features with consistent dimension (all 4 features)
            node_features = []
            
            # Node 0: Transmon (4 features)
            transmon_features = [
                row['junction_width_nm'] / 500.0,  # Normalize to [0,1]
                row['junction_length_nm'] / 300.0,
                row['pad_area_um2'] / 200.0,
                row['gap_to_ground_um'] / 50.0,
            ]
            node_features.append(transmon_features)
            
            # Node 1: Capacitor (pad with zeros to make 4 features)
            cap_features = [
                row['finger_length_um'] / 100.0,
                row['finger_width_um'] / 10.0,
                row['finger_count'] / 20.0,
                row['finger_gap_um'] / 8.0,
            ]
            node_features.append(cap_features)
            
            # Node 2: Resonator (pad with zeros to make 4 features)
            res_features = [
                row['hbar_thickness_um'] / 20.0,
                row['beam_length_um'] / 500.0,
                row['beam_width_um'] / 30.0,
                0.0  # Padding to make 4 features
            ]
            node_features.append(res_features)
            
            # Node 3: Ground (all zeros, 4 features)
            ground_features = [0.0, 0.0, 0.0, 0.0]
            node_features.append(ground_features)
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Create edges (connections between components)
            # Transmon <-> Capacitor
            # Capacitor <-> Resonator
            # All connected to ground
            edge_index = torch.tensor([
                [0, 1, 2, 0, 1, 2],  # Source nodes
                [1, 2, 3, 3, 3, 3],  # Target nodes
            ], dtype=torch.long)
            
            # Target values (Hamiltonian parameters - we'll use approximations for now)
            # In real scenario, these would come from pyEPR
            y = torch.tensor([
                5.0,  # qubit freq (GHz)
                0.215,  # EC (GHz)
                12.45,  # EJ (GHz)
                10.0,  # g0 (MHz)
            ], dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            
            # Save
            torch.save(data, self.processed_dir / f'data_{idx:06d}.pt')
    
    def len(self):
        return 10000
    
    def get(self, idx):
        data = torch.load(self.processed_dir / f'data_{idx:06d}.pt')
        return data

def main():
    print("="*60)
    print("Preparing Layout Dataset for GNN Training")
    print("="*60)
    
    # Create dataset
    dataset = LayoutDataset(PROCESSED_DIR)
    
    print(f"\n✅ Dataset prepared with {len(dataset)} samples")
    print(f"   Node features: {dataset[0].x.shape[1]} dimensions (all consistent)")
    print(f"   Edge connections: {dataset[0].edge_index.shape[1]}")
    print(f"   Target values: {dataset[0].y.shape[0]}")
    
    # Save dataset info
    info = {
        'num_samples': len(dataset),
        'node_features': dataset[0].x.shape[1],
        'num_nodes': dataset[0].x.shape[0],
        'num_edges': dataset[0].edge_index.shape[1],
        'target_dim': dataset[0].y.shape[0],
    }
    
    with open(PROCESSED_DIR / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n📊 Dataset info saved to: {PROCESSED_DIR / 'dataset_info.json'}")

if __name__ == "__main__":
    main()
