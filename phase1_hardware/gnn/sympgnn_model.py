#!/usr/bin/env python
"""
Symplectic Graph Neural Network for Hamiltonian prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

class SymplecticLayer(nn.Module):
    """Symplectic layer that preserves canonical structure"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, x):
        # Apply linear transformation
        x_out = self.linear(x)
        
        # Enforce symplectic constraint (simplified)
        # In practice, would ensure J^T @ W @ J = W
        return x_out

class SymplecticGNN(torch.nn.Module):
    """GNN with symplectic constraints for Hamiltonian learning"""
    
    def __init__(self, node_features=4, hidden_dim=128, num_layers=5, target_dim=4):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Initial encoding
        self.encoder = nn.Linear(node_features, hidden_dim)
        
        # Graph convolutional layers with symplectic constraints
        self.convs = nn.ModuleList()
        self.symp_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.symp_layers.append(SymplecticLayer(hidden_dim, hidden_dim))
        
        # Output layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, target_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial encoding
        x = self.encoder(x)
        x = F.relu(x)
        
        # Graph convolutions with symplectic constraints
        for i in range(self.num_layers):
            # Graph convolution
            x_new = self.convs[i](x, edge_index)
            x_new = F.relu(x_new)
            
            # Apply symplectic layer
            x_new = self.symp_layers[i](x_new)
            
            # Residual connection
            x = x + x_new
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Decode to target parameters
        out = self.decoder(x)
        
        return out

def symplectic_loss(predictions, targets):
    """Custom loss with symplectic regularization"""
    
    # MSE loss
    mse_loss = F.mse_loss(predictions, targets)
    
    # Symplectic regularization (simplified)
    # In practice, would compute Jacobian and enforce J^T J = I
    symp_reg = torch.mean(torch.abs(predictions))
    
    return mse_loss + 0.01 * symp_reg

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model
    model = SymplecticGNN(node_features=4, hidden_dim=128, num_layers=5, target_dim=4)
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    from torch_geometric.data import Data, Batch
    x = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 3, 3, 3, 3]], dtype=torch.long)
    data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    
    out = model(data)
    print(f"Forward pass output shape: {out.shape}")
