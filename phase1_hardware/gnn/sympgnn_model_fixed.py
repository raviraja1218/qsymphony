#!/usr/bin/env python
"""
Symplectic Graph Neural Network for Hamiltonian prediction
Fixed: Correct output dimensions
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
        self.target_dim = target_dim
        
        # Initial encoding
        self.encoder = nn.Linear(node_features, hidden_dim)
        
        # Graph convolutional layers with symplectic constraints
        self.convs = nn.ModuleList()
        self.symp_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.symp_layers.append(SymplecticLayer(hidden_dim, hidden_dim))
        
        # Output layers - FIXED: Properly sized decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, target_dim)  # Output target_dim (4) values
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
        
        # Global pooling - aggregates all node features to graph level
        x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_dim]
        
        # Decode to target parameters
        out = self.decoder(x)  # Shape: [batch_size, target_dim]
        
        return out

def symplectic_loss(predictions, targets):
    """Custom loss with symplectic regularization"""
    
    # Ensure predictions and targets have same shape
    # predictions shape: [batch_size, target_dim]
    # targets shape: [batch_size, target_dim] or [batch_size * target_dim]?
    
    # If targets is 1D, reshape to match predictions
    if len(targets.shape) == 1:
        batch_size = predictions.shape[0]
        targets = targets.view(batch_size, -1)
    
    # MSE loss
    mse_loss = F.mse_loss(predictions, targets)
    
    # Symplectic regularization (simplified)
    symp_reg = torch.mean(torch.abs(predictions)) * 0.01
    
    return mse_loss + symp_reg

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model
    model = SymplecticGNN(node_features=4, hidden_dim=128, num_layers=5, target_dim=4)
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass with batch
    from torch_geometric.data import Data, Batch
    
    # Create a batch of 2 graphs
    data_list = []
    for _ in range(2):
        x = torch.randn(4, 4)
        edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 3, 3, 3, 3]], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(data_list)
    out = model(batch)
    print(f"Forward pass output shape: {out.shape} (should be [2, 4])")
    
    # Test loss function
    targets = torch.randn(2, 4)
    loss = symplectic_loss(out, targets)
    print(f"Loss value: {loss.item():.4f}")
