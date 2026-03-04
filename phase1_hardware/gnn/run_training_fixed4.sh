#!/bin/bash
# Run GNN training pipeline with fixed model

echo "=================================================="
echo "Symplectic GNN Training Pipeline (Fixed v4)"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to GNN directory
cd ~/projects/qsymphony/phase1_hardware/gnn/

# Step 1: Check dataset
echo ""
echo "[1/3] Checking dataset..."
ls -la ~/Research/Datasets/qsymphony/processed/processed/data_*.pt | head -5

# Step 2: Train model with fixed architecture
echo ""
echo "[2/3] Training GNN with fixed model..."
python train_sympgnn_fixed3.py

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "GNN Training Complete"
echo "=================================================="
