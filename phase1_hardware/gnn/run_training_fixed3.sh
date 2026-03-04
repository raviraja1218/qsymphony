#!/bin/bash
# Run GNN training pipeline with fixed paths

echo "=================================================="
echo "Symplectic GNN Training Pipeline (Fixed v3)"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to GNN directory
cd ~/projects/qsymphony/phase1_hardware/gnn/

# Step 1: Prepare dataset (already done, but we'll check)
echo ""
echo "[1/4] Checking dataset..."
ls -la ~/Research/Datasets/qsymphony/processed/processed/data_*.pt | head -5

# Step 2: Train model
echo ""
echo "[2/4] Training GNN..."
python train_sympgnn_fixed2.py

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

# Step 3: Select top layouts for pyEPR (already done in previous run)
echo ""
echo "[3/4] Top 100 layouts already selected"

echo ""
echo "=================================================="
echo "GNN Training Complete"
echo "=================================================="
