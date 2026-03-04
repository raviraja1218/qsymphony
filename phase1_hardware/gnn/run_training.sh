#!/bin/bash
# Run GNN training pipeline

echo "=================================================="
echo "Symplectic GNN Training Pipeline"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to GNN directory
cd ~/projects/qsymphony/phase1_hardware/gnn/

# Step 1: Prepare dataset
echo ""
echo "[1/3] Preparing dataset..."
python prepare_dataset.py

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed"
    exit 1
fi

# Step 2: Train model
echo ""
echo "[2/3] Training GNN..."
python train_sympgnn.py

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

# Step 3: Select top layouts for pyEPR
echo ""
echo "[3/3] Selecting top layouts for pyEPR..."
python select_top_layouts.py

echo ""
echo "=================================================="
echo "GNN Training Complete"
echo "=================================================="
