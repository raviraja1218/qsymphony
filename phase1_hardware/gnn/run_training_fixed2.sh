#!/bin/bash
# Run GNN training pipeline with fixed scripts

echo "=================================================="
echo "Symplectic GNN Training Pipeline (Fixed v2)"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to GNN directory
cd ~/projects/qsymphony/phase1_hardware/gnn/

# Step 0: Test first
echo ""
echo "[0/4] Testing setup..."
python test_prepare.py

if [ $? -ne 0 ]; then
    echo "ERROR: Test failed"
    exit 1
fi

# Step 1: Prepare dataset
echo ""
echo "[1/4] Preparing dataset..."
python prepare_dataset_fixed2.py

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset preparation failed"
    exit 1
fi

# Step 2: Train model
echo ""
echo "[2/4] Training GNN..."
python train_sympgnn_fixed.py

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed"
    exit 1
fi

# Step 3: Select top layouts for pyEPR
echo ""
echo "[3/4] Selecting top layouts for pyEPR..."
python select_top_layouts.py

echo ""
echo "=================================================="
echo "GNN Training Complete"
echo "=================================================="
