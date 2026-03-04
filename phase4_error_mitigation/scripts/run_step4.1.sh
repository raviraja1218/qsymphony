#!/bin/bash
# Run Step 4.1-4.2 - Generate Readout Data and Train Classifiers

echo "=================================================="
echo "STEP 4.1-4.2: Generate Readout Data & Train Classifiers"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase4_error_mitigation/

# Install required packages if not present
pip install scikit-learn pandas matplotlib pyyaml tqdm 2>/dev/null

# Optional: Install cuML for GPU acceleration
# pip install cuml-cu11  # Uncomment if available

# Step 4.1: Generate readout data
echo ""
echo "[1/2] Generating synthetic readout data..."
python scripts/generate_readout_data.py

if [ $? -ne 0 ]; then
    echo "❌ Data generation failed"
    exit 1
fi

# Step 4.2: Train classifiers
echo ""
echo "[2/2] Training classifiers and generating Table 1..."
python scripts/train_classifiers.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 4.1-4.2 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Generated files:"
    ls -la ~/Research/Datasets/qsymphony/raw_simulations/readout_data/iq_data/
    echo ""
    echo "Table 1:"
    cat ~/projects/qsymphony/results/phase4/data/table1_readout_errors.csv
    echo ""
    echo "Next: Step 4.3 - Implement PINN for Gate Optimization"
else
    echo "❌ Classifier training failed"
    
fi
