#!/bin/bash
# Step 1.1 Execution Script

echo "=================================================="
echo "PHASE 1 - STEP 1.1: Generate Layout Candidates"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase1_hardware/

# Step 1: Generate layouts
echo ""
echo "[1/3] Generating 10,000 layouts..."
python scripts/generate_layouts.py

# Check if successful
if [ $? -ne 0 ]; then
    echo "ERROR: Layout generation failed"
    exit 1
fi

# Step 2: Analyze parameters
echo ""
echo "[2/3] Analyzing parameter distributions..."
python scripts/analyze_parameters.py

# Step 3: Create visualizations
echo ""
echo "[3/3] Creating layout previews..."
python scripts/visualize_layouts.py

echo ""
echo "=================================================="
echo "STEP 1.1 COMPLETE"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Layouts: ~/Research/Datasets/qsymphony/raw_simulations/layouts/raw_layouts/"
echo "  - Index: ~/Research/Datasets/qsymphony/raw_simulations/layouts/layouts_index.csv"
echo "  - Figures: ~/projects/qsymphony/results/phase1/figures/"
echo "  - Statistics: ~/projects/qsymphony/results/phase1/data/parameter_statistics.csv"
echo ""
