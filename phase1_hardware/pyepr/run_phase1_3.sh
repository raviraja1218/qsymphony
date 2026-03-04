#!/bin/bash
# Run Phase 1.3 and 1.4 - pyEPR simulations and parameter extraction

echo "=================================================="
echo "PHASE 1.3 & 1.4: pyEPR Simulations & Parameter Extraction"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to pyepr directory
cd ~/projects/qsymphony/phase1_hardware/pyepr/

# Step 1.3: Run pyEPR simulations
echo ""
echo "[1/4] Running pyEPR simulations on top 100 layouts..."
python run_pyepr_simulations.py

if [ $? -ne 0 ]; then
    echo "ERROR: pyEPR simulations failed"
    exit 1
fi

# Step 1.4: Extract parameters
echo ""
echo "[2/4] Extracting Hamiltonian parameters for Phase 2..."
python extract_parameters.py

if [ $? -ne 0 ]; then
    echo "ERROR: Parameter extraction failed"
    exit 1
fi

# Create Figure 1a
echo ""
echo "[3/4] Creating Figure 1a (3D schematic)..."
python create_figure_1a.py

# Create Figure 1c (will use heatmaps from simulations)
echo ""
echo "[4/4] Figure 1c already created during simulations"
echo "    Heatmaps saved in: ~/projects/qsymphony/results/phase1/epr_results/"

echo ""
echo "=================================================="
echo "PHASE 1 COMPLETE!"
echo "=================================================="
echo ""
echo "📁 Results:"
echo "  - EPR results: ~/projects/qsymphony/results/phase1/epr_results/"
echo "  - Figures: ~/projects/qsymphony/results/phase1/figures/"
echo "  - Phase 2 parameters: ~/projects/qsymphony/phase2_quantum_sim/hardware_params.json"
echo ""
echo "✅ Ready to proceed to PHASE 2!"
