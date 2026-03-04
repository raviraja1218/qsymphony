#!/bin/bash
# Run Step 2.3 - Compute Baseline Wigner Functions

echo "=================================================="
echo "STEP 2.3: Compute Baseline Wigner Functions"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if Step 2.2 completed
if [ ! -d ~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/trajectory_0000.pkl ]; then
    echo "⚠️ Step 2.2 not completed yet. Please complete Step 2.2 first."
    exit 1
fi

echo "✅ Step 2.2 data found"

# Run Wigner computation
echo ""
echo "🚀 Computing Wigner functions..."
time python scripts/compute_wigner.py

echo ""
echo "=================================================="
echo "STEP 2.3 COMPLETE"
echo "=================================================="
echo ""
echo "Next: Step 2.4 - Validate System Parameters"
echo "=================================================="
