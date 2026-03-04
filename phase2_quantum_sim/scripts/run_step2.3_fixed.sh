#!/bin/bash
# Run Step 2.3 - Compute Baseline Wigner Functions (Fixed path check)

echo "=================================================="
echo "STEP 2.3: Compute Baseline Wigner Functions"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if Step 2.2 completed - look for any trajectory file
if ls ~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/trajectory_*.pkl 1> /dev/null 2>&1; then
    TRAJ_COUNT=$(ls -1 ~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/trajectory_*.pkl 2>/dev/null | wc -l)
    echo "✅ Step 2.2 data found: $TRAJ_COUNT trajectory files"
else
    echo "❌ No trajectory files found in ~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/"
    echo "Please complete Step 2.2 first."
    exit 1
fi

# Check for metadata
if [ -f ~/Research/Datasets/qsymphony/raw_simulations/baseline_trajectories/metadata/trajectory_summary.json ]; then
    echo "✅ Metadata found"
else
    echo "⚠️ Metadata file not found, but continuing anyway"
fi

# Run Wigner computation
echo ""
echo "🚀 Computing Wigner functions..."
time python scripts/compute_wigner.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 2.3 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Generated files:"
    ls -la ~/projects/qsymphony/results/phase2/wigner_baseline/
    echo ""
    echo "Next: Step 2.4 - Validate System Parameters"
else
    echo "❌ Step 2.3 failed"
    exit 1
fi
echo "=================================================="
