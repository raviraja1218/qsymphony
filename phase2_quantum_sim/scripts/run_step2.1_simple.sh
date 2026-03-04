#!/bin/bash
# Run Step 2.1 with simplified approach

echo "=================================================="
echo "STEP 2.1: SME Implementation (Simplified)"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if hardware parameters exist
if [ ! -f hardware_params.json ]; then
    echo "❌ hardware_params.json not found!"
    exit 1
fi

echo "✅ Hardware parameters found"

# Run simplified SME
echo ""
echo "Running simplified SME solver..."
python scripts/setup_sme_simple.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 2.1 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Results saved to:"
    echo "  - Verification plot: ~/projects/qsymphony/results/phase2/validation/sme_verification.png"
    echo "  - Solver object: ~/projects/qsymphony/results/phase2/validation/sme_solver.pkl"
    echo ""
    echo "Next: Step 2.2 - Generate 1000 Baseline Trajectories"
else
    echo "❌ Step 2.1 failed"
    exit 1
fi
