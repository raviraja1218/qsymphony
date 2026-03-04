#!/bin/bash
# Run Step 2.1 - SME Implementation and Verification (Fixed)

echo "=================================================="
echo "STEP 2.1: Implement Stochastic Master Equation (Fixed)"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if hardware parameters exist
if [ ! -f hardware_params.json ]; then
    echo "❌ hardware_params.json not found!"
    echo "Please complete Phase 1 first."
    exit 1
fi

echo "✅ Hardware parameters found:"
cat hardware_params.json | head -10

# Run the fixed SME implementation
echo ""
echo "[1/2] Running fixed SME solver with verification..."
python scripts/setup_sme_fixed.py

if [ $? -ne 0 ]; then
    echo "❌ SME implementation failed"
    exit 1
fi

# Run quick test
echo ""
echo "[2/2] Running quick test..."
python scripts/test_sme_fixed.py

echo ""
echo "=================================================="
echo "STEP 2.1 COMPLETE"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - Verification plot: ~/projects/qsymphony/results/phase2/validation/sme_verification.png"
echo "  - Solver object: ~/projects/qsymphony/results/phase2/validation/sme_solver.pkl"
echo ""
echo "Next: Step 2.2 - Generate 1000 Baseline Trajectories"
echo "=================================================="
