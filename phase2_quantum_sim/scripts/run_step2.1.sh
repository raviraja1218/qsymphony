#!/bin/bash
# Run Step 2.1 - SME Implementation and Verification

echo "=================================================="
echo "STEP 2.1: Implement Stochastic Master Equation"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if QuTiP is installed
python -c "import qutip" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing QuTiP..."
    pip install qutip
fi

# Check if hardware parameters exist
if [ ! -f hardware_params.json ]; then
    echo "❌ hardware_params.json not found!"
    echo "Please complete Phase 1 first."
    exit 1
fi

echo "✅ Hardware parameters found:"
cat hardware_params.json | head -10

# Run the SME implementation
echo ""
echo "[1/2] Running SME solver with verification..."
python scripts/setup_sme.py

if [ $? -ne 0 ]; then
    echo "❌ SME implementation failed"
    exit 1
fi

# Run quick test
echo ""
echo "[2/2] Running quick test..."
python scripts/test_sme.py

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
