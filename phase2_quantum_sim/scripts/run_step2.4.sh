#!/bin/bash
# Run Step 2.4 - Validate System Parameters

echo "=================================================="
echo "STEP 2.4: Validate System Parameters"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if Step 2.3 completed
if [ ! -d ~/projects/qsymphony/results/phase2/wigner_baseline ]; then
    echo "⚠️ Step 2.3 not completed yet. Please complete Step 2.3 first."
    exit 1
fi

echo "✅ Step 2.3 data found"

# Run validation
echo ""
echo "🔍 Running system validation..."
time python scripts/validate_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 2.4 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Generated files:"
    ls -la ~/projects/qsymphony/results/phase2/validation/
    echo ""
    echo "Next: Step 2.5 - Prepare RL Environment Interface"
else
    echo "❌ Step 2.4 failed"
    exit 1
fi
echo "=================================================="
