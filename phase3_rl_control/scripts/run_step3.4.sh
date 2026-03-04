#!/bin/bash
# Run Step 3.4-3.6 - Generate paper figures

echo "=================================================="
echo "STEP 3.4-3.6: Generate Paper Figures"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase3_rl_control/

# Check if measurement model exists
if [ ! -f ~/projects/qsymphony/results/models/ppo_measurement_final.zip ]; then
    echo "❌ Measurement model not found!"
    echo "Please complete Step 3.3 first."
    exit 1
fi

echo "✅ Measurement model found"

# Run evaluation and figure generation
echo ""
echo "[1/1] Generating paper figures..."
python scripts/evaluate_agent.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 3.4-3.6 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Figures generated:"
    ls -la ~/projects/qsymphony/results/phase3/figures/fig2*.png
    echo ""
    echo "Next: Step 3.7 - Benchmark Against Analytical Control"
else
    echo "❌ Figure generation failed"
    exit 1
fi
