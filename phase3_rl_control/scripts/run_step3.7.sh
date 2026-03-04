#!/bin/bash
# Run Step 3.7 - Benchmark Against Analytical Control

echo "=================================================="
echo "STEP 3.7: Benchmark Against Analytical Control"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase3_rl_control/

# Check if RL trajectory exists
if [ -z "$(ls -A ~/projects/qsymphony/results/phase3/trajectories/)" ]; then
    echo "❌ No RL trajectory found!"
    echo "Please complete Step 3.4-3.6 first."
    exit 1
fi

echo "✅ RL trajectory found"

# Run benchmark
echo ""
echo "[1/1] Running benchmark comparison..."
python scripts/benchmark_control.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 3.7 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Benchmark results:"
    ls -la ~/projects/qsymphony/results/phase3/data/benchmark_*
    echo ""
    echo "✅✅✅ PHASE 3 COMPLETE! ✅✅✅"
    echo ""
    echo "Next: Phase 4 - Error Mitigation & Readout Classification"
else
    echo "❌ Benchmark failed"
    exit 1
fi
