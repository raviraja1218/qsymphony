#!/bin/bash
# Run Step 2.5 - Prepare RL Environment Interface

echo "=================================================="
echo "STEP 2.5: Prepare RL Environment Interface"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if gymnasium is installed
python -c "import gymnasium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing gymnasium..."
    pip install gymnasium
fi

# Test the environment
echo ""
echo "🚀 Testing RL environment..."
python qsymphony_env.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 2.5 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Environment file: $(pwd)/qsymphony_env.py"
    echo "Test trajectory: test_trajectory.json"
    echo ""
    echo "✅✅✅ PHASE 2 COMPLETE! Ready for Phase 3! ✅✅✅"
    echo ""
    echo "Next: Phase 3 - Reinforcement Learning Control"
else
    echo "❌ Step 2.5 failed"
   
fi
echo "=================================================="
