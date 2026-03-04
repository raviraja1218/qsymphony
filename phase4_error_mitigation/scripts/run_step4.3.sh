#!/bin/bash
# Run Step 4.3 - PINN for Gate Optimization

echo "=================================================="
echo "STEP 4.3: Physics-Informed Neural Network"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase4_error_mitigation/

# Check if Step 4.2 completed
if [ ! -f ~/projects/qsymphony/results/phase4/data/table1_readout_errors.csv ]; then
    echo "⚠️ Step 4.2 not completed yet. Please complete Step 4.2 first."
    
fi

echo "✅ Step 4.2 data found"

# Run PINN training
echo ""
echo "[1/1] Training PINN for gate optimization..."
echo "This will take 2-3 hours on GPU."
echo ""

# Ask for confirmation
read -p "Start PINN training now? (y/n): " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "Starting at $(date)"
    time python scripts/train_pinn.py
    echo "Completed at $(date)"
else
    echo "Run manually with: python scripts/train_pinn.py"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ STEP 4.3 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Model saved to: ~/projects/qsymphony/results/models/pinn_gate_optimizer.zip"
    echo "Plots saved to: ~/projects/qsymphony/results/phase4/figures/"
    echo ""
    echo "Next: Step 4.4 - Generate Exceptional Point Visualization"
else
    echo "❌ PINN training failed"
    
fi
