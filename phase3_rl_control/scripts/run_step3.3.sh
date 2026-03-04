#!/bin/bash
# Run Step 3.3 - Measurement-Based Training

echo "=================================================="
echo "STEP 3.3: Measurement-Based Training"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase3_rl_control/

# Check if oracle model exists
if [ ! -f ~/projects/qsymphony/results/models/ppo_oracle_final.zip ]; then
    echo "❌ Oracle model not found!"
    echo "Please complete Step 3.2 first."
    exit 1
fi

echo "✅ Oracle model found"

# Extract golden path
echo ""
echo "[1/3] Extracting golden path..."
python scripts/extract_golden_path.py

# Run measurement training
echo ""
echo "[2/3] Starting measurement-based training..."
echo "This will take several hours. Using 100k timesteps for overnight run."
echo ""

# Ask for timesteps
read -p "Enter number of timesteps (default: 100000): " timesteps
if [ -z "$timesteps" ]; then
    timesteps=100000
fi

python scripts/train_measurement.py --timesteps $timesteps

if [ $? -eq 0 ]; then
    echo ""
    echo "[3/3] Training complete!"
    echo ""
    echo "=================================================="
    echo "✅ STEP 3.3 COMPLETE"
    echo "=================================================="
    echo ""
    echo "Models saved:"
    echo "  - ~/projects/qsymphony/results/models/ppo_measurement_final.zip"
    echo "  - Checkpoints in ~/Research/Datasets/qsymphony/raw_simulations/rl_training/checkpoints/"
    echo ""
    echo "Next: Step 3.4 - Generate Control Visualization"
else
    echo "❌ Training failed"
    exit 1
fi
