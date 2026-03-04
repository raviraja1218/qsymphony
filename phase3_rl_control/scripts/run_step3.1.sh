#!/bin/bash
# Run Step 3.1 - Implement and test PPO with LSTM

echo "=================================================="
echo "STEP 3.1: Implement PPO with LSTM"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase3_rl_control/

# Install required packages if not present
pip install gymnasium pandas 2>/dev/null

# Check GPU availability
echo ""
echo "🎮 Checking GPU..."
python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

# Test network implementation
echo ""
echo "[1/4] Testing PPO network architecture..."
python -c "
from utils.ppo_network import PPOActorCritic, count_parameters
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PPOActorCritic(obs_dim=13, action_dim=2).to(device)
print(f'✅ Network created with {count_parameters(model):,} parameters')
# Test forward pass
obs = torch.randn(32, 10, 13).to(device)
action_mean, log_std, value, hidden = model(obs)
print(f'✅ Forward pass successful')
print(f'   Action mean shape: {action_mean.shape}')
print(f'   Value shape: {value.shape}')
"

# Test reward functions
echo ""
echo "[2/4] Testing reward functions..."
python utils/reward_functions.py

# Test environment wrapper
echo ""
echo "[3/4] Testing environment wrapper..."
python utils/environment_wrapper.py

# Test training script (dry run with small timesteps)
echo ""
echo "[4/4] Testing training script (dry run)..."
python scripts/train_ppo.py --mode oracle --timesteps 1000

echo ""
echo "=================================================="
echo "✅ STEP 3.1 COMPLETE"
echo "=================================================="
echo ""
echo "Next: Step 3.2 - Oracle Training (Full State Access)"
echo "Run: python scripts/train_ppo.py --mode oracle"
echo "=================================================="
