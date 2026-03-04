#!/bin/bash
# Run Step 2.2 - Generate 1000 Baseline Trajectories (Fixed)

echo "=================================================="
echo "STEP 2.2: Generate 1000 Baseline Trajectories (Fixed)"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Navigate to phase directory
cd ~/projects/qsymphony/phase2_quantum_sim/

# Check if Step 2.1 completed
if [ ! -f ~/projects/qsymphony/results/phase2/validation/sme_solver.pkl ]; then
    echo "⚠️ Step 2.1 not verified yet. Running verification first..."
    python scripts/setup_sme_simple.py
fi

# Check number of CPU cores
echo ""
echo "💻 CPU Information:"
lscpu | grep "CPU(s)" | head -1
echo ""

# Ask user for number of processes
read -p "Enter number of parallel processes (default: all cores): " n_proc
if [ -z "$n_proc" ]; then
    n_proc="auto"
fi

# Run trajectory generation
echo ""
echo "🚀 Starting trajectory generation at $(date)"
echo "This will take several hours. Use screen/tmux to run in background."
echo ""

# Optional: Use screen if available
if command -v screen &> /dev/null; then
    echo "To run in background with screen:"
    echo "  screen -S qsymphony"
    echo "  python scripts/generate_trajectories_fixed.py"
    echo "  Ctrl+A, D to detach"
    echo "  screen -r qsymphony to reattach"
    echo ""
fi

# Ask for confirmation
read -p "Start trajectory generation now? (y/n): " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "Starting at $(date)"
    time python scripts/generate_trajectories_fixed.py
    echo "Completed at $(date)"
else
    echo "Run manually with: python scripts/generate_trajectories_fixed.py"
fi

echo ""
echo "=================================================="
echo "STEP 2.2 execution initiated"
echo "=================================================="
