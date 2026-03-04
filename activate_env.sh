#!/bin/bash
# Activate Q-SYMPHONY environment with all settings

# Activate conda environment
source ~/miniconda3/bin/activate qsymphony

# Set PYTHONPATH
export PYTHONPATH="$HOME/projects/qsymphony:$PYTHONPATH"

# Set dataset paths
export QSYMPHONY_DATA="$HOME/Research/Datasets/qsymphony"
export QSYMPHONY_RESULTS="$HOME/projects/qsymphony/results"

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Print status
echo "================================="
echo "Q-SYMPHONY Environment Activated"
echo "================================="
echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Data path: $QSYMPHONY_DATA"
echo "Results path: $QSYMPHONY_RESULTS"
echo "================================="
