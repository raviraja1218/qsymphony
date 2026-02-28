#!/bin/bash
conda activate qsymphony
echo "✅ Q-SYMPHONY environment activated"
echo "📁 Project root: ~/projects/qsymphony"
echo "💾 Dataset root: ~/Research/Datasets/qsymphony"
echo "🎮 GPU status:"
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv,noheader
