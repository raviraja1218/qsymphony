#!/bin/bash
echo "=================================================="
echo "Fixing Environment - NumPy Compatibility"
echo "=================================================="

# Activate environment
source ~/projects/qsymphony/activate_env.sh

# Downgrade numpy to compatible version
echo "\n[1/3] Downgrading numpy to 1.23.5..."
pip install numpy==1.23.5 --force-reinstall

# Reinstall packages that broke
echo "\n[2/3] Reinstalling broken packages..."
pip uninstall tensorflow pennylane -y
pip install tensorflow==2.13.0
pip install pennylane==0.30.0

# Verify
echo "\n[3/3] Verifying fixes..."
python -c "
import numpy as np
import tensorflow as tf
import pennylane as qml
print(f'✅ NumPy: {np.__version__}')
print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ PennyLane: {qml.__version__}')
"
echo "\n=================================================="
echo "✅ Environment fixed"
echo "=================================================="
