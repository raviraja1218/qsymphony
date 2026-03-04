#!/usr/bin/env python
"""Complete verification of all Q-SYMPHONY packages"""

import importlib
import subprocess
import sys
import pkg_resources

# Define all packages we need with their expected versions
REQUIRED_PACKAGES = {
    # Core scientific (CRITICAL)
    'numpy': '1.24.3',
    'scipy': '1.10.1',
    'matplotlib': '3.7.1',
    'pandas': '2.0.3',
    'jupyter': '1.0.0',
    'ipython': '8.14.0',
    'tqdm': '4.65.0',
    'pyyaml': '6.0',
    'h5py': '3.9.0',
    
    # Deep Learning (CRITICAL)
    'torch': '2.0.1',
    'tensorflow': '2.13.0',
    'tensorflow_probability': '0.20.0',
    'pytorch_lightning': '2.0.6',
    
    # Graph Neural Networks (CRITICAL for Phase 1)
    'torch_geometric': '2.3.1',
    'torch_scatter': None,
    'torch_sparse': None,
    'torch_cluster': None,
    'torch_spline_conv': None,
    'networkx': '3.1',
    
    # Quantum Computing (CRITICAL for Phase 2-4)
    'qiskit': '0.44.1',
    'qiskit_metal': '0.1.2',
    'qiskit_dynamics': '2.0.0',
    'qiskit_experiments': '0.5.0',
    'qiskit_ibm_runtime': '0.14.0',
    'qutip': '4.7.5',
    'qutip_qip': '0.3.0',
    'pennylane': '0.31.0',
    'pennylane_lightning': '0.31.0',
    'cvxpy': '1.4.1',
    
    # Reinforcement Learning (CRITICAL for Phase 3)
    'gym': '0.26.2',
    'stable_baselines3': '2.0.0',
    'sb3_contrib': '2.0.0',
    'tensorboard': '2.13.0',
    'wandb': '0.15.4',
    
    # Visualization (Important for figures)
    'plotly': '5.15.0',
    'seaborn': '0.12.2',
    'ipywidgets': '8.0.6',
    'mayavi': '4.8.1',  # FAILED - needs special handling
    'pyvista': '0.40.1',
    'pyvistaqt': '0.11.0',
    'vtk': '9.2.6',
    
    # Development Tools
    'pytest': '7.4.0',
    'black': '23.7.0',
    'flake8': '6.0.0',
    'mypy': '1.4.1',
    'pre_commit': '3.3.3',
}

def check_package(package_name, expected_version=None):
    """Check if package is installed and version matches"""
    try:
        module = importlib.import_module(package_name)
        
        # Get version using different methods
        version = None
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        
        if version and expected_version:
            version_match = version.startswith(str(expected_version))
        else:
            version_match = None
        
        return {
            'installed': True,
            'version': version,
            'version_match': version_match,
            'expected': expected_version
        }
    except ImportError as e:
        return {
            'installed': False,
            'error': str(e),
            'expected': expected_version
        }
    except Exception as e:
        return {
            'installed': False,
            'error': f'Unexpected error: {e}',
            'expected': expected_version
        }

def check_gpu():
    """Check GPU availability"""
    results = {}
    
    # PyTorch GPU
    try:
        import torch
        results['pytorch_cuda'] = torch.cuda.is_available()
        if results['pytorch_cuda']:
            results['pytorch_cuda_version'] = torch.version.cuda
            results['gpu_name'] = torch.cuda.get_device_name(0)
            results['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    except:
        results['pytorch_cuda'] = False
    
    # TensorFlow GPU
    try:
        import tensorflow as tf
        results['tensorflow_cuda'] = tf.config.list_physical_devices('GPU')
    except:
        results['tensorflow_cuda'] = False
    
    return results

def main():
    print("\n" + "="*60)
    print("Q-SYMPHONY COMPLETE PACKAGE VERIFICATION")
    print("="*60)
    
    # Check system info
    print(f"\n📋 System Info:")
    print(f"  Python: {sys.version}")
    
    # Check GPU
    gpu_status = check_gpu()
    print(f"\n🎮 GPU Status:")
    print(f"  PyTorch CUDA: {'✅' if gpu_status.get('pytorch_cuda') else '❌'}")
    if gpu_status.get('pytorch_cuda'):
        print(f"    - CUDA Version: {gpu_status.get('pytorch_cuda_version')}")
        print(f"    - GPU: {gpu_status.get('gpu_name')}")
        print(f"    - Memory: {gpu_status.get('gpu_memory')}")
    print(f"  TensorFlow GPU: {'✅' if gpu_status.get('tensorflow_cuda') else '❌'}")
    
    # Check all packages by category
    categories = {
        'Core Scientific': ['numpy', 'scipy', 'matplotlib', 'pandas', 'jupyter', 'ipython', 'tqdm', 'pyyaml', 'h5py'],
        'Deep Learning': ['torch', 'tensorflow', 'tensorflow_probability', 'pytorch_lightning'],
        'Graph Neural Networks': ['torch_geometric', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv', 'networkx'],
        'Quantum Computing': ['qiskit', 'qiskit_metal', 'qiskit_dynamics', 'qiskit_experiments', 'qiskit_ibm_runtime', 'qutip', 'qutip_qip', 'pennylane', 'pennylane_lightning', 'cvxpy'],
        'Reinforcement Learning': ['gym', 'stable_baselines3', 'sb3_contrib', 'tensorboard', 'wandb'],
        'Visualization': ['plotly', 'seaborn', 'ipywidgets', 'mayavi', 'pyvista', 'pyvistaqt', 'vtk'],
        'Development Tools': ['pytest', 'black', 'flake8', 'mypy', 'pre_commit'],
    }
    
    all_passed = True
    failed_packages = []
    
    for category, packages in categories.items():
        print(f"\n📦 {category}:")
        for pkg in packages:
            result = check_package(pkg, REQUIRED_PACKAGES.get(pkg))
            
            if result['installed']:
                version_info = f"v{result['version']}" if result['version'] else "unknown version"
                if result['version_match'] is False:
                    status = f"⚠️ {version_info} (expected {result['expected']})"
                    all_passed = False
                    failed_packages.append(f"{pkg}: version mismatch")
                else:
                    status = f"✅ {version_info}"
            else:
                status = f"❌ NOT INSTALLED - {result.get('error', '')}"
                all_passed = False
                failed_packages.append(pkg)
            
            print(f"    {pkg:<25} {status}")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if all_passed:
        print("\n✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
    else:
        print("\n❌ SOME PACKAGES ARE MISSING OR HAVE ISSUES:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
    
    print("\n📁 Directory Structure Check:")
    dirs = [
        '~/projects/qsymphony',
        '~/projects/qsymphony/phase1_hardware',
        '~/projects/qsymphony/results',
        '~/Research/Datasets/qsymphony',
        '~/Research/Datasets/qsymphony/raw_simulations'
    ]
    import os
    for d in dirs:
        expanded = os.path.expanduser(d)
        exists = os.path.exists(expanded)
        print(f"    {'✅' if exists else '❌'} {expanded}")
    
    print("\n" + "="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
