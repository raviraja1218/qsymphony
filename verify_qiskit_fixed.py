#!/usr/bin/env python3
"""
Final verification for Qiskit 1.x environment - Fixed syntax
"""

import importlib
import sys
from datetime import datetime

def check_qiskit():
    """Special check for Qiskit 1.x"""
    try:
        import qiskit
        print(f"✅ qiskit core: {qiskit.__version__}")
        
        # Check for qiskit-terra (should NOT exist)
        try:
            import qiskit_terra
            print("❌ ERROR: qiskit-terra still present!")
            return False
        except ImportError:
            print("✅ No qiskit-terra (good)")
        
        return True
    except ImportError as e:
        error_msg = str(e).split('\n')[0]
        print(f"❌ qiskit import failed: {error_msg}")
        return False
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"❌ qiskit error: {error_msg}")
        return False

def check_extension(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'installed')
        print(f"✅ {name:25} {version}")
        return True
    except ImportError as e:
        error_msg = str(e).split('\n')[0]
        print(f"❌ {name:25} {error_msg}")
        return False
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"❌ {name:25} Error: {error_msg}")
        return False

def main():
    print("="*70)
    print(f" QISKIT 1.x VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    print("\n📦 QISKIT CHECK")
    print("-"*50)
    qiskit_ok = check_qiskit()
    
    if qiskit_ok:
        print("\n📦 EXTENSIONS")
        print("-"*50)
        extensions = [
            ('qiskit_metal', 'qiskit_metal'),
            ('qiskit_dynamics', 'qiskit_dynamics'),
            ('qiskit_experiments', 'qiskit_experiments'),
            ('qiskit_optimization', 'qiskit_optimization'),
            ('qiskit_machine_learning', 'qiskit_machine_learning'),
            ('qiskit_nature', 'qiskit_nature'),
            ('qutip', 'qutip'),
        ]
        
        all_ok = True
        for name, import_name in extensions:
            if not check_extension(name, import_name):
                all_ok = False
        
        print("\n📦 OTHER PACKAGES")
        print("-"*50)
        others = ['numpy', 'scipy', 'matplotlib', 'pandas', 'pyyaml']
        for pkg in others:
            check_extension(pkg, pkg)
        
        print("\n" + "="*70)
        if all_ok:
            print("🎉 SUCCESS! All Qiskit 1.x extensions work")
            print("Phase 0 COMPLETE - Ready for Phase 1")
        else:
            print("⚠️ Some extensions failed - see above")
    else:
        print("\n❌ Qiskit core failed - need to fix first")
    
    print("="*70)

if __name__ == "__main__":
    main()
