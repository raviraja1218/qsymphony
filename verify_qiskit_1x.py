#!/usr/bin/env python3
"""
Verification for Qiskit 1.4.2 environment - Fixed syntax
"""

import importlib
import sys
from datetime import datetime

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'installed')
        return True, version
    except ImportError as e:
        error_msg = str(e).split('\n')[0]
        return False, error_msg
    except Exception as e:
        error_msg = str(e)[:50]
        return False, error_msg

def test_qiskit_modules():
    """Test specific Qiskit modules that extensions need"""
    tests = [
        ('qiskit.pulse', 'pulse module'),
        ('qiskit.providers.models', 'providers.models'),
        ('qiskit.primitives.BaseSampler', 'BaseSampler'),
    ]
    print("\n🔍 Qiskit Module Tests:")
    for module_path, description in tests:
        try:
            if '.' in module_path:
                # Handle nested imports
                parts = module_path.split('.')
                module_name = '.'.join(parts[:-1])
                attr_name = parts[-1]
                module = importlib.import_module(module_name)
                if hasattr(module, attr_name):
                    print(f"✅ {description:30} Found")
                else:
                    print(f"❌ {description:30} Missing")
            else:
                importlib.import_module(module_path)
                print(f"✅ {description:30} Found")
        except ImportError:
            print(f"❌ {description:30} Missing")
        except Exception as e:
            error_msg = str(e)[:30]
            print(f"❌ {description:30} Error: {error_msg}")

def main():
    print("="*70)
    print(f" QISKIT 1.4.2 VERIFICATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # GPU Check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    # Qiskit Version
    qiskit_version = None
    try:
        import qiskit
        qiskit_version = qiskit.__version__
        print(f"\n✅ Qiskit Version: {qiskit_version} (should be 1.4.2)")
    except ImportError as e:
        error_msg = str(e).split('\n')[0]
        print(f"\n❌ Qiskit not installed: {error_msg}")
        return
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"\n❌ Qiskit error: {error_msg}")
        return
    
    # Test required modules
    test_qiskit_modules()
    
    # Test Extensions
    print("\n📦 Qiskit Extensions:")
    print("-"*50)
    extensions = [
        ('qiskit_metal', 'qiskit_metal'),
        ('qiskit_dynamics', 'qiskit_dynamics'),
        ('qiskit_experiments', 'qiskit_experiments'),
        ('qiskit_optimization', 'qiskit_optimization'),
        ('qiskit_machine_learning', 'qiskit_machine_learning'),
        ('qiskit_nature', 'qiskit_nature'),
    ]
    
    all_ok = True
    for display_name, import_name in extensions:
        ok, info = check_package(display_name, import_name)
        if ok:
            print(f"✅ {display_name:25} {info}")
        else:
            print(f"❌ {display_name:25} {info}")
            all_ok = False
    
    # Final Status
    print("\n" + "="*70)
    if all_ok and qiskit_version == '1.4.2':
        print("🎉 SUCCESS! Qiskit 1.4.2 with all extensions working")
        print("Phase 0 COMPLETE - Ready for Phase 1")
    elif qiskit_version != '1.4.2':
        print(f"⚠️ Wrong Qiskit version: {qiskit_version} (need 1.4.2)")
        print("Run: pip install qiskit==1.4.2")
    else:
        print("⚠️ Some extensions failed - see above")
        print("Run the fix script: ~/projects/qsymphony/fix_qiskit_compatibility.sh")
    print("="*70)

if __name__ == "__main__":
    main()
