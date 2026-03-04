#!/usr/bin/env python
"""Debug version of QSymphony environment"""

import sys
import traceback
from pathlib import Path

print("="*60)
print("DEBUGGING QSYMPHONY ENVIRONMENT")
print("="*60)

try:
    from qsymphony_env import QSymphonyEnv
    print("✅ Environment imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    # Create environment
    print("\n📦 Creating environment...")
    env = QSymphonyEnv(seed=42)
    print("✅ Environment created")
    
    # Reset
    print("\n🔄 Resetting...")
    obs, _ = env.reset()
    print(f"✅ Reset successful, obs shape: {obs.shape}")
    print(f"   obs: {obs}")
    
    # Take one step with a simple action
    print("\n👣 Taking one step...")
    action = np.array([0.0, 0.5])  # Simple test action
    print(f"   action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Step successful")
    print(f"   obs: {obs[:5]}... (truncated)")
    print(f"   reward: {reward}")
    print(f"   terminated: {terminated}")
    print(f"   truncated: {truncated}")
    
    # Save trajectory
    print("\n💾 Saving trajectory...")
    env.save_trajectory("test_trajectory_debug.json")
    
    # Check file
    import os
    file_size = os.path.getsize("test_trajectory_debug.json")
    print(f"   File size: {file_size} bytes")
    
    if file_size > 10:
        print("✅ Trajectory saved successfully")
    else:
        print("❌ Trajectory file is empty!")
        
        # Print what's in trajectory_data
        print(f"\n📊 trajectory_data length: {len(env.trajectory_data)}")
        if len(env.trajectory_data) > 0:
            print(f"   First entry: {env.trajectory_data[0]}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()

print("\n" + "="*60)
