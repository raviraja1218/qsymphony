#!/usr/bin/env python
"""Check training progress by looking at saved models"""

from pathlib import Path
import glob

models = sorted(Path("models").glob("ppo_oracle_seed_*.pt"))
print(f"Found {len(models)} saved models:")
for m in models:
    size = m.stat().st_size / 1024 / 1024
    print(f"  {m.name} ({size:.1f} MB)")
