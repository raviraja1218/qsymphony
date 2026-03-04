#!/usr/bin/env python
"""Simple test for SME solver"""

import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

print("Testing SME solver...")
exec(open('setup_sme_simple.py').read())
print("\n✅ Test complete!")
