#!/usr/bin/env python
"""
Extract optimal TMS parameters for Phase 3
"""

import numpy as np

# From the optimization output
optimal_g_tms_mhz = 1583.33  # MHz
optimal_E_N = 0.2505

print("="*60)
print("OPTIMAL TWO-MODE SQUEEZING PARAMETERS")
print("="*60)
print(f"\n📊 Results from optimization:")
print(f"   Optimal g_tms = {optimal_g_tms_mhz:.1f} MHz")
print(f"   Maximum E_N = {optimal_E_N:.4f}")
print(f"\n📝 Parameters for Phase 3:")
print(f"   g_tms = {optimal_g_tms_mhz * 1e6:.1e} Hz")
print(f"   g_tms/2π = {optimal_g_tms_mhz:.1f} MHz")
print(f"\n🎯 Target entanglement: E_N ≈ {optimal_E_N:.3f}")

# Save to file
with open('tms_params_phase3.txt', 'w') as f:
    f.write("# Phase 3 Two-Mode Squeezing Parameters\n")
    f.write(f"G_TMS_MHZ = {optimal_g_tms_mhz:.1f}\n")
    f.write(f"G_TMS_HZ = {optimal_g_tms_mhz * 1e6:.1e}\n")
    f.write(f"EXPECTED_E_N = {optimal_E_N:.4f}\n")
    f.write(f"# Generated: $(date)\n")

print("\n✅ Parameters saved to: tms_params_phase3.txt")
