#!/usr/bin/env python
"""Test batch 0 generation"""
from generate_layouts_qiskit import process_batch

if __name__ == "__main__":
    print("Testing batch 0 generation...")
    batch_num, results = process_batch(0)
    print(f"Test complete: {len(results)} layouts generated")
