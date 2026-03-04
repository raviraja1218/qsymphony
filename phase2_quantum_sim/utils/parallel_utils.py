#!/usr/bin/env python
"""Parallel processing utilities for QuTiP simulations"""

import multiprocessing as mp
from functools import partial
import numpy as np
from tqdm import tqdm
import time
import os

def get_cpu_count():
    """Get number of available CPU cores"""
    return mp.cpu_count()

def parallel_map(func, iterable, num_processes=None, desc="Processing", **kwargs):
    """
    Parallel map with progress bar
    Falls back to serial if multiprocessing fails
    """
    if num_processes is None:
        num_processes = get_cpu_count()
    
    # For small jobs, run serially
    if len(iterable) < num_processes:
        results = []
        for item in tqdm(iterable, desc=desc):
            results.append(func(item, **kwargs))
        return results
    
    # For larger jobs, use multiprocessing
    try:
        with mp.Pool(processes=num_processes) as pool:
            # Use imap for progress tracking
            results = list(tqdm(
                pool.imap(partial(func, **kwargs), iterable),
                total=len(iterable),
                desc=desc
            ))
        return results
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to serial execution...")
        results = []
        for item in tqdm(iterable, desc=desc):
            results.append(func(item, **kwargs))
        return results

def chunk_list(lst, n_chunks):
    """Split list into roughly equal chunks"""
    chunk_size = len(lst) // n_chunks
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])
    return chunks

class ProgressBar:
    """Simple progress bar for logging"""
    
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.n = 0
    
    def update(self, n=1):
        self.n += n
        elapsed = time.time() - self.start_time
        if self.n > 0:
            rate = self.n / elapsed if elapsed > 0 else 0
            eta = (self.total - self.n) / rate if rate > 0 else 0
            print(f"\r{self.desc}: {self.n}/{self.total} "
                  f"[{elapsed:.1f}s, {rate:.2f}it/s, ETA: {eta:.1f}s]", end="")
    
    def close(self):
        print()  # Newline
