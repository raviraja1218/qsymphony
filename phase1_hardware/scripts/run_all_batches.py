#!/usr/bin/env python
"""Run all 10 batches sequentially with monitoring"""

import time
import psutil
import os
from datetime import datetime

def log_status(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/generation.log', 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def monitor_resources():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    return cpu, memory

def main():
    log_status("="*60)
    log_status("Starting full layout generation (10 batches)")
    log_status("="*60)
    
    start_time = time.time()
    
    for batch in range(10):
        log_status(f"Starting batch {batch}/9")
        
        # Monitor before
        cpu, mem = monitor_resources()
        log_status(f"Resources - CPU: {cpu}%, Memory: {mem}%")
        
        # Run batch
        batch_start = time.time()
        from generate_layouts_qiskit import process_batch
        batch_num, results = process_batch(batch)
        batch_time = time.time() - batch_start
        
        # Monitor after
        cpu, mem = monitor_resources()
        log_status(f"Batch {batch} complete in {batch_time:.1f} seconds")
        log_status(f"Generated {len(results)} layouts")
        log_status(f"Resources after - CPU: {cpu}%, Memory: {mem}%")
        
        # Suggest break if running hot
        if cpu > 80 or mem > 80:
            log_status("⚠️ High resource usage - consider taking a break")
        
        # Small break between batches
        if batch < 9:
            log_status("Cooling down for 30 seconds...")
            time.sleep(30)
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    log_status("="*60)
    log_status(f"✅ COMPLETE: All 10 batches finished")
    log_status(f"Total time: {hours}h {minutes}m")
    log_status("="*60)

if __name__ == "__main__":
    main()
