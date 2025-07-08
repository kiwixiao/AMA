#!/usr/bin/env python3
"""
Dynamic Memory and CPU Allocation System

This system automatically:
1. Detects available memory and CPU cores
2. Calculates safe memory limits to prevent kill -9 crashes
3. Dynamically adjusts batch sizes based on actual system resources
4. Distributes work across all available cores
"""

import os
import sys
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DynamicResourceManager:
    """Manages dynamic allocation of memory and CPU resources"""
    
    def __init__(self, safety_margin: float = 0.8):
        """
        Initialize with safety margin to prevent crashes
        
        Args:
            safety_margin: Use this fraction of available memory (0.8 = 80%)
        """
        self.safety_margin = safety_margin
        self.available_memory = self._get_available_memory()
        self.available_cores = self._get_available_cores()
        self.process_memory_limit = self._calculate_process_memory_limit()
        
        print(f"System Resources Detected:")
        print(f"  Available Memory: {self.available_memory / 1024**3:.1f} GB")
        print(f"  Available CPU Cores: {self.available_cores}")
        print(f"  Process Memory Limit: {self.process_memory_limit / 1024**3:.1f} GB")
        
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes"""
        memory = psutil.virtual_memory()
        return memory.available
    
    def _get_available_cores(self) -> int:
        """Get number of available CPU cores"""
        # Use all available cores but leave one for system
        return max(1, mp.cpu_count() - 1)
    
    def _calculate_process_memory_limit(self) -> int:
        """Calculate safe memory limit for each process"""
        # Reserve memory for system and other processes
        safe_memory = int(self.available_memory * self.safety_margin)
        # Divide by number of cores to get per-process limit
        return safe_memory // self.available_cores
    
    def calculate_optimal_batch_size(self, data_size: int, item_memory_size: int) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            data_size: Total number of items to process
            item_memory_size: Estimated memory per item in bytes
            
        Returns:
            Optimal batch size
        """
        # Calculate how many items can fit in process memory limit
        max_items_per_batch = self.process_memory_limit // item_memory_size
        
        # Don't make batches too small or too large
        min_batch_size = max(1, data_size // (self.available_cores * 4))  # At least 4 batches per core
        max_batch_size = min(max_items_per_batch, data_size // self.available_cores)
        
        optimal_batch_size = min(max_batch_size, max(min_batch_size, 100))
        
        # Ensure batch size is at least 1
        optimal_batch_size = max(1, optimal_batch_size)
        
        print(f"Batch size calculation:")
        print(f"  Data size: {data_size}")
        print(f"  Item memory: {item_memory_size / 1024**2:.1f} MB")
        print(f"  Max items per batch: {max_items_per_batch}")
        print(f"  Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage as fraction of total"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / psutil.virtual_memory().total
    
    def monitor_memory_usage(self) -> bool:
        """Check if memory usage is safe"""
        current_usage = self.get_current_memory_usage()
        return current_usage < (self.safety_margin * 0.9)  # 90% of safety margin

class DynamicDataProcessor:
    """Processes data with dynamic resource allocation"""
    
    def __init__(self, resource_manager: DynamicResourceManager):
        self.resource_manager = resource_manager
        
    def estimate_memory_per_file(self, file_path: Path) -> int:
        """Estimate memory required to process one file"""
        try:
            file_size = file_path.stat().st_size
            # Rough estimate: CSV files need 3-5x their size in memory when loaded
            return file_size * 4
        except:
            return 100 * 1024 * 1024  # 100MB default
    
    def process_file_batch(self, file_batch: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of files in current process"""
        results = []
        
        for file_path in file_batch:
            try:
                # Check memory before processing each file
                if not self.resource_manager.monitor_memory_usage():
                    print(f"Memory limit approached, triggering garbage collection")
                    gc.collect()
                    
                    if not self.resource_manager.monitor_memory_usage():
                        print(f"Skipping {file_path} due to memory constraints")
                        continue
                
                # Simulate file processing (replace with your actual processing)
                result = self._process_single_file(file_path)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return results
    
    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file (replace with your actual processing logic)"""
        # This is a placeholder - replace with your actual file processing
        # For demonstration, we'll just return file info
        return {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'process_time': time.time(),
            'processed': True
        }

def process_files_dynamically(file_paths: List[Path], 
                            resource_manager: DynamicResourceManager) -> List[Dict[str, Any]]:
    """
    Process files using dynamic resource allocation
    
    Args:
        file_paths: List of file paths to process
        resource_manager: Resource manager instance
        
    Returns:
        List of processing results
    """
    processor = DynamicDataProcessor(resource_manager)
    
    # Estimate memory requirement per file
    sample_file = file_paths[0] if file_paths else None
    if sample_file:
        memory_per_file = processor.estimate_memory_per_file(sample_file)
    else:
        memory_per_file = 100 * 1024 * 1024  # 100MB default
    
    # Calculate optimal batch size
    batch_size = resource_manager.calculate_optimal_batch_size(
        len(file_paths), memory_per_file
    )
    
    # Create batches
    batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    
    print(f"Processing {len(file_paths)} files in {len(batches)} batches using {resource_manager.available_cores} cores")
    
    all_results = []
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=resource_manager.available_cores) as executor:
        future_to_batch = {
            executor.submit(processor.process_file_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                print(f"Completed batch {batch_idx + 1}/{len(batches)} ({len(batch_results)} files)")
                
            except Exception as e:
                print(f"Batch {batch_idx + 1} failed: {e}")
    
    return all_results

def demonstrate_dynamic_allocation():
    """Demonstrate the dynamic allocation system"""
    print("=== Dynamic Memory and CPU Allocation Demo ===\n")
    
    # Initialize resource manager
    resource_manager = DynamicResourceManager(safety_margin=0.8)
    
    # Find some files to process (example)
    current_dir = Path(".")
    file_paths = list(current_dir.glob("*.csv"))[:20]  # Process up to 20 CSV files
    
    if not file_paths:
        print("No CSV files found for demonstration")
        # Create some dummy files for demo
        print("Creating dummy files for demonstration...")
        for i in range(10):
            dummy_file = Path(f"dummy_{i}.csv")
            dummy_file.write_text(f"dummy,data,{i}\n" * 1000)
            file_paths.append(dummy_file)
    
    print(f"\nProcessing {len(file_paths)} files...")
    
    # Process files with dynamic allocation
    start_time = time.time()
    results = process_files_dynamically(file_paths, resource_manager)
    end_time = time.time()
    
    print(f"\nResults:")
    print(f"  Files processed: {len(results)}")
    print(f"  Total time: {end_time - start_time:.2f} seconds")
    print(f"  Average time per file: {(end_time - start_time) / len(results):.3f} seconds")
    
    # Clean up dummy files
    for file_path in file_paths:
        if file_path.name.startswith("dummy_"):
            file_path.unlink()

if __name__ == "__main__":
    demonstrate_dynamic_allocation() 