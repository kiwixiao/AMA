#!/usr/bin/env python3
"""
Integrated Dynamic Processing with Existing Codebase

This integrates the dynamic allocation system with the existing 
file processing functions from main.py
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

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from main import (
        track_patch_region_in_file,
        track_region_in_file,
        calculate_derived_quantities,
        save_trajectory_data
    )
    MAIN_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import main.py functions")
    MAIN_FUNCTIONS_AVAILABLE = False

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
        min_batch_size = max(1, data_size // (self.available_cores * 4))
        max_batch_size = min(max_items_per_batch, data_size // self.available_cores)
        
        optimal_batch_size = min(max_batch_size, max(min_batch_size, 10))
        
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
        return current_usage < (self.safety_margin * 0.9)

def process_csv_files_batch(file_batch: List[Path], 
                          patch_number: int, 
                          face_index: int, 
                          radius: float,
                          resource_manager: DynamicResourceManager) -> List[Dict[str, Any]]:
    """
    Process a batch of CSV files using the existing main.py functions
    
    Args:
        file_batch: List of CSV file paths to process
        patch_number: Patch number for tracking
        face_index: Face index for tracking
        radius: Radius for region tracking
        resource_manager: Resource manager for monitoring
        
    Returns:
        List of processing results
    """
    results = []
    
    for file_path in file_batch:
        try:
            # Check memory before processing each file
            if not resource_manager.monitor_memory_usage():
                print(f"Memory limit approached, triggering garbage collection")
                gc.collect()
                
                if not resource_manager.monitor_memory_usage():
                    print(f"Skipping {file_path} due to memory constraints")
                    continue
            
            print(f"Processing: {file_path.name}")
            
            if MAIN_FUNCTIONS_AVAILABLE:
                # Use the actual tracking functions from main.py
                tracking_result = track_patch_region_in_file(
                    file_path, patch_number, face_index, radius
                )
                
                if tracking_result and 'trajectories' in tracking_result:
                    # Calculate derived quantities
                    derived_data = calculate_derived_quantities(tracking_result['trajectories'])
                    
                    result = {
                        'file_path': str(file_path),
                        'patch_number': patch_number,
                        'face_index': face_index,
                        'radius': radius,
                        'tracking_result': tracking_result,
                        'derived_data': derived_data,
                        'processed': True
                    }
                else:
                    result = {
                        'file_path': str(file_path),
                        'patch_number': patch_number,
                        'face_index': face_index,
                        'radius': radius,
                        'error': 'No tracking data found',
                        'processed': False
                    }
            else:
                # Fallback simulation
                result = {
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'patch_number': patch_number,
                    'face_index': face_index,
                    'radius': radius,
                    'processed': True,
                    'simulation': True
                }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results.append({
                'file_path': str(file_path),
                'error': str(e),
                'processed': False
            })
    
    return results

def process_subject_data_dynamically(subject_name: str, 
                                   patch_number: int = 35,
                                   face_index: int = 0,
                                   radius: float = 0.002,
                                   resource_manager: DynamicResourceManager = None) -> List[Dict[str, Any]]:
    """
    Process all CSV files for a subject using dynamic resource allocation
    
    Args:
        subject_name: Name of the subject (e.g., "OSAMRI007")
        patch_number: Patch number for tracking
        face_index: Face index for tracking
        radius: Radius for region tracking
        resource_manager: Resource manager (will create if None)
        
    Returns:
        List of processing results
    """
    if resource_manager is None:
        resource_manager = DynamicResourceManager()
    
    # Find CSV files for the subject
    csv_files = []
    
    # Look for CSV files in different mesh directories
    for mesh_dir in [f"{subject_name}_xyz_tables", f"2mmesh{subject_name}_xyz_tables", 
                     f"23mmesh{subject_name}_xyz_tables", f"less1mmesh{subject_name}_xyz_tables"]:
        mesh_path = Path(mesh_dir)
        if mesh_path.exists():
            csv_files.extend(list(mesh_path.glob("*.csv")))
    
    if not csv_files:
        print(f"No CSV files found for subject {subject_name}")
        return []
    
    print(f"Found {len(csv_files)} CSV files for {subject_name}")
    
    # Estimate memory per file (CSV files typically need 3-5x their size in memory)
    sample_file = csv_files[0]
    estimated_memory_per_file = sample_file.stat().st_size * 4
    
    # Calculate optimal batch size
    batch_size = resource_manager.calculate_optimal_batch_size(
        len(csv_files), estimated_memory_per_file
    )
    
    # Create batches
    batches = [csv_files[i:i + batch_size] for i in range(0, len(csv_files), batch_size)]
    
    print(f"Processing {len(csv_files)} files in {len(batches)} batches using {resource_manager.available_cores} cores")
    
    all_results = []
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=resource_manager.available_cores) as executor:
        future_to_batch = {
            executor.submit(process_csv_files_batch, batch, patch_number, face_index, radius, resource_manager): i 
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                processed_count = sum(1 for r in batch_results if r.get('processed', False))
                print(f"Completed batch {batch_idx + 1}/{len(batches)} ({processed_count}/{len(batch_results)} files processed)")
                
            except Exception as e:
                print(f"Batch {batch_idx + 1} failed: {e}")
    
    return all_results

def demonstrate_integrated_processing():
    """Demonstrate the integrated processing system"""
    print("=== Integrated Dynamic Processing Demo ===\n")
    
    # Initialize resource manager
    resource_manager = DynamicResourceManager(safety_margin=0.8)
    
    # Example subject processing
    subject_name = "OSAMRI007"
    patch_number = 35
    face_index = 0
    radius = 0.002
    
    print(f"Processing subject: {subject_name}")
    print(f"Parameters: patch={patch_number}, face={face_index}, radius={radius}")
    
    # Process subject data
    start_time = time.time()
    results = process_subject_data_dynamically(
        subject_name, patch_number, face_index, radius, resource_manager
    )
    end_time = time.time()
    
    # Display results
    print(f"\nResults:")
    print(f"  Total files found: {len(results)}")
    processed_count = sum(1 for r in results if r.get('processed', False))
    print(f"  Files processed successfully: {processed_count}")
    print(f"  Processing time: {end_time - start_time:.2f} seconds")
    
    if processed_count > 0:
        print(f"  Average time per file: {(end_time - start_time) / processed_count:.3f} seconds")
    
    # Show some example results
    print("\nExample results:")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"  {i+1}. {Path(result['file_path']).name}: {'SUCCESS' if result.get('processed') else 'FAILED'}")
        if 'error' in result:
            print(f"     Error: {result['error']}")

if __name__ == "__main__":
    demonstrate_integrated_processing() 