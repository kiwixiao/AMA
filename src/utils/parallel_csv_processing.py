#!/usr/bin/env python3
"""
Parallel CSV Processing Module

This module provides parallel processing capabilities for CSV operations,
specifically optimized for large CFD data files. It can be used throughout
the pipeline for various CSV processing tasks.

Key Features:
- Parallel CSV to patched CSV conversion
- Chunked processing for memory efficiency
- Progress tracking and logging
- Modular design for reusability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import psutil
import os
from functools import partial
from tqdm import tqdm
import h5py


def get_optimal_process_count(file_size_mb: float = 600, available_memory_gb: float = None) -> int:
    """
    Determine optimal number of processes based on system resources.
    
    Args:
        file_size_mb: Average file size in MB
        available_memory_gb: Available memory in GB (auto-detected if None)
    
    Returns:
        Optimal number of processes
    """
    # Get system info
    cpu_count = mp.cpu_count()
    
    if available_memory_gb is None:
        # Get available memory
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
    
    # Calculate optimal processes based on memory constraints
    # Assume each process needs ~2x file size in memory for processing
    memory_limited_processes = max(1, int(available_memory_gb * 1024 / (file_size_mb * 2)))
    
    # Use 75% of available CPU cores to leave some for system
    cpu_limited_processes = max(1, int(cpu_count * 0.75))
    
    # Take the minimum to avoid overwhelming the system
    optimal_processes = min(memory_limited_processes, cpu_limited_processes, 16)  # Cap at 16
    
    print(f"System resources detected:")
    print(f"  CPU cores: {cpu_count}")
    print(f"  Available memory: {available_memory_gb:.1f} GB")
    print(f"  Memory-limited processes: {memory_limited_processes}")
    print(f"  CPU-limited processes: {cpu_limited_processes}")
    print(f"  Optimal processes: {optimal_processes}")
    
    return optimal_processes


def add_patch_numbers_to_single_file(args: Tuple[Path, str, int]) -> Path:
    """
    Add patch numbers to a single CSV file with chunked processing.
    Uses the same Face Index reset logic as sequential processing.
    
    Args:
        args: Tuple of (input_file_path, subject_name, chunk_size)
    
    Returns:
        Path to the processed file
    """
    input_file, subject_name, chunk_size = args
    
    # Create output directory
    output_dir = Path(f'{subject_name}_xyz_tables_with_patches')
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename
    output_file = output_dir / f'patched_{input_file.name}'
    
    # Skip if already exists
    if output_file.exists():
        return output_file
    
    try:
        # Process in chunks to handle large files
        first_chunk = True
        current_patch = 1
        prev_face_idx = -1
        
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, low_memory=False):
            # Calculate patch numbers for this chunk using Face Index reset logic
            patch_numbers = []
            
            for _, row in chunk.iterrows():
                face_idx = row['Face Index']
                # Start new patch when Face Index resets to 0 (after being > 0)
                if face_idx == 0 and prev_face_idx > 0:
                    current_patch += 1
                patch_numbers.append(current_patch)
                prev_face_idx = face_idx
            
            # Add Patch Number column right after Face Index
            chunk.insert(chunk.columns.get_loc('Face Index') + 1, 'Patch Number', patch_numbers)
            
            # Write to output file (append mode after first chunk)
            chunk.to_csv(output_file, 
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False)
            first_chunk = False
        
        return output_file
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return None


def process_csv_files_parallel(input_files: List[Path], subject_name: str, 
                             chunk_size: int = 50000) -> List[Path]:
    """
    Process multiple CSV files in parallel while maintaining chronological order.
    
    Args:
        input_files: List of input CSV files (should be pre-sorted chronologically)
        subject_name: Subject identifier
        chunk_size: Chunk size for processing large files
    
    Returns:
        List of processed file paths in chronological order
    """
    if not input_files:
        return []
    
    print(f"Processing {len(input_files)} CSV files in parallel...")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, subject_name, chunk_size) for file_path in input_files]
    
    # Process files in parallel
    start_time = time.time()
    
    try:
        with mp.Pool(processes=optimal_processes) as pool:
            # Use tqdm for progress tracking
            results = list(tqdm(
                pool.imap(add_patch_numbers_to_single_file, args_list),
                total=len(args_list),
                desc="Processing files"
            ))
        
        # Filter out None results (failed processing)
        processed_files = [f for f in results if f is not None]
        
        # Sort results to maintain chronological order
        # Extract timesteps and sort
        timestep_file_pairs = []
        for file_path in processed_files:
            try:
                # Extract timestep from filename
                timestep_str = file_path.stem.split('_')[-1]
                if timestep_str.replace('.', '').isdigit():
                    timestep = float(timestep_str)
                else:
                    timestep = float(timestep_str.replace('table', ''))
                timestep_file_pairs.append((timestep, file_path))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse timestep from {file_path.name}")
                continue
        
        # Sort by timestep
        timestep_file_pairs.sort(key=lambda x: x[0])
        processed_files = [file_path for timestep, file_path in timestep_file_pairs]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Parallel processing completed:")
        print(f"  Processed: {len(processed_files)}/{len(input_files)} files")
        print(f"  Time taken: {processing_time:.1f} seconds")
        print(f"  Average: {processing_time/len(input_files):.2f} seconds per file")
        
        return processed_files
        
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing
        processed_files = []
        for file_path in tqdm(input_files, desc="Sequential processing"):
            result = add_patch_numbers_to_single_file((file_path, subject_name, chunk_size))
            if result:
                processed_files.append(result)
        
        return processed_files


def track_single_point_in_file(args: Tuple[Path, int, int]) -> Optional[Dict]:
    """
    Track a single point in one CSV file.
    
    Args:
        args: Tuple of (file_path, patch_number, face_index)
    
    Returns:
        Dictionary with tracking data or None if not found
    """
    file_path, patch_number, face_index = args
    
    try:
        # Read only necessary columns for efficiency
        required_columns = [
            'Patch Number', 'Face Index', 'X (m)', 'Y (m)', 'Z (m)',
            'Total Pressure (Pa)', 'Velocity: Magnitude (m/s)',
            'Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)',
            'Area[i] (m^2)', 'Area[j] (m^2)', 'Area[k] (m^2)'
        ]
        
        # Check if VdotN exists
        df_sample = pd.read_csv(file_path, nrows=1)
        if 'VdotN' in df_sample.columns:
            required_columns.append('VdotN')
        
        # Read the file with selected columns
        df = pd.read_csv(file_path, usecols=required_columns, low_memory=False)
        
        # Find the specific point
        point_data = df[(df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)]
        
        if len(point_data) == 0:
            return None
        
        if len(point_data) > 1:
            point_data = point_data.iloc[0:1]
        
        # Extract timestep from filename
        timestep_str = file_path.stem.split('_')[-1]
        if 'e+' in timestep_str or 'e-' in timestep_str or '.' in timestep_str:
            timestep = float(timestep_str)
            time_sec = timestep
            time_point = int(timestep * 1000)
        else:
            timestep = float(timestep_str.replace('table', ''))
            time_sec = timestep * 0.001
            time_point = int(timestep)
        
        # Calculate velocity and VdotN
        velocity_vector = np.array([
            float(point_data['Velocity[i] (m/s)'].iloc[0]),
            float(point_data['Velocity[j] (m/s)'].iloc[0]),
            float(point_data['Velocity[k] (m/s)'].iloc[0])
        ])
        velocity = float(np.linalg.norm(velocity_vector))
        
        # Calculate VdotN if not directly available
        if 'VdotN' in point_data.columns:
            vdotn = float(point_data['VdotN'].iloc[0])
        else:
            area_vector = np.array([
                float(point_data['Area[i] (m^2)'].iloc[0]),
                float(point_data['Area[j] (m^2)'].iloc[0]),
                float(point_data['Area[k] (m^2)'].iloc[0])
            ])
            if np.linalg.norm(area_vector) > 0:
                normal_vector = area_vector / np.linalg.norm(area_vector)
                vdotn = float(np.dot(velocity_vector, normal_vector))
            else:
                vdotn = 0.0
        
        signed_velocity = velocity * np.sign(vdotn)
        
        return {
            'time': time_sec,
            'time_point': time_point,
            'x': float(point_data['X (m)'].iloc[0]),
            'y': float(point_data['Y (m)'].iloc[0]),
            'z': float(point_data['Z (m)'].iloc[0]),
            'pressure': float(point_data['Total Pressure (Pa)'].iloc[0]),
            'adjusted_pressure': float(point_data['Total Pressure (Pa)'].iloc[0]),
            'velocity': velocity,
            'velocity_i': float(point_data['Velocity[i] (m/s)'].iloc[0]),
            'velocity_j': float(point_data['Velocity[j] (m/s)'].iloc[0]),
            'velocity_k': float(point_data['Velocity[k] (m/s)'].iloc[0]),
            'area_i': float(point_data['Area[i] (m^2)'].iloc[0]),
            'area_j': float(point_data['Area[j] (m^2)'].iloc[0]),
            'area_k': float(point_data['Area[k] (m^2)'].iloc[0]),
            'vdotn': vdotn,
            'signed_velocity': signed_velocity,
            'patch35_avg_pressure': 0.0  # Placeholder
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def track_point_parallel(csv_files: List[Path], patch_number: int, face_index: int) -> List[Dict]:
    """
    Track a single point across multiple CSV files in parallel.
    
    Args:
        csv_files: List of CSV files in chronological order
        patch_number: Patch number to track
        face_index: Face index to track
    
    Returns:
        List of tracking data dictionaries in chronological order
    """
    if not csv_files:
        return []
    
    print(f"Tracking point (Patch {patch_number}, Face {face_index}) across {len(csv_files)} files...")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, patch_number, face_index) for file_path in csv_files]
    
    start_time = time.time()
    
    try:
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(track_single_point_in_file, args_list),
                total=len(args_list),
                desc=f"Tracking P{patch_number}F{face_index}"
            ))
        
        # Filter out None results and maintain chronological order
        tracking_data = [result for result in results if result is not None]
        
        # Sort by time to ensure chronological order
        tracking_data.sort(key=lambda x: x['time'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Parallel point tracking completed:")
        print(f"  Found: {len(tracking_data)}/{len(csv_files)} time points")
        print(f"  Time taken: {processing_time:.1f} seconds")
        print(f"  Average: {processing_time/len(csv_files):.3f} seconds per file")
        
        return tracking_data
        
    except Exception as e:
        print(f"Error in parallel point tracking: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing
        tracking_data = []
        for file_path in tqdm(csv_files, desc="Sequential tracking"):
            result = track_single_point_in_file((file_path, patch_number, face_index))
            if result:
                tracking_data.append(result)
        
        return tracking_data


def track_patch_region_in_file(args: Tuple[Path, int, int, float]) -> Optional[Dict]:
    """
    Track a patch region around a center point in a single CSV file.
    
    Args:
        args: Tuple of (file_path, patch_number, face_index, radius)
    
    Returns:
        Dictionary with patch tracking data or None if not found
    """
    file_path, patch_number, face_index, radius = args
    
    try:
        # Read the file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Find center point
        center_data = df[(df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)]
        if len(center_data) == 0:
            return None
        
        center_point = (
            float(center_data['X (m)'].iloc[0]),
            float(center_data['Y (m)'].iloc[0]),
            float(center_data['Z (m)'].iloc[0])
        )
        
        # Find points within radius
        distances = np.sqrt(
            (df['X (m)'] - center_point[0])**2 +
            (df['Y (m)'] - center_point[1])**2 +
            (df['Z (m)'] - center_point[2])**2
        )
        
        # Filter by radius and patch number
        region_mask = (distances <= radius) & (df['Patch Number'] == patch_number)
        region_df = df[region_mask]
        
        if len(region_df) == 0:
            return None
        
        # Calculate statistics
        stats = {
            'num_points': len(region_df),
            'pressure': {
                'mean': region_df['Total Pressure (Pa)'].mean(),
                'std': region_df['Total Pressure (Pa)'].std(),
            },
            'velocity': {
                'mean': region_df['Velocity: Magnitude (m/s)'].mean(),
                'std': region_df['Velocity: Magnitude (m/s)'].std(),
            },
            'vdotn': {
                'mean': region_df['VdotN'].mean() if 'VdotN' in region_df.columns else 0,
                'std': region_df['VdotN'].std() if 'VdotN' in region_df.columns else 0,
            }
        }
        
        # Extract timestep from filename
        timestep_str = file_path.stem.split('_')[-1]
        if 'e+' in timestep_str or 'e-' in timestep_str or '.' in timestep_str:
            timestep = float(timestep_str)
            time_sec = timestep
        else:
            timestep = float(timestep_str.replace('table', ''))
            time_sec = timestep * 0.001
        
        return {
            'Time (s)': time_sec,
            'X (m)': center_point[0],
            'Y (m)': center_point[1],
            'Z (m)': center_point[2],
            'Total Pressure (Pa)': stats['pressure']['mean'],
            'Pressure Std (Pa)': stats['pressure']['std'],
            'Velocity: Magnitude (m/s)': stats['velocity']['mean'],
            'Velocity Std (m/s)': stats['velocity']['std'],
            'VdotN': stats['vdotn']['mean'],
            'VdotN Std': stats['vdotn']['std'],
            'Patch Points Count': stats['num_points'],
            'Patch Radius (mm)': radius * 1000
        }
        
    except Exception as e:
        print(f"Error processing patch region in {file_path}: {e}")
        return None


def track_patch_region_parallel(csv_files: List[Path], patch_number: int, face_index: int, radius: float) -> List[Dict]:
    """
    Track a patch region across multiple CSV files in parallel.
    
    Args:
        csv_files: List of CSV files in chronological order
        patch_number: Patch number for center point
        face_index: Face index for center point
        radius: Radius for patch region
    
    Returns:
        List of patch tracking data dictionaries in chronological order
    """
    if not csv_files:
        return []
    
    print(f"Tracking patch region (Patch {patch_number}, Face {face_index}, Radius {radius*1000:.1f}mm) across {len(csv_files)} files...")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, patch_number, face_index, radius) for file_path in csv_files]
    
    start_time = time.time()
    
    try:
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(track_patch_region_in_file, args_list),
                total=len(args_list),
                desc=f"Tracking P{patch_number}F{face_index} R{radius*1000:.1f}mm"
            ))
        
        # Filter out None results and maintain chronological order
        tracking_data = [result for result in results if result is not None]
        
        # Sort by time to ensure chronological order
        tracking_data.sort(key=lambda x: x['Time (s)'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Parallel patch tracking completed:")
        print(f"  Found: {len(tracking_data)}/{len(csv_files)} time points")
        print(f"  Time taken: {processing_time:.1f} seconds")
        print(f"  Average: {processing_time/len(csv_files):.3f} seconds per file")
        
        return tracking_data
        
    except Exception as e:
        print(f"Error in parallel patch tracking: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing
        tracking_data = []
        for file_path in tqdm(csv_files, desc="Sequential patch tracking"):
            result = track_patch_region_in_file((file_path, patch_number, face_index, radius))
            if result:
                tracking_data.append(result)
        
        return tracking_data


def track_fixed_patch_region_in_file(args: Tuple[Path, List[Tuple[int, int]]]) -> Optional[Dict]:
    """
    Track a fixed set of points (determined from first time step) in a single CSV file.
    
    Args:
        args: Tuple of (file_path, point_pairs)
    
    Returns:
        Dictionary with tracking data or None if not found
    """
    file_path, point_pairs = args
    
    try:
        # Read the file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Find all the specified points in this time step
        region_points = []
        for patch_num, face_idx in point_pairs:
            point_data = df[(df['Patch Number'] == patch_num) & (df['Face Index'] == face_idx)]
            if len(point_data) > 0:
                region_points.append(point_data.iloc[0])
        
        if len(region_points) == 0:
            return None
        
        # Convert to DataFrame for easier processing
        region_df = pd.DataFrame(region_points)
        
        # Calculate statistics
        stats = {
            'num_points': len(region_df),
            'pressure': {
                'mean': region_df['Total Pressure (Pa)'].mean(),
                'std': region_df['Total Pressure (Pa)'].std(),
            },
            'velocity': {
                'mean': region_df['Velocity: Magnitude (m/s)'].mean(),
                'std': region_df['Velocity: Magnitude (m/s)'].std(),
            },
            'vdotn': {
                'mean': region_df['VdotN'].mean() if 'VdotN' in region_df.columns else 0,
                'std': region_df['VdotN'].std() if 'VdotN' in region_df.columns else 0,
            }
        }
        
        # Extract timestep from filename
        timestep_str = file_path.stem.split('_')[-1]
        if 'e+' in timestep_str or 'e-' in timestep_str or '.' in timestep_str:
            timestep = float(timestep_str)
            time_sec = timestep
            time_point = int(timestep * 1000)
        else:
            timestep = float(timestep_str.replace('table', ''))
            time_sec = timestep * 0.001
            time_point = int(timestep)
        
        # Use center point coordinates (mean of all points in region)
        center_x = region_df['X (m)'].mean()
        center_y = region_df['Y (m)'].mean()
        center_z = region_df['Z (m)'].mean()
        
        # Return data in same format as single point tracking
        return {
            'time_point': time_point,
            'time': time_sec,
            'x': center_x,
            'y': center_y,
            'z': center_z,
            'pressure': stats['pressure']['mean'],
            'patch35_avg_pressure': 0.0,  # Not applicable for patch analysis
            'adjusted_pressure': stats['pressure']['mean'],
            'velocity': stats['velocity']['mean'],
            'velocity_i': region_df['Velocity[i] (m/s)'].mean(),
            'velocity_j': region_df['Velocity[j] (m/s)'].mean(),
            'velocity_k': region_df['Velocity[k] (m/s)'].mean(),
            'area_i': region_df['Area[i] (m^2)'].mean(),
            'area_j': region_df['Area[j] (m^2)'].mean(),
            'area_k': region_df['Area[k] (m^2)'].mean(),
            'vdotn': stats['vdotn']['mean'],
            'signed_velocity': stats['vdotn']['mean'],  # Use VdotN as signed velocity
            'num_points_in_region': stats['num_points'],
            'pressure_std': stats['pressure']['std'],
            'velocity_std': stats['velocity']['std'],
            'vdotn_std': stats['vdotn']['std']
        }
        
    except Exception as e:
        print(f"Error processing fixed patch region in {file_path}: {e}")
        return None


def track_fixed_patch_region_parallel(csv_files: List[Path], point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Track a fixed set of points across multiple CSV files in parallel.
    
    Args:
        csv_files: List of CSV files in chronological order
        point_pairs: List of (patch_number, face_index) tuples to track
    
    Returns:
        List of tracking data dictionaries in chronological order
    """
    if not csv_files:
        return []
    
    print(f"Tracking fixed patch region ({len(point_pairs)} points) across {len(csv_files)} files...")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, point_pairs) for file_path in csv_files]
    
    start_time = time.time()
    
    try:
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(track_fixed_patch_region_in_file, args_list),
                total=len(args_list),
                desc=f"Tracking fixed patch ({len(point_pairs)} points)"
            ))
        
        # Filter out None results and maintain chronological order
        tracking_data = [result for result in results if result is not None]
        
        # Sort by time to ensure chronological order
        tracking_data.sort(key=lambda x: x['time'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Parallel fixed patch tracking completed:")
        print(f"  Found: {len(tracking_data)}/{len(csv_files)} time points")
        print(f"  Time taken: {processing_time:.1f} seconds")
        print(f"  Average: {processing_time/len(csv_files):.3f} seconds per file")
        
        return tracking_data
        
    except Exception as e:
        print(f"Error in parallel fixed patch tracking: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing
        tracking_data = []
        for file_path in tqdm(csv_files, desc="Sequential fixed patch tracking"):
            result = track_fixed_patch_region_in_file((file_path, point_pairs))
            if result:
                tracking_data.append(result)
        
        return tracking_data


def parallel_csv_operation(csv_files: List[Path], operation_func, operation_args: List, 
                          description: str = "Processing") -> List[Any]:
    """
    Generic parallel CSV operation processor.
    
    Args:
        csv_files: List of CSV files to process
        operation_func: Function to apply to each file
        operation_args: Additional arguments for the operation function
        description: Description for progress bar
    
    Returns:
        List of results from the operation function
    """
    if not csv_files:
        return []
    
    print(f"{description} {len(csv_files)} files in parallel...")
    
    # Get optimal process count
    optimal_processes = get_optimal_process_count()
    
    # Prepare arguments for parallel processing
    args_list = [(file_path, *operation_args) for file_path in csv_files]
    
    start_time = time.time()
    
    try:
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(operation_func, args_list),
                total=len(args_list),
                desc=description
            ))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Parallel operation completed:")
        print(f"  Processed: {len(results)} files")
        print(f"  Time taken: {processing_time:.1f} seconds")
        print(f"  Average: {processing_time/len(csv_files):.3f} seconds per file")
        
        return results
        
    except Exception as e:
        print(f"Error in parallel operation: {e}")
        return []


def get_file_processing_stats(csv_files: List[Path]) -> Dict[str, Any]:
    """
    Get statistics about files to be processed for planning purposes.
    
    Args:
        csv_files: List of CSV files
    
    Returns:
        Dictionary with file statistics
    """
    if not csv_files:
        return {}
    
    # Sample a few files to estimate size
    sample_files = csv_files[:min(5, len(csv_files))]
    file_sizes = []
    
    for file_path in sample_files:
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_sizes.append(size_mb)
        except:
            continue
    
    if not file_sizes:
        return {}
    
    avg_file_size = np.mean(file_sizes)
    total_size_gb = (avg_file_size * len(csv_files)) / 1024
    
    # Estimate processing time
    optimal_processes = get_optimal_process_count(avg_file_size)
    estimated_time_parallel = (len(csv_files) * 0.5) / optimal_processes  # Rough estimate
    estimated_time_sequential = len(csv_files) * 2.0  # Rough estimate
    
    return {
        'total_files': len(csv_files),
        'avg_file_size_mb': avg_file_size,
        'total_size_gb': total_size_gb,
        'optimal_processes': optimal_processes,
        'estimated_time_parallel_min': estimated_time_parallel / 60,
        'estimated_time_sequential_min': estimated_time_sequential / 60,
        'speedup_factor': estimated_time_sequential / estimated_time_parallel
    }


# Example usage and testing functions
def test_parallel_processing(subject_name: str = "test_subject"):
    """Test the parallel processing system with a sample subject."""
    
    # Find test files
    test_dir = Path(f"{subject_name}_xyz_tables")
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    csv_files = list(test_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {test_dir}")
        return
    
    # Get file stats
    stats = get_file_processing_stats(csv_files)
    print(f"📊 Test Files Statistics:")
    print(f"   • Files: {stats['total_files']}")
    print(f"   • Total size: {stats['total_size_gb']:.1f} GB")
    print(f"   • Average size: {stats['avg_file_size_mb']:.1f} MB")
    print(f"   • Estimated time: {stats['estimated_time_min']:.1f} minutes")
    
    # Run parallel processing
    processed_files = process_csv_files_parallel(csv_files, subject_name)
    
    print(f"\n✅ Test completed: {len(processed_files)} files processed")
    return processed_files


def track_point_hdf5_parallel(hdf5_file_path: str, patch_number: int, face_index: int) -> List[Dict]:
    """
    Track a single point through time using HDF5 data.
    Memory-efficient version that processes data in adaptive chunks.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_number: Patch number to track
        face_index: Face index to track
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        
        print(f"🚀 Using HDF5 cache for single point tracking: Patch {patch_number}, Face {face_index}")
        
        # Calculate optimal chunk size based on dataset characteristics
        chunk_size = calculate_optimal_chunk_size(hdf5_file_path)
        
        trajectory_data = []
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset references (don't load into memory yet)
            data_dataset = f['data']
            times_dataset = f['times']
            
            # Get properties
            if 'properties' in f.attrs:
                properties = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f.attrs['properties']]
            else:
                return []
            
            # Find property indices
            try:
                x_idx = next(i for i, p in enumerate(properties) if p in ['Position[X] (m)', 'X (m)'])
                y_idx = next(i for i, p in enumerate(properties) if p in ['Position[Y] (m)', 'Y (m)'])
                z_idx = next(i for i, p in enumerate(properties) if p in ['Position[Z] (m)', 'Z (m)'])
                face_idx = next(i for i, p in enumerate(properties) if p == 'Face Index')
                patch_idx = next(i for i, p in enumerate(properties) if p == 'Patch Number')
                pressure_idx = next((i for i, p in enumerate(properties) if p == 'Total Pressure (Pa)'), None)
                velocity_idx = next((i for i, p in enumerate(properties) if p == 'Velocity: Magnitude (m/s)'), None)
                vdotn_idx = next((i for i, p in enumerate(properties) if p == 'VdotN'), None)
            except StopIteration:
                return []
            
            # Get number of time steps
            n_times = data_dataset.shape[0]
            
            # Process in adaptive chunks to avoid memory issues
            for chunk_start in range(0, n_times, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_times)
                
                # Load only this chunk into memory
                data_chunk = data_dataset[chunk_start:chunk_end]
                times_chunk = times_dataset[chunk_start:chunk_end]
                
                # Process each time step in this chunk
                for local_t_idx, time in enumerate(times_chunk):
                    # Find the specific point in this time step
                    matches = np.where(
                        (data_chunk[local_t_idx, :, patch_idx] == patch_number) & 
                        (data_chunk[local_t_idx, :, face_idx] == face_index)
                    )[0]
                    
                    if len(matches) > 0:
                        point_idx = matches[0]
                        point_data = {
                            'time_point': int(time * 1000),
                            'time': time,
                            'x': data_chunk[local_t_idx, point_idx, x_idx],
                            'y': data_chunk[local_t_idx, point_idx, y_idx],
                            'z': data_chunk[local_t_idx, point_idx, z_idx],
                            'patch35_avg_pressure': 0.0,  # Not available in single point analysis
                            'signed_velocity': 0.0,  # Will be calculated if velocity data available
                        }
                        
                        if pressure_idx is not None:
                            pressure_val = data_chunk[local_t_idx, point_idx, pressure_idx]
                            point_data['pressure'] = pressure_val
                            point_data['adjusted_pressure'] = pressure_val  # No patch35 adjustment in HDF5
                            
                        if velocity_idx is not None:
                            point_data['velocity'] = data_chunk[local_t_idx, point_idx, velocity_idx]
                            
                        if vdotn_idx is not None:
                            vdotn_val = data_chunk[local_t_idx, point_idx, vdotn_idx]
                            point_data['vdotn'] = vdotn_val
                            # Use vdotn as signed velocity if available
                            if 'velocity' in point_data:
                                point_data['signed_velocity'] = point_data['velocity'] * np.sign(vdotn_val)
                        
                        trajectory_data.append(point_data)
                
                # Print progress for large datasets
                if chunk_end < n_times:
                    print(f"  Processed {chunk_end}/{n_times} time steps ({chunk_end/n_times*100:.1f}%)")
        
        print(f"✅ HDF5 single point tracking completed: {len(trajectory_data)} time points")
        return trajectory_data
        
    except Exception as e:
        print(f"Error in HDF5 single point tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def track_fixed_patch_region_hdf5_parallel(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Track multiple points (fixed patch region) through time using HDF5 data.
    Memory-efficient version that processes data in adaptive chunks.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries with averaged values
    """
    try:
        import h5py
        
        print(f"🚀 Using HDF5 cache for patch tracking: {len(patch_point_pairs)} points")
        
        # Calculate optimal chunk size based on dataset characteristics
        chunk_size = calculate_optimal_chunk_size(hdf5_file_path)
        
        trajectory_data = []
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset references (don't load into memory yet)
            data_dataset = f['data']
            times_dataset = f['times']
            
            # Get properties
            if 'properties' in f.attrs:
                properties = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f.attrs['properties']]
            else:
                return []
            
            # Find property indices
            try:
                x_idx = next(i for i, p in enumerate(properties) if p in ['Position[X] (m)', 'X (m)'])
                y_idx = next(i for i, p in enumerate(properties) if p in ['Position[Y] (m)', 'Y (m)'])
                z_idx = next(i for i, p in enumerate(properties) if p in ['Position[Z] (m)', 'Z (m)'])
                face_idx = next(i for i, p in enumerate(properties) if p == 'Face Index')
                patch_idx = next(i for i, p in enumerate(properties) if p == 'Patch Number')
                pressure_idx = next((i for i, p in enumerate(properties) if p == 'Total Pressure (Pa)'), None)
                velocity_idx = next((i for i, p in enumerate(properties) if p == 'Velocity: Magnitude (m/s)'), None)
                vdotn_idx = next((i for i, p in enumerate(properties) if p == 'VdotN'), None)
            except StopIteration:
                return []
            
            # Get number of time steps
            n_times = data_dataset.shape[0]
            
            # Process in adaptive chunks to avoid memory issues
            for chunk_start in range(0, n_times, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_times)
                
                # Load only this chunk into memory
                data_chunk = data_dataset[chunk_start:chunk_end]
                times_chunk = times_dataset[chunk_start:chunk_end]
                
                # Process each time step in this chunk
                for local_t_idx, time in enumerate(times_chunk):
                    global_t_idx = chunk_start + local_t_idx
                    region_points = []
                    
                    # Find all points in the region for this time step
                    for patch_num, face_idx_val in patch_point_pairs:
                        matches = np.where(
                            (data_chunk[local_t_idx, :, patch_idx] == patch_num) & 
                            (data_chunk[local_t_idx, :, face_idx] == face_idx_val)
                        )[0]
                        
                        if len(matches) > 0:
                            point_idx = matches[0]
                            point_data = {
                                'x': data_chunk[local_t_idx, point_idx, x_idx],
                                'y': data_chunk[local_t_idx, point_idx, y_idx],
                                'z': data_chunk[local_t_idx, point_idx, z_idx],
                            }
                            
                            if pressure_idx is not None:
                                point_data['pressure'] = data_chunk[local_t_idx, point_idx, pressure_idx]
                            if velocity_idx is not None:
                                point_data['velocity'] = data_chunk[local_t_idx, point_idx, velocity_idx]
                            if vdotn_idx is not None:
                                point_data['vdotn'] = data_chunk[local_t_idx, point_idx, vdotn_idx]
                            
                            region_points.append(point_data)
                    
                    if region_points:
                        # Calculate averages
                        avg_data = {
                            'time_point': int(time * 1000),
                            'time': time,
                            'x': np.mean([p['x'] for p in region_points]),
                            'y': np.mean([p['y'] for p in region_points]),
                            'z': np.mean([p['z'] for p in region_points]),
                            'num_points_in_region': len(region_points),
                            'patch35_avg_pressure': 0.0,  # Not available in patch analysis
                            'signed_velocity': 0.0,  # Will be calculated if velocity data available
                        }
                        
                        if pressure_idx is not None and all('pressure' in p for p in region_points):
                            avg_pressure = np.mean([p['pressure'] for p in region_points])
                            avg_data['pressure'] = avg_pressure
                            avg_data['adjusted_pressure'] = avg_pressure  # No patch35 adjustment in HDF5
                        
                        if velocity_idx is not None and all('velocity' in p for p in region_points):
                            avg_data['velocity'] = np.mean([p['velocity'] for p in region_points])
                            
                        if vdotn_idx is not None and all('vdotn' in p for p in region_points):
                            avg_vdotn = np.mean([p['vdotn'] for p in region_points])
                            avg_data['vdotn'] = avg_vdotn
                            # Use vdotn as signed velocity if available
                            if 'velocity' in avg_data:
                                avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                        
                        trajectory_data.append(avg_data)
                
                # Print progress for large datasets
                if chunk_end < n_times:
                    print(f"  Processed {chunk_end}/{n_times} time steps ({chunk_end/n_times*100:.1f}%)")
        
        print(f"✅ HDF5 patch tracking completed: {len(trajectory_data)} time points")
        return trajectory_data
        
    except Exception as e:
        print(f"Error in HDF5 patch tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def find_initial_region_points_hdf5(hdf5_file_path: str, patch_number: int, face_index: int, 
                                   radius: float = 0.002, normal_angle_threshold: float = 60.0) -> List[Tuple[int, int]]:
    """
    Find all points in the circular region around a center point using HDF5 data (first time step).
    Uses the same spatial analysis logic as the CSV version but operates on HDF5 arrays.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_number: Patch number of the center point
        face_index: Face index of the center point
        radius: Radius of the circular region in meters (default: 2mm)
        normal_angle_threshold: Maximum angle difference from reference normal (degrees)
        
    Returns:
        List of (patch_number, face_index) tuples for all points in the region
    """
    try:
        import h5py
        import numpy as np
        import pandas as pd
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        from scipy.spatial.distance import cdist
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get first time step data
            data = f['data'][0]  # First time step
            
            # Get properties
            if 'properties' in f.attrs:
                properties = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f.attrs['properties']]
            else:
                print("Error: No properties found in HDF5 file")
                return []
            
            # Find property indices
            try:
                x_idx = next(i for i, p in enumerate(properties) if p in ['Position[X] (m)', 'X (m)'])
                y_idx = next(i for i, p in enumerate(properties) if p in ['Position[Y] (m)', 'Y (m)'])
                z_idx = next(i for i, p in enumerate(properties) if p in ['Position[Z] (m)', 'Z (m)'])
                face_idx = next(i for i, p in enumerate(properties) if p == 'Face Index')
                patch_idx = next(i for i, p in enumerate(properties) if p == 'Patch Number')
            except StopIteration:
                print("Error: Could not find required properties in HDF5 file")
                return []
            
            # Find center point
            center_matches = np.where(
                (data[:, patch_idx] == patch_number) & 
                (data[:, face_idx] == face_index)
            )[0]
            
            if len(center_matches) == 0:
                print(f"Error: Center point (Patch {patch_number}, Face {face_index}) not found in HDF5 data")
                return []
            
            center_point_idx = center_matches[0]
            center_point = (
                float(data[center_point_idx, x_idx]),
                float(data[center_point_idx, y_idx]),
                float(data[center_point_idx, z_idx])
            )
            
            # Convert HDF5 data to DataFrame format for compatibility with existing spatial functions
            # Only extract necessary columns to save memory
            df_data = {
                'X (m)': data[:, x_idx],
                'Y (m)': data[:, y_idx], 
                'Z (m)': data[:, z_idx],
                'Patch Number': data[:, patch_idx],
                'Face Index': data[:, face_idx]
            }
            df = pd.DataFrame(df_data)
            
            # Use existing spatial analysis function
            region_points = find_connected_points_with_normal_filter_hdf5(
                df, center_point, radius,
                connectivity_threshold=0.001,
                normal_angle_threshold=normal_angle_threshold
            )
            
            # Get patch/face pairs for all points in the region
            point_pairs = list(zip(region_points['Patch Number'].astype(int), region_points['Face Index'].astype(int)))
            
            print(f"🚀 HDF5: Found {len(point_pairs)} connected points in {radius*1000:.1f}mm radius around Patch {patch_number}, Face {face_index} (normal filter: {normal_angle_threshold}°)")
            return point_pairs
            
    except Exception as e:
        print(f"Error in HDF5 spatial analysis: {e}")
        return []


def calculate_surface_normal_hdf5(points: np.ndarray, center_idx: int, k: int = 10) -> np.ndarray:
    """
    Calculate surface normal at a point using local neighborhood PCA.
    HDF5-compatible version of the spatial analysis function.
    
    Args:
        points: Array of 3D points (N, 3)
        center_idx: Index of the point to calculate normal for
        k: Number of nearest neighbors to use for normal calculation
    
    Returns:
        Unit normal vector (3,)
    """
    if len(points) < 3:
        return np.array([0, 0, 1])  # Default normal if insufficient points
    
    # Find k nearest neighbors
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
    distances, indices = nbrs.kneighbors([points[center_idx]])
    
    # Get local neighborhood points
    local_points = points[indices[0]]
    
    # Center the points
    centered = local_points - local_points.mean(axis=0)
    
    # Compute PCA to find the normal (smallest eigenvector)
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Last row is the normal direction
        return normal / np.linalg.norm(normal)
    except:
        return np.array([0, 0, 1])  # Default normal if SVD fails


def find_connected_points_with_normal_filter_hdf5(df: pd.DataFrame, center_point: Tuple[float, float, float], 
                                                  radius: float, connectivity_threshold: float = 0.001,
                                                  normal_angle_threshold: float = 60.0) -> pd.DataFrame:
    """
    Find connected points within radius that also satisfy surface normal constraints.
    HDF5-compatible version that works with DataFrame created from HDF5 arrays.
    
    Args:
        df: DataFrame with airway surface points (from HDF5 data)
        center_point: (x, y, z) coordinates of center
        radius: Maximum distance from center
        connectivity_threshold: Maximum distance between connected points
        normal_angle_threshold: Maximum angle difference from reference normal (degrees)
    
    Returns:
        DataFrame with filtered connected points
    """
    import numpy as np
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import cdist
    
    # First, get connected points using original method
    distances = np.sqrt(
        (df['X (m)'] - center_point[0])**2 + 
        (df['Y (m)'] - center_point[1])**2 + 
        (df['Z (m)'] - center_point[2])**2
    )
    
    within_radius_mask = distances <= radius
    candidate_points = df[within_radius_mask].copy()
    
    if len(candidate_points) <= 1:
        if len(candidate_points) == 1:
            candidate_points['distance_from_center'] = distances[within_radius_mask]
        return candidate_points
    
    # Extract coordinates for clustering
    coords = candidate_points[['X (m)', 'Y (m)', 'Z (m)']].values
    
    # Use DBSCAN to find connected components
    clustering = DBSCAN(eps=connectivity_threshold, min_samples=1).fit(coords)
    candidate_points['cluster_label'] = clustering.labels_
    
    # Find which cluster contains the center point
    center_distances = cdist([center_point], coords)[0]
    center_point_idx = np.argmin(center_distances)
    center_cluster = candidate_points.iloc[center_point_idx]['cluster_label']
    
    # Keep only points in the same cluster as the center point
    connected_points = candidate_points[candidate_points['cluster_label'] == center_cluster].copy()
    
    # Now apply surface normal filtering
    if len(connected_points) > 3 and radius >= 0.001:  # Only apply normal filtering for patches >= 1mm
        coords_connected = connected_points[['X (m)', 'Y (m)', 'Z (m)']].values
        
        # Find the reference normal using 1mm radius around center point
        ref_radius = min(0.001, radius)  # Use 1mm or smaller if radius is smaller
        ref_distances = np.sqrt(np.sum((coords_connected - np.array(center_point))**2, axis=1))
        ref_mask = ref_distances <= ref_radius
        
        if np.sum(ref_mask) >= 3:  # Need at least 3 points for normal calculation
            ref_points = coords_connected[ref_mask]
            ref_center_idx = np.argmin(ref_distances[ref_mask])
            reference_normal = calculate_surface_normal_hdf5(ref_points, ref_center_idx)
            
            # Calculate normals for all connected points and filter by angle
            valid_indices = []
            for i, point_coords in enumerate(coords_connected):
                if ref_mask[i]:  # Always keep reference points
                    valid_indices.append(i)
                else:
                    # Calculate normal at this point
                    point_normal = calculate_surface_normal_hdf5(coords_connected, i, k=8)
                    
                    # Calculate angle between normals
                    cos_angle = np.dot(reference_normal, point_normal)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
                    angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))  # Use absolute value for undirected normals
                    
                    if angle_deg <= normal_angle_threshold:
                        valid_indices.append(i)
            
            # Filter points based on normal constraints
            if valid_indices:
                connected_points = connected_points.iloc[valid_indices].copy()
                
                # Final connectivity check: Normal filtering might create disconnected components
                # Keep only the component that contains the original center point
                if len(connected_points) > 1:
                    coords_filtered = connected_points[['X (m)', 'Y (m)', 'Z (m)']].values
                    
                    # Re-cluster the normal-filtered points to find disconnected components
                    final_clustering = DBSCAN(eps=connectivity_threshold, min_samples=1).fit(coords_filtered)
                    connected_points['final_cluster_label'] = final_clustering.labels_
                    
                    # Find which cluster contains the center point
                    final_center_distances = cdist([center_point], coords_filtered)[0]
                    final_center_point_idx = np.argmin(final_center_distances)
                    final_center_cluster = connected_points.iloc[final_center_point_idx]['final_cluster_label']
                    
                    # Keep only points in the same final cluster as the center point
                    connected_points = connected_points[connected_points['final_cluster_label'] == final_center_cluster].copy()
                    connected_points = connected_points.drop('final_cluster_label', axis=1)
    
    connected_points['distance_from_center'] = distances[within_radius_mask][connected_points.index]
    connected_points = connected_points.drop('cluster_label', axis=1)
    
    return connected_points


def calculate_optimal_chunk_size(hdf5_file_path: str, target_memory_gb: float = 2.0) -> int:
    """
    Calculate optimal chunk size for HDF5 processing based on dataset characteristics.
    
    Args:
        hdf5_file_path: Path to HDF5 file
        target_memory_gb: Target memory usage per chunk in GB (default: 2GB)
    
    Returns:
        Optimal chunk size (number of time steps)
    """
    try:
        import h5py
        import psutil
        
        with h5py.File(hdf5_file_path, 'r') as f:
            data_shape = f['data'].shape  # (n_times, n_points, n_properties)
            n_times, n_points, n_properties = data_shape
            
            # Calculate memory per time step (in GB)
            # Assuming float64 (8 bytes per value)
            memory_per_timestep_gb = (n_points * n_properties * 8) / (1024**3)
            
            # Get available system memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Use conservative target memory (default 2GB or 25% of available, whichever is smaller)
            safe_target_memory = min(target_memory_gb, available_memory_gb * 0.25)
            
            # Calculate optimal chunk size
            optimal_chunk_size = int(safe_target_memory / memory_per_timestep_gb)
            
            # Apply reasonable bounds
            optimal_chunk_size = max(1, min(optimal_chunk_size, n_times, 1000))
            
            print(f"📊 Dataset info: {n_times} time steps, {n_points:,} points, {n_properties} properties")
            print(f"📊 Memory per time step: {memory_per_timestep_gb:.3f} GB")
            print(f"📊 Available memory: {available_memory_gb:.1f} GB")
            print(f"📊 Target chunk memory: {safe_target_memory:.1f} GB")
            print(f"📊 Optimal chunk size: {optimal_chunk_size} time steps")
            
            return optimal_chunk_size
            
    except Exception as e:
        print(f"Warning: Could not calculate optimal chunk size: {e}")
        print("Using default chunk size of 50")
        return 50


if __name__ == "__main__":
    # Command line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parallel CSV processing")
    parser.add_argument("--subject", default="test_subject", help="Subject name for testing")
    parser.add_argument("--processes", type=int, help="Number of processes to use")
    
    args = parser.parse_args()
    
    if args.processes:
        # Override the process count for testing
        test_files = [Path("dummy.csv")]  # Dummy for stats calculation
        processed = process_csv_files_parallel(test_files, args.subject, n_processes=args.processes)
    else:
        processed = test_parallel_processing(args.subject) 