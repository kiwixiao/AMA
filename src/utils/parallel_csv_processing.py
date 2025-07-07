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
import gc


def get_system_memory_info():
    """Get detailed system memory information."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent_used': memory.percent,
        'free_gb': memory.free / (1024**3)
    }


def calculate_safe_chunk_size(hdf5_file_path: str, target_memory_gb: float = None, safety_factor: float = 0.8) -> int:
    """
    Calculate safe chunk size with enhanced memory monitoring and device adaptation.
    
    Args:
        hdf5_file_path: Path to HDF5 file
        target_memory_gb: Target memory usage per chunk (None for auto-detection)
        safety_factor: Safety factor for memory usage (0.8 = use 80% of available)
    
    Returns:
        Optimal chunk size (number of time steps)
    """
    try:
        import h5py
        import psutil
        
        # Get system memory info
        memory_info = get_system_memory_info()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            data_shape = f['data'].shape  # (n_times, n_points, n_properties)
            n_times, n_points, n_properties = data_shape
            
            # Calculate memory per time step (in GB)
            # Assuming float64 (8 bytes per value)
            memory_per_timestep_gb = (n_points * n_properties * 8) / (1024**3)
            
            # Auto-detect target memory if not specified
            if target_memory_gb is None:
                # Use percentage of available memory based on system size
                available_gb = memory_info['available_gb']
                if available_gb > 32:  # High-end system
                    target_memory_gb = min(8.0, available_gb * 0.4)
                elif available_gb > 16:  # Mid-range system
                    target_memory_gb = min(4.0, available_gb * 0.3)
                elif available_gb > 8:   # Low-end system
                    target_memory_gb = min(2.0, available_gb * 0.25)
                else:  # Very low memory system
                    target_memory_gb = min(1.0, available_gb * 0.2)
            
            # Apply safety factor
            safe_target_memory = target_memory_gb * safety_factor
            
            # Calculate optimal chunk size
            optimal_chunk_size = int(safe_target_memory / memory_per_timestep_gb)
            
            # Apply reasonable bounds
            min_chunk_size = max(1, min(10, n_times))  # At least 1, preferably 10
            max_chunk_size = min(1000, n_times)       # At most 1000 or total
            optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
            
            print(f"🖥️  System Memory Info:")
            print(f"   Total RAM: {memory_info['total_gb']:.1f} GB")
            print(f"   Available RAM: {memory_info['available_gb']:.1f} GB")
            print(f"   Used RAM: {memory_info['used_gb']:.1f} GB ({memory_info['percent_used']:.1f}%)")
            print(f"📊 Dataset Info:")
            print(f"   Time steps: {n_times:,}")
            print(f"   Points per step: {n_points:,}")
            print(f"   Properties: {n_properties}")
            print(f"   Memory per timestep: {memory_per_timestep_gb:.3f} GB")
            print(f"   Total dataset size: {memory_per_timestep_gb * n_times:.1f} GB")
            print(f"🔧 Chunk Configuration:")
            print(f"   Target chunk memory: {safe_target_memory:.1f} GB")
            print(f"   Optimal chunk size: {optimal_chunk_size} time steps")
            print(f"   Memory per chunk: {optimal_chunk_size * memory_per_timestep_gb:.2f} GB")
            print(f"   Safety factor: {safety_factor:.1f}")
            
            return optimal_chunk_size
            
    except Exception as e:
        print(f"Warning: Could not calculate optimal chunk size: {e}")
        # Conservative fallback
        memory_info = get_system_memory_info()
        if memory_info['available_gb'] > 16:
            return 100
        elif memory_info['available_gb'] > 8:
            return 50
        else:
            return 10


def monitor_memory_usage(operation_name: str = "HDF5 Operation"):
    """Monitor and log memory usage during operations."""
    memory_info = get_system_memory_info()
    if memory_info['percent_used'] > 85:
        print(f"⚠️  High memory usage detected during {operation_name}:")
        print(f"   RAM usage: {memory_info['percent_used']:.1f}%")
        print(f"   Available: {memory_info['available_gb']:.1f} GB")
        if memory_info['percent_used'] > 95:
            print(f"🚨 Critical memory usage! Consider reducing chunk size.")
    return memory_info


def filter_data_by_breathing_cycle_chunked(data_info: Dict, subject_name: str) -> Dict:
    """
    Memory-safe version of filter_data_by_breathing_cycle using chunked loading.
    
    Args:
        data_info: Dictionary containing the HDF5 file path and metadata
        subject_name: Name of the subject (e.g., 'OSAMRI007')
        
    Returns:
        dict: Updated data_info dictionary with filtered data path
    """
    import h5py
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    print("🔍 Filtering data based on breathing cycle (memory-safe chunked version)...")
    
    # Load flow profile data
    flow_data = pd.read_csv(f'{subject_name}FlowProfile.csv')
    
    # Get column names and identify time and flow columns
    columns = flow_data.columns.tolist()
    print(f"Flow profile columns: {columns}")
    
    # Try to identify time and flow columns
    time_col = next((col for col in columns if 'time' in col.lower()), columns[0])
    flow_col = next((col for col in columns if 'flow' in col.lower() or 'mass' in col.lower()), columns[1])
    
    print(f"Using columns: Time = '{time_col}', Flow = '{flow_col}'")
    
    # Convert time from seconds to milliseconds if needed
    flow_times = flow_data[time_col].values
    if 'time (s)' in columns:  # If time is in seconds, convert to ms
        flow_times = flow_times * 1000
        print("Converting time from seconds to milliseconds")
    
    flow_values = flow_data[flow_col].values
    
    # Find zero crossings to identify breathing cycles
    zero_crossings = np.where(np.diff(np.signbit(flow_values)))[0]
    
    # Get the time points for zero crossings
    cycle_times = flow_times[zero_crossings]
    
    print(f"Found {len(zero_crossings)} zero crossings")
    print(f"Time range: {cycle_times[0]:.2f} ms to {cycle_times[-1]:.2f} ms")
    
    # Calculate optimal chunk size for this dataset
    chunk_size = calculate_safe_chunk_size(data_info['file_path'])
    
    # Process in chunks to find valid time indices
    valid_indices = []
    start_time = cycle_times[0]
    end_time = cycle_times[-1]
    
    print(f"🔄 Processing dataset in chunks of {chunk_size} time steps...")
    
    with h5py.File(data_info['file_path'], 'r') as f:
        print(f"Available datasets in HDF5 file: {list(f.keys())}")
        
        # Get dataset references (don't load into memory)
        data_dataset = f['data']
        times_dataset = f['times']
        n_times = times_dataset.shape[0]
        
        # Get properties
        properties = None
        if 'properties' in f.attrs:
            properties = f.attrs['properties']
            print("Found properties in HDF5 file")
        else:
            print("No properties found in HDF5 file")
        
        # Process times in chunks to find valid indices
        for chunk_start in range(0, n_times, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_times)
            
            # Load only time chunk
            times_chunk = times_dataset[chunk_start:chunk_end]
            
            # Find indices within breathing cycle for this chunk
            chunk_valid_mask = (times_chunk >= start_time) & (times_chunk <= end_time)
            chunk_valid_indices = np.where(chunk_valid_mask)[0] + chunk_start
            
            valid_indices.extend(chunk_valid_indices)
            
            # Monitor memory usage
            if chunk_end % (chunk_size * 10) == 0:  # Every 10 chunks
                monitor_memory_usage(f"Breathing cycle filtering (chunk {chunk_end//chunk_size})")
    
    valid_indices = np.array(valid_indices)
    
    if len(valid_indices) == 0:
        print("Warning: No valid breathing cycles found!")
        return data_info
    
    print(f"Found {len(valid_indices)} timesteps within breathing cycle")
    
    # Create filtered dataset using chunked approach
    filtered_file_path = Path(f'{subject_name}_cfd_data_filtered.h5')
    
    print(f"🔄 Creating filtered dataset: {filtered_file_path}")
    
    with h5py.File(data_info['file_path'], 'r') as f_in:
        with h5py.File(filtered_file_path, 'w') as f_out:
            # Get input dataset info
            input_data = f_in['data']
            input_times = f_in['times']
            
            # Create output datasets
            n_valid = len(valid_indices)
            n_points = input_data.shape[1]
            n_properties = input_data.shape[2]
            
            f_out.create_dataset('times', data=input_times[valid_indices])
            output_data = f_out.create_dataset('data', (n_valid, n_points, n_properties), dtype='float64')
            
            # Save properties if they exist
            if properties is not None:
                f_out.attrs['properties'] = properties
            
            # Copy data in chunks
            print(f"🔄 Copying filtered data ({n_valid} timesteps)...")
            
            for i, global_idx in enumerate(tqdm(valid_indices, desc="Copying filtered data")):
                # Load single timestep
                timestep_data = input_data[global_idx]
                output_data[i] = timestep_data
                
                # Monitor memory periodically
                if i % 100 == 0:
                    monitor_memory_usage(f"Filtering copy (step {i+1}/{n_valid})")
                    gc.collect()  # Force garbage collection
            
            # Add metadata about filtering
            f_out.attrs['num_original_timesteps'] = n_times
            f_out.attrs['num_filtered_timesteps'] = len(valid_indices)
            f_out.attrs['filtering_description'] = 'Data filtered to include only complete breathing cycles'
            f_out.attrs['subject_name'] = subject_name
    
    print(f"✅ Filtered data saved successfully:")
    print(f"   Original timesteps: {n_times}")
    print(f"   Filtered timesteps: {len(valid_indices)}")
    print(f"   Filtered file: {filtered_file_path}")
    
    # Force garbage collection
    gc.collect()
    
    # Return updated data info
    return {
        'file_path': str(filtered_file_path),
        'times': input_times[valid_indices],
        'properties': properties
    }


def track_point_movement_3d_chunked(data_info: Dict, patch_number: int, face_index: int) -> Dict:
    """
    Memory-safe version of track_point_movement_3d using chunked loading.
    
    Args:
        data_info: Dictionary containing the HDF5 file path and metadata
        patch_number: The patch number to track
        face_index: The face index within the patch to track
        
    Returns:
        dict: Dictionary containing trajectory information
    """
    import h5py
    import numpy as np
    
    print(f"🎯 Tracking point movement (memory-safe): Patch {patch_number}, Face {face_index}")
    
    # Calculate optimal chunk size
    chunk_size = calculate_safe_chunk_size(data_info['file_path'])
    
    trajectory = []
    velocities = []
    accelerations = []
    pressures = []
    vdotn_values = []
    
    prev_pos = None
    prev_vel = None
    prev_time = None
    
    with h5py.File(data_info['file_path'], 'r') as f:
        # Get dataset references
        data_dataset = f['data']
        times_dataset = f['times']
        n_times = times_dataset.shape[0]
        
        # Get properties from HDF5 file
        if 'properties' in f.attrs:
            properties = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f.attrs['properties']]
            print(f"Available properties: {properties}")
        else:
            print("Error: No properties found in HDF5 file")
            return {}
        
        # Find indices for required properties
        try:
            # Look for coordinate columns - try both Position[X] and X formats
            x_idx = next(i for i, p in enumerate(properties) if p in ['Position[X] (m)', 'X (m)'])
            y_idx = next(i for i, p in enumerate(properties) if p in ['Position[Y] (m)', 'Y (m)'])
            z_idx = next(i for i, p in enumerate(properties) if p in ['Position[Z] (m)', 'Z (m)'])
            
            # Look for face and patch columns
            face_idx = next(i for i, p in enumerate(properties) if p == 'Face Index')
            patch_idx = next(i for i, p in enumerate(properties) if p == 'Patch Number')
            
            # Look for additional properties
            pressure_idx = next((i for i, p in enumerate(properties) if p == 'Total Pressure (Pa)'), None)
            velocity_idx = next((i for i, p in enumerate(properties) if p == 'Velocity: Magnitude (m/s)'), None)
            vdotn_idx = next((i for i, p in enumerate(properties) if p == 'VdotN'), None)
            
            print("Found property indices:")
            print(f"  Coordinates: X={properties[x_idx]}, Y={properties[y_idx]}, Z={properties[z_idx]}")
            print(f"  Face Index: {properties[face_idx]}, Patch Number: {properties[patch_idx]}")
            if pressure_idx is not None:
                print(f"  Pressure: {properties[pressure_idx]}")
            if velocity_idx is not None:
                print(f"  Velocity: {properties[velocity_idx]}")
            if vdotn_idx is not None:
                print(f"  VdotN: {properties[vdotn_idx]}")
            
        except StopIteration as e:
            print(f"Error: Could not find all required properties in the data")
            print(f"Available properties: {properties}")
            return {}
        
        # Calculate number of chunks for progress bar
        n_chunks = (n_times + chunk_size - 1) // chunk_size
        
        # Process in chunks
        print(f"🔄 Processing {n_times} time steps in {n_chunks} chunks of {chunk_size} steps each...")
        
        with tqdm(total=n_times, desc="Tracking point movement", unit="timesteps") as pbar:
            for chunk_start in range(0, n_times, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_times)
                
                # Load only this chunk
                print(f"  📦 Loading chunk {chunk_start//chunk_size + 1}/{n_chunks} (timesteps {chunk_start}-{chunk_end-1})")
                data_chunk = data_dataset[chunk_start:chunk_end]
                times_chunk = times_dataset[chunk_start:chunk_end]
                
                # Process each time step in this chunk
                for local_t_idx, time in enumerate(times_chunk):
                    # Find the point with matching patch and face
                    matches = np.where(
                        (data_chunk[local_t_idx, :, patch_idx] == patch_number) & 
                        (data_chunk[local_t_idx, :, face_idx] == face_index)
                    )[0]
                    
                    if len(matches) == 0:
                        # Update progress bar even for missing points
                        pbar.update(1)
                        continue
                    elif len(matches) > 1:
                        print(f"Warning: Multiple matches found at time {time}, using first")
                    
                    point_idx = matches[0]
                    
                    # Get position
                    pos = data_chunk[local_t_idx, point_idx, [x_idx, y_idx, z_idx]]
                    trajectory.append((time, pos))
                    
                    # Get pressure if available
                    if pressure_idx is not None:
                        pressure = data_chunk[local_t_idx, point_idx, pressure_idx]
                        pressures.append((time, pressure))
                    
                    # Get VdotN if available
                    if vdotn_idx is not None:
                        vdotn = data_chunk[local_t_idx, point_idx, vdotn_idx]
                        vdotn_values.append((time, vdotn))
                    
                    # Calculate velocity if we have a previous position
                    if prev_pos is not None and prev_time is not None:
                        dt = time - prev_time
                        if dt > 0:
                            # Calculate velocity
                            vel = (pos - prev_pos) / dt
                            velocities.append((time, vel))
                            
                            # Calculate acceleration if we have a previous velocity
                            if prev_vel is not None:
                                acc = (vel - prev_vel) / dt
                                accelerations.append((time, acc))
                            
                            prev_vel = vel
                    
                    prev_pos = pos
                    prev_time = time
                    
                    # Update progress bar
                    pbar.update(1)
                
                # Force garbage collection after each chunk
                gc.collect()
    
    if not trajectory:
        print("Warning: No trajectory points found!")
        return {}
    
    # Convert lists to numpy arrays
    times = np.array([t for t, _ in trajectory])
    positions = np.array([p for _, p in trajectory])
    velocities = np.array([v for _, v in velocities]) if velocities else np.array([])
    accelerations = np.array([a for _, a in accelerations]) if accelerations else np.array([])
    pressures = np.array([p for _, p in pressures]) if pressures else np.array([])
    vdotn_values = np.array([v for _, v in vdotn_values]) if vdotn_values else np.array([])
    
    result = {
        'times': times,
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations
    }
    
    # Add optional data if available
    if len(pressures) > 0:
        result['pressures'] = pressures
    if len(vdotn_values) > 0:
        result['vdotn'] = vdotn_values
    
    print(f"✅ Point tracking completed: {len(trajectory)} time points")
    
    # Force garbage collection
    gc.collect()
    
    return result


def find_initial_region_points_hdf5_safe(hdf5_file_path: str, patch_number: int, face_index: int, 
                                         radius: float = 0.002, normal_angle_threshold: float = 60.0) -> List[Tuple[int, int]]:
    """
    Memory-safe version of find_initial_region_points_hdf5 that only loads the first timestep.
    
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
        
        print(f"🔍 Finding initial region points (memory-safe): Patch {patch_number}, Face {face_index}")
        print(f"   Radius: {radius*1000:.1f}mm, Normal threshold: {normal_angle_threshold}°")
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get first time step data only (much smaller than full dataset)
            first_timestep_data = f['data'][0]  # Shape: (n_points, n_properties)
            
            # Monitor memory usage
            memory_info = monitor_memory_usage("Initial region analysis")
            
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
                (first_timestep_data[:, patch_idx] == patch_number) & 
                (first_timestep_data[:, face_idx] == face_index)
            )[0]
            
            if len(center_matches) == 0:
                print(f"Error: Center point (Patch {patch_number}, Face {face_index}) not found in HDF5 data")
                return []
            
            center_point_idx = center_matches[0]
            center_point = (
                float(first_timestep_data[center_point_idx, x_idx]),
                float(first_timestep_data[center_point_idx, y_idx]),
                float(first_timestep_data[center_point_idx, z_idx])
            )
            
            print(f"   Center point: ({center_point[0]:.6f}, {center_point[1]:.6f}, {center_point[2]:.6f})")
            
            # Convert to DataFrame for spatial analysis (only necessary columns)
            df_data = {
                'X (m)': first_timestep_data[:, x_idx],
                'Y (m)': first_timestep_data[:, y_idx], 
                'Z (m)': first_timestep_data[:, z_idx],
                'Patch Number': first_timestep_data[:, patch_idx],
                'Face Index': first_timestep_data[:, face_idx]
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
            
            print(f"✅ Found {len(point_pairs)} connected points in {radius*1000:.1f}mm radius")
            
            # Force garbage collection
            gc.collect()
            
            return point_pairs
            
    except Exception as e:
        print(f"Error in HDF5 spatial analysis: {e}")
        import traceback
        traceback.print_exc()
        return []


# Update the existing calculate_optimal_chunk_size function to use the new safe version
def calculate_optimal_chunk_size(hdf5_file_path: str, target_memory_gb: float = 2.0) -> int:
    """
    Legacy function - redirects to calculate_safe_chunk_size for backward compatibility.
    """
    return calculate_safe_chunk_size(hdf5_file_path, target_memory_gb)


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
        chunk_size = calculate_safe_chunk_size(hdf5_file_path)
        
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
            
            # Calculate number of chunks for progress bar
            n_chunks = (n_times + chunk_size - 1) // chunk_size
            
            print(f"🔄 Processing {n_times} time steps in {n_chunks} chunks of {chunk_size} steps each...")
            
            # Process in adaptive chunks to avoid memory issues
            with tqdm(total=n_times, desc="Tracking single point", unit="timesteps") as pbar:
                for chunk_start in range(0, n_times, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_times)
                    
                    # Load only this chunk into memory
                    print(f"  📦 Loading chunk {chunk_start//chunk_size + 1}/{n_chunks} (timesteps {chunk_start}-{chunk_end-1})")
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
                        
                        # Update progress bar
                        pbar.update(1)
                    
                    # Force garbage collection after each chunk
                    gc.collect()
        
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
        chunk_size = calculate_safe_chunk_size(hdf5_file_path)
        
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
            
            # Calculate number of chunks for progress bar
            n_chunks = (n_times + chunk_size - 1) // chunk_size
            
            print(f"🔄 Processing {n_times} time steps in {n_chunks} chunks of {chunk_size} steps each...")
            
            # Process in adaptive chunks to avoid memory issues
            with tqdm(total=n_times, desc="Loading HDF5 chunks", unit="timesteps") as pbar:
                for chunk_start in range(0, n_times, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_times)
                    
                    # Load only this chunk into memory
                    print(f"  📦 Loading chunk {chunk_start//chunk_size + 1}/{n_chunks} (timesteps {chunk_start}-{chunk_end-1})")
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
                        
                        # Update progress bar
                        pbar.update(1)
                    
                    # Force garbage collection after each chunk
                    gc.collect()
        
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


def track_fixed_patch_region_hdf5_optimized(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Optimized version for large datasets (24mmesh+) with minimal overhead.
    Reduces I/O calls and progress output for maximum speed.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries with averaged values
    """
    try:
        import h5py
        
        print(f"🚀 Using optimized HDF5 cache for large patch tracking: {len(patch_point_pairs)} points")
        
        # Use larger chunk sizes for big datasets to reduce I/O overhead
        chunk_size = calculate_safe_chunk_size(hdf5_file_path, target_memory_gb=8.0)  # Use more memory
        
        trajectory_data = []
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset references
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
            
            # Get dataset info
            n_times = data_dataset.shape[0]
            n_chunks = (n_times + chunk_size - 1) // chunk_size
            
            print(f"🔄 Processing {n_times:,} timesteps in {n_chunks} optimized chunks...")
            
            # Process with minimal overhead
            processed_timesteps = 0
            
            for chunk_start in range(0, n_times, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_times)
                
                # Minimal progress output (only every 10 chunks for large datasets)
                if chunk_start % (chunk_size * 10) == 0 or chunk_end == n_times:
                    progress_pct = (chunk_end / n_times) * 100
                    print(f"  Progress: {progress_pct:.1f}% ({chunk_end:,}/{n_times:,} timesteps)")
                
                # Load chunk with optimized read
                data_chunk = data_dataset[chunk_start:chunk_end]
                times_chunk = times_dataset[chunk_start:chunk_end]
                
                # Process timesteps in batch
                for local_t_idx, time in enumerate(times_chunk):
                    region_points = []
                    
                    # Vectorized point finding for better performance
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
                        # Optimized averaging using numpy
                        positions = np.array([[p['x'], p['y'], p['z']] for p in region_points])
                        avg_pos = np.mean(positions, axis=0)
                        
                        avg_data = {
                            'time_point': int(time * 1000),
                            'time': time,
                            'x': avg_pos[0],
                            'y': avg_pos[1], 
                            'z': avg_pos[2],
                            'num_points_in_region': len(region_points),
                            'patch35_avg_pressure': 0.0,
                            'signed_velocity': 0.0,
                        }
                        
                        if pressure_idx is not None and all('pressure' in p for p in region_points):
                            pressures = np.array([p['pressure'] for p in region_points])
                            avg_data['pressure'] = np.mean(pressures)
                            avg_data['adjusted_pressure'] = avg_data['pressure']
                        
                        if velocity_idx is not None and all('velocity' in p for p in region_points):
                            velocities = np.array([p['velocity'] for p in region_points])
                            avg_data['velocity'] = np.mean(velocities)
                            
                        if vdotn_idx is not None and all('vdotn' in p for p in region_points):
                            vdotns = np.array([p['vdotn'] for p in region_points])
                            avg_vdotn = np.mean(vdotns)
                            avg_data['vdotn'] = avg_vdotn
                            if 'velocity' in avg_data:
                                avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                        
                        trajectory_data.append(avg_data)
                    
                    processed_timesteps += 1
                
                # Less frequent garbage collection
                if chunk_start % (chunk_size * 5) == 0:
                    gc.collect()
        
        print(f"✅ Optimized HDF5 patch tracking completed: {len(trajectory_data):,} time points")
        return trajectory_data
        
    except Exception as e:
        print(f"Error in optimized HDF5 patch tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def auto_select_hdf5_tracking_method(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Automatically select the best HDF5 tracking method based on dataset size.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import os
        
        # Get file size
        file_size_gb = os.path.getsize(hdf5_file_path) / (1024**3)
        
        with h5py.File(hdf5_file_path, 'r') as f:
            n_times = f['data'].shape[0]
        
        # Use optimized version for large datasets
        if file_size_gb > 50 or n_times > 10000:  # >50GB or >10k timesteps
            print(f"📊 Large dataset detected ({file_size_gb:.1f}GB, {n_times:,} timesteps)")
            print("🚀 Using optimized tracking method for maximum speed...")
            return track_fixed_patch_region_hdf5_optimized(hdf5_file_path, patch_point_pairs)
        else:
            print(f"📊 Standard dataset ({file_size_gb:.1f}GB, {n_times:,} timesteps)")
            print("🔄 Using standard tracking method with progress bars...")
            return track_fixed_patch_region_hdf5_parallel(hdf5_file_path, patch_point_pairs)
            
    except Exception as e:
        print(f"Warning: Could not determine optimal method, using standard: {e}")
        return track_fixed_patch_region_hdf5_parallel(hdf5_file_path, patch_point_pairs)


def process_hdf5_chunk_parallel(args):
    """
    Process a single HDF5 chunk for patch tracking in parallel.
    
    Args:
        args: Tuple of (hdf5_file_path, chunk_start, chunk_end, patch_point_pairs, property_indices)
    
    Returns:
        List of trajectory data for this chunk
    """
    hdf5_file_path, chunk_start, chunk_end, patch_point_pairs, property_indices = args
    
    try:
        import h5py
        import numpy as np
        
        trajectory_data = []
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load only this chunk
            data_chunk = f['data'][chunk_start:chunk_end]
            times_chunk = f['times'][chunk_start:chunk_end]
            
            # Unpack property indices
            x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
            pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
            
            # Process each timestep in this chunk
            for local_t_idx, time in enumerate(times_chunk):
                region_points = []
                
                # Find all points in the region for this timestep
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
                    # Optimized averaging using numpy
                    positions = np.array([[p['x'], p['y'], p['z']] for p in region_points])
                    avg_pos = np.mean(positions, axis=0)
                    
                    avg_data = {
                        'time_point': int(time * 1000),
                        'time': time,
                        'x': avg_pos[0],
                        'y': avg_pos[1],
                        'z': avg_pos[2],
                        'num_points_in_region': len(region_points),
                        'patch35_avg_pressure': 0.0,
                        'signed_velocity': 0.0,
                    }
                    
                    if pressure_idx is not None and all('pressure' in p for p in region_points):
                        pressures = np.array([p['pressure'] for p in region_points])
                        avg_data['pressure'] = np.mean(pressures)
                        avg_data['adjusted_pressure'] = avg_data['pressure']
                    
                    if velocity_idx is not None and all('velocity' in p for p in region_points):
                        velocities = np.array([p['velocity'] for p in region_points])
                        avg_data['velocity'] = np.mean(velocities)
                        
                    if vdotn_idx is not None and all('vdotn' in p for p in region_points):
                        vdotns = np.array([p['vdotn'] for p in region_points])
                        avg_vdotn = np.mean(vdotns)
                        avg_data['vdotn'] = avg_vdotn
                        if 'velocity' in avg_data:
                            avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                    
                    trajectory_data.append(avg_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
        return []


def track_fixed_patch_region_hdf5_parallel(csv_files: List[Path], point_pairs: List[Tuple[int, int]]) -> List[Dict]:
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


def process_hdf5_point_chunk_parallel(args):
    """
    Process a single HDF5 chunk for single point tracking in parallel.
    
    Args:
        args: Tuple of (hdf5_file_path, chunk_start, chunk_end, patch_number, face_index, property_indices)
    
    Returns:
        List of point trajectory data for this chunk
    """
    hdf5_file_path, chunk_start, chunk_end, patch_number, face_index, property_indices = args
    
    try:
        import h5py
        import numpy as np
        
        trajectory_data = []
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load only this chunk
            data_chunk = f['data'][chunk_start:chunk_end]
            times_chunk = f['times'][chunk_start:chunk_end]
            
            # Unpack property indices
            x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
            pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
            
            # Process each timestep in this chunk
            for local_t_idx, time in enumerate(times_chunk):
                # Find the specific point
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
                        'patch35_avg_pressure': 0.0,
                        'signed_velocity': 0.0,
                    }
                    
                    if pressure_idx is not None:
                        pressure_val = data_chunk[local_t_idx, point_idx, pressure_idx]
                        point_data['pressure'] = pressure_val
                        point_data['adjusted_pressure'] = pressure_val
                        
                    if velocity_idx is not None:
                        point_data['velocity'] = data_chunk[local_t_idx, point_idx, velocity_idx]
                        
                    if vdotn_idx is not None:
                        vdotn_val = data_chunk[local_t_idx, point_idx, vdotn_idx]
                        point_data['vdotn'] = vdotn_val
                        if 'velocity' in point_data:
                            point_data['signed_velocity'] = point_data['velocity'] * np.sign(vdotn_val)
                    
                    trajectory_data.append(point_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing point chunk {chunk_start}-{chunk_end}: {e}")
        return []


def track_point_hdf5_parallel_multicore(hdf5_file_path: str, patch_number: int, face_index: int) -> List[Dict]:
    """
    Multi-core parallel version for HDF5 single point tracking.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_number: Patch number to track
        face_index: Face index to track
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import multiprocessing as mp
        
        print(f"🚀 Using multi-core HDF5 point tracking: Patch {patch_number}, Face {face_index}")
        
        # Calculate chunk size and available processes
        chunk_size = calculate_safe_chunk_size(hdf5_file_path, target_memory_gb=4.0)  # Conservative for parallel
        optimal_processes = get_optimal_process_count()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset info
            n_times = f['data'].shape[0]
            
            # Get properties and indices
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
            
            # Pack property indices for parallel processing
            property_indices = (x_idx, y_idx, z_idx, face_idx, patch_idx, pressure_idx, velocity_idx, vdotn_idx)
        
        # Create chunk arguments for parallel processing
        chunk_args = []
        for chunk_start in range(0, n_times, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_times)
            chunk_args.append((hdf5_file_path, chunk_start, chunk_end, patch_number, face_index, property_indices))
        
        n_chunks = len(chunk_args)
        print(f"🔄 Processing {n_times:,} timesteps in {n_chunks} chunks using {optimal_processes} CPU cores...")
        
        # Process chunks in parallel
        trajectory_data = []
        
        start_time = time.time()
        
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(process_hdf5_point_chunk_parallel, chunk_args),
                total=len(chunk_args),
                desc=f"Multi-core point tracking ({optimal_processes} cores)"
            ))
        
        # Combine results from all chunks
        for chunk_results in results:
            trajectory_data.extend(chunk_results)
        
        # Sort by time to ensure chronological order
        trajectory_data.sort(key=lambda x: x['time'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Multi-core HDF5 point tracking completed:")
        print(f"   Tracked: {len(trajectory_data):,} time points")
        print(f"   Time: {processing_time:.1f} seconds")
        print(f"   Speed: {len(trajectory_data)/processing_time:.1f} points/second")
        print(f"   CPU cores used: {optimal_processes}")
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in multi-core HDF5 point tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def auto_select_hdf5_tracking_method(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Automatically select the best HDF5 tracking method based on dataset size.
    Prioritizes OPTIMIZED approach with direct array indexing for large datasets.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import os
        
        # Get file size and system info
        file_size_gb = os.path.getsize(hdf5_file_path) / (1024**3)
        memory_info = get_system_memory_info()
        cpu_cores = mp.cpu_count()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            n_times = f['data'].shape[0]
        
        print(f"📊 Dataset: {file_size_gb:.1f}GB, {n_times:,} timesteps")
        print(f"🖥️  System: {cpu_cores} cores, {memory_info['available_gb']:.1f}GB RAM available")
        
        # Use OPTIMIZED approach for datasets with multiple timesteps (User's optimization request)
        if n_times > 5:  # For any dataset with more than 5 timesteps - enables testing with 23mmesh
            print(f"🚀 Using OPTIMIZED approach (find indices once, direct array access)")
            return auto_select_optimized_hdf5_tracking_method(hdf5_file_path, patch_point_pairs)
        # Use memory-safe multi-core approach for small datasets  
        elif cpu_cores >= 4 and memory_info['available_gb'] > 8:
            print(f"🚀 Using MEMORY-SAFE multi-core processing ({cpu_cores} cores)")
            return track_fixed_patch_region_hdf5_memory_safe_multicore(hdf5_file_path, patch_point_pairs)
        elif file_size_gb > 50 or n_times > 10000:
            print(f"🔄 Using optimized single-thread processing (memory constraints)")
            return track_fixed_patch_region_hdf5_optimized(hdf5_file_path, patch_point_pairs)
        else:
            print(f"📊 Using standard processing with progress bars")
            return track_fixed_patch_region_hdf5_parallel(hdf5_file_path, patch_point_pairs)
            
    except Exception as e:
        print(f"Warning: Could not determine optimal method, using standard: {e}")
        return track_fixed_patch_region_hdf5_parallel(hdf5_file_path, patch_point_pairs)


def auto_select_hdf5_point_tracking_method(hdf5_file_path: str, patch_number: int, face_index: int) -> List[Dict]:
    """
    Automatically select the best HDF5 point tracking method based on dataset size.
    Prioritizes memory-efficient multi-core approach (load once, process in parallel).
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_number: Patch number to track
        face_index: Face index to track
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import os
        
        # Get file size and system info
        file_size_gb = os.path.getsize(hdf5_file_path) / (1024**3)
        memory_info = get_system_memory_info()
        cpu_cores = mp.cpu_count()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            n_times = f['data'].shape[0]
        
        print(f"📊 Dataset: {file_size_gb:.1f}GB, {n_times:,} timesteps")
        print(f"🖥️  System: {cpu_cores} cores, {memory_info['available_gb']:.1f}GB RAM available")
        
        # Use memory-safe multi-core approach (avoids data duplication) - USER'S SUGGESTION
        if cpu_cores >= 4 and memory_info['available_gb'] > 8:
            print(f"🚀 Using MEMORY-SAFE multi-core point tracking ({cpu_cores} cores)")
            return track_point_hdf5_memory_safe_multicore(hdf5_file_path, patch_number, face_index)
        else:
            print(f"📊 Using standard point tracking with progress bars")
            return track_point_hdf5_parallel(hdf5_file_path, patch_number, face_index)
            
    except Exception as e:
        print(f"Warning: Could not determine optimal method, using standard: {e}")
        return track_point_hdf5_parallel(hdf5_file_path, patch_number, face_index)


def can_load_entire_dataset(hdf5_file_path: str, safety_factor: float = 0.7) -> bool:
    """
    Check if we can safely load the entire dataset into memory.
    
    Args:
        hdf5_file_path: Path to HDF5 file
        safety_factor: Safety factor for memory usage (0.7 = use 70% of available memory)
    
    Returns:
        True if dataset can be loaded entirely into memory
    """
    try:
        import h5py
        import os
        
        # Get file size and system memory
        file_size_gb = os.path.getsize(hdf5_file_path) / (1024**3)
        memory_info = get_system_memory_info()
        
        # Estimate memory needed (file size + processing overhead)
        estimated_memory_gb = file_size_gb * 1.2  # 20% overhead for processing
        safe_memory_limit = memory_info['available_gb'] * safety_factor
        
        can_load = estimated_memory_gb <= safe_memory_limit
        
        print(f"📊 Memory check:")
        print(f"   Dataset size: {file_size_gb:.1f}GB")
        print(f"   Available RAM: {memory_info['available_gb']:.1f}GB")
        print(f"   Estimated need: {estimated_memory_gb:.1f}GB")
        print(f"   Safe limit: {safe_memory_limit:.1f}GB")
        print(f"   Can load entire dataset: {'✅ YES' if can_load else '❌ NO'}")
        
        return can_load
        
    except Exception as e:
        print(f"Error checking memory: {e}")
        return False


def process_timestep_range_parallel(args):
    """
    Process a range of timesteps for patch tracking in parallel.
    This works on already-loaded data in memory (no I/O operations).
    
    Args:
        args: Tuple of (data_chunk, times_chunk, start_idx, end_idx, patch_point_pairs, property_indices)
    
    Returns:
        List of trajectory data for this timestep range
    """
    data_chunk, times_chunk, start_idx, end_idx, patch_point_pairs, property_indices = args
    
    try:
        import numpy as np
        
        trajectory_data = []
        
        # Unpack property indices
        x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
        pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
        
        # Process each timestep in this range
        for local_t_idx in range(start_idx, end_idx):
            time = times_chunk[local_t_idx]
            region_points = []
            
            # Find all points in the region for this timestep
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
                # Optimized averaging using numpy
                positions = np.array([[p['x'], p['y'], p['z']] for p in region_points])
                avg_pos = np.mean(positions, axis=0)
                
                avg_data = {
                    'time_point': int(time * 1000),
                    'time': time,
                    'x': avg_pos[0],
                    'y': avg_pos[1],
                    'z': avg_pos[2],
                    'num_points_in_region': len(region_points),
                    'patch35_avg_pressure': 0.0,
                    'signed_velocity': 0.0,
                }
                
                if pressure_idx is not None and all('pressure' in p for p in region_points):
                    pressures = np.array([p['pressure'] for p in region_points])
                    avg_data['pressure'] = np.mean(pressures)
                    avg_data['adjusted_pressure'] = avg_data['pressure']
                
                if velocity_idx is not None and all('velocity' in p for p in region_points):
                    velocities = np.array([p['velocity'] for p in region_points])
                    avg_data['velocity'] = np.mean(velocities)
                    
                if vdotn_idx is not None and all('vdotn' in p for p in region_points):
                    vdotns = np.array([p['vdotn'] for p in region_points])
                    avg_vdotn = np.mean(vdotns)
                    avg_data['vdotn'] = avg_vdotn
                    if 'velocity' in avg_data:
                        avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                
                trajectory_data.append(avg_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing timestep range {start_idx}-{end_idx}: {e}")
        return []


def track_fixed_patch_region_hdf5_memory_efficient_multicore(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Memory-efficient multi-core HDF5 patch tracking.
    Loads entire dataset once (if memory allows) and splits work across processes.
    This is the approach suggested by the user - much more efficient than chunked I/O.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries with averaged values
    """
    try:
        import h5py
        import multiprocessing as mp
        import numpy as np
        
        print(f"🚀 Using memory-efficient multi-core HDF5 patch tracking: {len(patch_point_pairs)} points")
        
        # Check if we can load the entire dataset
        if not can_load_entire_dataset(hdf5_file_path):
            print("⚠️  Dataset too large for memory-efficient approach, falling back to chunked method")
            return track_fixed_patch_region_hdf5_parallel_multicore(hdf5_file_path, patch_point_pairs)
        
        # Get optimal process count
        optimal_processes = get_optimal_process_count()
        
        # Load the entire dataset once
        print("📥 Loading entire dataset into memory...")
        start_load = time.time()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load all data at once
            data = f['data'][:]
            times = f['times'][:]
            
            # Get properties and indices
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
            
            # Pack property indices
            property_indices = (x_idx, y_idx, z_idx, face_idx, patch_idx, pressure_idx, velocity_idx, vdotn_idx)
        
        load_time = time.time() - start_load
        n_times = len(times)
        
        print(f"✅ Dataset loaded: {n_times:,} timesteps in {load_time:.1f} seconds")
        print(f"🚀 Starting multi-core processing with {optimal_processes} CPU cores...")
        
        # Split timesteps across processes
        timesteps_per_process = n_times // optimal_processes
        process_args = []
        
        for i in range(optimal_processes):
            start_idx = i * timesteps_per_process
            if i == optimal_processes - 1:
                end_idx = n_times  # Last process gets remaining timesteps
            else:
                end_idx = (i + 1) * timesteps_per_process
            
            process_args.append((data, times, start_idx, end_idx, patch_point_pairs, property_indices))
        
        # Process in parallel - no I/O, just computation
        trajectory_data = []
        
        start_process = time.time()
        
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(process_timestep_range_parallel, process_args),
                total=len(process_args),
                desc=f"Memory-efficient multi-core processing ({optimal_processes} cores)"
            ))
        
        # Combine results from all processes
        for process_results in results:
            trajectory_data.extend(process_results)
        
        # Sort by time to ensure chronological order
        trajectory_data.sort(key=lambda x: x['time'])
        
        process_time = time.time() - start_process
        total_time = load_time + process_time
        
        print(f"✅ Memory-efficient multi-core HDF5 patch tracking completed:")
        print(f"   Dataset load time: {load_time:.1f} seconds")
        print(f"   Processing time: {process_time:.1f} seconds")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Tracked: {len(trajectory_data):,} time points")
        print(f"   Speed: {len(trajectory_data)/total_time:.1f} points/second")
        print(f"   CPU cores used: {optimal_processes}")
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in memory-efficient multi-core HDF5 patch tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_point_timestep_range_parallel(args):
    """
    Process a range of timesteps for single point tracking in parallel.
    This works on already-loaded data in memory (no I/O operations).
    
    Args:
        args: Tuple of (data_chunk, times_chunk, start_idx, end_idx, patch_number, face_index, property_indices)
    
    Returns:
        List of point trajectory data for this timestep range
    """
    data_chunk, times_chunk, start_idx, end_idx, patch_number, face_index, property_indices = args
    
    try:
        import numpy as np
        
        trajectory_data = []
        
        # Unpack property indices
        x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
        pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
        
        # Process each timestep in this range
        for local_t_idx in range(start_idx, end_idx):
            time = times_chunk[local_t_idx]
            
            # Find the specific point
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
                    'patch35_avg_pressure': 0.0,
                    'signed_velocity': 0.0,
                }
                
                if pressure_idx is not None:
                    pressure_val = data_chunk[local_t_idx, point_idx, pressure_idx]
                    point_data['pressure'] = pressure_val
                    point_data['adjusted_pressure'] = pressure_val
                    
                if velocity_idx is not None:
                    point_data['velocity'] = data_chunk[local_t_idx, point_idx, velocity_idx]
                    
                if vdotn_idx is not None:
                    vdotn_val = data_chunk[local_t_idx, point_idx, vdotn_idx]
                    point_data['vdotn'] = vdotn_val
                    if 'velocity' in point_data:
                        point_data['signed_velocity'] = point_data['velocity'] * np.sign(vdotn_val)
                
                trajectory_data.append(point_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing point timestep range {start_idx}-{end_idx}: {e}")
        return []


def track_point_hdf5_memory_efficient_multicore(hdf5_file_path: str, patch_number: int, face_index: int) -> List[Dict]:
    """
    Memory-efficient multi-core HDF5 single point tracking.
    Loads entire dataset once (if memory allows) and splits work across processes.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_number: Patch number to track
        face_index: Face index to track
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import multiprocessing as mp
        import numpy as np
        
        print(f"🚀 Using memory-efficient multi-core HDF5 point tracking: Patch {patch_number}, Face {face_index}")
        
        # Check if we can load the entire dataset
        if not can_load_entire_dataset(hdf5_file_path):
            print("⚠️  Dataset too large for memory-efficient approach, falling back to chunked method")
            return track_point_hdf5_parallel_multicore(hdf5_file_path, patch_number, face_index)
        
        # Get optimal process count
        optimal_processes = get_optimal_process_count()
        
        # Load the entire dataset once
        print("📥 Loading entire dataset into memory...")
        start_load = time.time()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load all data at once
            data = f['data'][:]
            times = f['times'][:]
            
            # Get properties and indices
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
            
            # Pack property indices
            property_indices = (x_idx, y_idx, z_idx, face_idx, patch_idx, pressure_idx, velocity_idx, vdotn_idx)
        
        load_time = time.time() - start_load
        n_times = len(times)
        
        print(f"✅ Dataset loaded: {n_times:,} timesteps in {load_time:.1f} seconds")
        print(f"🚀 Starting multi-core processing with {optimal_processes} CPU cores...")
        
        # Split timesteps across processes
        timesteps_per_process = n_times // optimal_processes
        process_args = []
        
        for i in range(optimal_processes):
            start_idx = i * timesteps_per_process
            if i == optimal_processes - 1:
                end_idx = n_times  # Last process gets remaining timesteps
            else:
                end_idx = (i + 1) * timesteps_per_process
            
            process_args.append((data, times, start_idx, end_idx, patch_number, face_index, property_indices))
        
        # Process in parallel - no I/O, just computation
        trajectory_data = []
        
        start_process = time.time()
        
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(process_point_timestep_range_parallel, process_args),
                total=len(process_args),
                desc=f"Memory-efficient point tracking ({optimal_processes} cores)"
            ))
        
        # Combine results from all processes
        for process_results in results:
            trajectory_data.extend(process_results)
        
        # Sort by time to ensure chronological order
        trajectory_data.sort(key=lambda x: x['time'])
        
        process_time = time.time() - start_process
        total_time = load_time + process_time
        
        print(f"✅ Memory-efficient multi-core HDF5 point tracking completed:")
        print(f"   Dataset load time: {load_time:.1f} seconds")
        print(f"   Processing time: {process_time:.1f} seconds")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Tracked: {len(trajectory_data):,} time points")
        print(f"   Speed: {len(trajectory_data)/total_time:.1f} points/second")
        print(f"   CPU cores used: {optimal_processes}")
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in memory-efficient multi-core HDF5 point tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_hdf5_chunk_range_parallel(args):
    """
    Process a chunk range in parallel by loading only the needed data.
    Each process loads only its assigned chunk, avoiding memory duplication.
    
    Args:
        args: Tuple of (hdf5_file_path, chunk_start, chunk_end, patch_point_pairs, property_indices)
    
    Returns:
        List of trajectory data for this chunk range
    """
    hdf5_file_path, chunk_start, chunk_end, patch_point_pairs, property_indices = args
    
    try:
        import h5py
        import numpy as np
        
        trajectory_data = []
        
        # Each process opens the file and loads ONLY its assigned chunk
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load only the chunk this process needs (not the entire dataset)
            data_chunk = f['data'][chunk_start:chunk_end]  # Only this chunk!
            times_chunk = f['times'][chunk_start:chunk_end]
            
            # Unpack property indices
            x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
            pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
            
            # Process each timestep in this chunk
            for local_t_idx, time in enumerate(times_chunk):
                region_points = []
                
                # Find all points in the region for this timestep
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
                    # Optimized averaging using numpy
                    positions = np.array([[p['x'], p['y'], p['z']] for p in region_points])
                    avg_pos = np.mean(positions, axis=0)
                    
                    avg_data = {
                        'time_point': int(time * 1000),
                        'time': time,
                        'x': avg_pos[0],
                        'y': avg_pos[1],
                        'z': avg_pos[2],
                        'num_points_in_region': len(region_points),
                        'patch35_avg_pressure': 0.0,
                        'signed_velocity': 0.0,
                    }
                    
                    if pressure_idx is not None and all('pressure' in p for p in region_points):
                        pressures = np.array([p['pressure'] for p in region_points])
                        avg_data['pressure'] = np.mean(pressures)
                        avg_data['adjusted_pressure'] = avg_data['pressure']
                    
                    if velocity_idx is not None and all('velocity' in p for p in region_points):
                        velocities = np.array([p['velocity'] for p in region_points])
                        avg_data['velocity'] = np.mean(velocities)
                        
                    if vdotn_idx is not None and all('vdotn' in p for p in region_points):
                        vdotns = np.array([p['vdotn'] for p in region_points])
                        avg_vdotn = np.mean(vdotns)
                        avg_data['vdotn'] = avg_vdotn
                        if 'velocity' in avg_data:
                            avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                    
                    trajectory_data.append(avg_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing chunk range {chunk_start}-{chunk_end}: {e}")
        return []


def track_fixed_patch_region_hdf5_memory_safe_multicore(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Memory-safe multi-core HDF5 patch tracking.
    Uses all CPU cores while guaranteeing memory safety by chunking data appropriately.
    Each process loads only its assigned chunk, avoiding memory duplication.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries with averaged values
    """
    try:
        import h5py
        import multiprocessing as mp
        import numpy as np
        
        print(f"🚀 Using memory-safe multi-core HDF5 patch tracking: {len(patch_point_pairs)} points")
        
        # Get optimal process count and chunk size
        optimal_processes = get_optimal_process_count()
        
        # Calculate memory-safe chunk size per process
        # Each process will load its own chunk, so we can use more memory per chunk
        memory_info = get_system_memory_info()
        memory_per_process = memory_info['available_gb'] / optimal_processes * 0.7  # 70% of available per process
        
        chunk_size = calculate_safe_chunk_size(hdf5_file_path, target_memory_gb=memory_per_process)
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset info
            n_times = f['data'].shape[0]
            
            # Get properties and indices
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
            
            # Pack property indices
            property_indices = (x_idx, y_idx, z_idx, face_idx, patch_idx, pressure_idx, velocity_idx, vdotn_idx)
        
        # Create chunk arguments - each process gets its own chunk range
        chunk_args = []
        for chunk_start in range(0, n_times, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_times)
            chunk_args.append((hdf5_file_path, chunk_start, chunk_end, patch_point_pairs, property_indices))
        
        n_chunks = len(chunk_args)
        
        print(f"📊 Memory-safe processing plan:")
        print(f"   Dataset: {n_times:,} timesteps")
        print(f"   Chunk size: {chunk_size:,} timesteps")
        print(f"   Number of chunks: {n_chunks}")
        print(f"   CPU cores: {optimal_processes}")
        print(f"   Memory per process: {memory_per_process:.1f}GB")
        print(f"   Total memory usage: ~{memory_per_process * optimal_processes:.1f}GB")
        
        # Process chunks in parallel - each process loads only its chunk
        trajectory_data = []
        
        start_time = time.time()
        
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(process_hdf5_chunk_range_parallel, chunk_args),
                total=len(chunk_args),
                desc=f"Memory-safe multi-core processing ({optimal_processes} cores)"
            ))
        
        # Combine results from all chunks
        for chunk_results in results:
            trajectory_data.extend(chunk_results)
        
        # Sort by time to ensure chronological order
        trajectory_data.sort(key=lambda x: x['time'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Memory-safe multi-core HDF5 patch tracking completed:")
        print(f"   Tracked: {len(trajectory_data):,} time points")
        print(f"   Time: {processing_time:.1f} seconds")
        print(f"   Speed: {len(trajectory_data)/processing_time:.1f} points/second")
        print(f"   CPU cores used: {optimal_processes}")
        print(f"   Memory safety: ✅ Guaranteed (no data duplication)")
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in memory-safe multi-core HDF5 patch tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_hdf5_point_chunk_range_parallel(args):
    """
    Process a chunk range for single point tracking in parallel.
    Each process loads only its assigned chunk, avoiding memory duplication.
    
    Args:
        args: Tuple of (hdf5_file_path, chunk_start, chunk_end, patch_number, face_index, property_indices)
    
    Returns:
        List of point trajectory data for this chunk range
    """
    hdf5_file_path, chunk_start, chunk_end, patch_number, face_index, property_indices = args
    
    try:
        import h5py
        import numpy as np
        
        trajectory_data = []
        
        # Each process opens the file and loads ONLY its assigned chunk
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load only the chunk this process needs
            data_chunk = f['data'][chunk_start:chunk_end]
            times_chunk = f['times'][chunk_start:chunk_end]
            
            # Unpack property indices
            x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
            pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
            
            # Process each timestep in this chunk
            for local_t_idx, time in enumerate(times_chunk):
                # Find the specific point
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
                        'patch35_avg_pressure': 0.0,
                        'signed_velocity': 0.0,
                    }
                    
                    if pressure_idx is not None:
                        pressure_val = data_chunk[local_t_idx, point_idx, pressure_idx]
                        point_data['pressure'] = pressure_val
                        point_data['adjusted_pressure'] = pressure_val
                        
                    if velocity_idx is not None:
                        point_data['velocity'] = data_chunk[local_t_idx, point_idx, velocity_idx]
                        
                    if vdotn_idx is not None:
                        vdotn_val = data_chunk[local_t_idx, point_idx, vdotn_idx]
                        point_data['vdotn'] = vdotn_val
                        if 'velocity' in point_data:
                            point_data['signed_velocity'] = point_data['velocity'] * np.sign(vdotn_val)
                    
                    trajectory_data.append(point_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing point chunk range {chunk_start}-{chunk_end}: {e}")
        return []


def track_point_hdf5_memory_safe_multicore(hdf5_file_path: str, patch_number: int, face_index: int) -> List[Dict]:
    """
    Memory-safe multi-core HDF5 single point tracking.
    Uses all CPU cores while guaranteeing memory safety.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_number: Patch number to track
        face_index: Face index to track
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import multiprocessing as mp
        import numpy as np
        
        print(f"🚀 Using memory-safe multi-core HDF5 point tracking: Patch {patch_number}, Face {face_index}")
        
        # Get optimal process count and chunk size
        optimal_processes = get_optimal_process_count()
        
        # Calculate memory-safe chunk size per process
        memory_info = get_system_memory_info()
        memory_per_process = memory_info['available_gb'] / optimal_processes * 0.7
        
        chunk_size = calculate_safe_chunk_size(hdf5_file_path, target_memory_gb=memory_per_process)
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset info
            n_times = f['data'].shape[0]
            
            # Get properties and indices
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
            
            # Pack property indices
            property_indices = (x_idx, y_idx, z_idx, face_idx, patch_idx, pressure_idx, velocity_idx, vdotn_idx)
        
        # Create chunk arguments
        chunk_args = []
        for chunk_start in range(0, n_times, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_times)
            chunk_args.append((hdf5_file_path, chunk_start, chunk_end, patch_number, face_index, property_indices))
        
        n_chunks = len(chunk_args)
        
        print(f"📊 Memory-safe processing plan:")
        print(f"   Dataset: {n_times:,} timesteps")
        print(f"   Chunk size: {chunk_size:,} timesteps")
        print(f"   Number of chunks: {n_chunks}")
        print(f"   CPU cores: {optimal_processes}")
        print(f"   Memory per process: {memory_per_process:.1f}GB")
        
        # Process chunks in parallel
        trajectory_data = []
        
        start_time = time.time()
        
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(process_hdf5_point_chunk_range_parallel, chunk_args),
                total=len(chunk_args),
                desc=f"Memory-safe point tracking ({optimal_processes} cores)"
            ))
        
        # Combine results from all chunks
        for chunk_results in results:
            trajectory_data.extend(chunk_results)
        
        # Sort by time to ensure chronological order
        trajectory_data.sort(key=lambda x: x['time'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Memory-safe multi-core HDF5 point tracking completed:")
        print(f"   Tracked: {len(trajectory_data):,} time points")
        print(f"   Time: {processing_time:.1f} seconds")
        print(f"   Speed: {len(trajectory_data)/processing_time:.1f} points/second")
        print(f"   CPU cores used: {optimal_processes}")
        print(f"   Memory safety: ✅ Guaranteed")
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in memory-safe multi-core HDF5 point tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def track_fixed_patch_region_hdf5_optimized_index_lookup(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    OPTIMIZED: Find patch point indices once, then use direct array indexing for all timesteps.
    This is dramatically faster for large datasets (2000+ timesteps).
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries with averaged values
    """
    try:
        import h5py
        import numpy as np
        
        print(f"🚀 Using OPTIMIZED index-lookup HDF5 patch tracking: {len(patch_point_pairs)} points")
        
        trajectory_data = []
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset references
            data_dataset = f['data']
            times_dataset = f['times']
            
            # Get properties and indices
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
            
            n_times = data_dataset.shape[0]
            
            # OPTIMIZATION: Find patch point indices ONCE at first timestep
            print("🔍 Finding patch point indices at first timestep...")
            first_timestep_data = data_dataset[0]  # Load only first timestep
            
            patch_point_indices = []
            for patch_num, face_idx_val in patch_point_pairs:
                matches = np.where(
                    (first_timestep_data[:, patch_idx] == patch_num) & 
                    (first_timestep_data[:, face_idx] == face_idx_val)
                )[0]
                
                if len(matches) > 0:
                    patch_point_indices.append(matches[0])  # Use first match
                else:
                    print(f"⚠️  Warning: Point P{patch_num}F{face_idx_val} not found in first timestep")
            
            print(f"✅ Found {len(patch_point_indices)} patch point indices")
            
            if len(patch_point_indices) == 0:
                print("❌ No patch points found!")
                return []
            
            # Convert to numpy array for efficient indexing
            patch_indices = np.array(patch_point_indices)
            
            # Process all timesteps using direct indexing
            print(f"⚡ Processing {n_times:,} timesteps with direct indexing...")
            
            # Calculate chunk size for memory efficiency
            chunk_size = calculate_safe_chunk_size(hdf5_file_path, target_memory_gb=4.0)
            n_chunks = (n_times + chunk_size - 1) // chunk_size
            
            for chunk_start in range(0, n_times, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_times)
                
                # Load chunk
                data_chunk = data_dataset[chunk_start:chunk_end]
                times_chunk = times_dataset[chunk_start:chunk_end]
                
                # Progress update
                if chunk_start % (chunk_size * 5) == 0 or chunk_end == n_times:
                    progress_pct = (chunk_end / n_times) * 100
                    print(f"  Progress: {progress_pct:.1f}% ({chunk_end:,}/{n_times:,} timesteps)")
                
                # Process each timestep in this chunk
                for local_t_idx, time in enumerate(times_chunk):
                    # OPTIMIZED: Direct array indexing - no searching!
                    patch_data = data_chunk[local_t_idx, patch_indices, :]
                    
                    # Extract coordinates
                    positions = patch_data[:, [x_idx, y_idx, z_idx]]
                    avg_pos = np.mean(positions, axis=0)
                    
                    # Create trajectory point
                    avg_data = {
                        'time_point': int(time * 1000),
                        'time': time,
                        'x': avg_pos[0],
                        'y': avg_pos[1],
                        'z': avg_pos[2],
                        'num_points_in_region': len(patch_indices),
                        'patch35_avg_pressure': 0.0,
                        'signed_velocity': 0.0,
                    }
                    
                    # Add pressure if available
                    if pressure_idx is not None:
                        pressures = patch_data[:, pressure_idx]
                        avg_data['pressure'] = np.mean(pressures)
                        avg_data['adjusted_pressure'] = avg_data['pressure']
                    
                    # Add velocity if available
                    if velocity_idx is not None:
                        velocities = patch_data[:, velocity_idx]
                        avg_data['velocity'] = np.mean(velocities)
                        
                    # Add VdotN if available
                    if vdotn_idx is not None:
                        vdotns = patch_data[:, vdotn_idx]
                        avg_vdotn = np.mean(vdotns)
                        avg_data['vdotn'] = avg_vdotn
                        if 'velocity' in avg_data:
                            avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                    
                    trajectory_data.append(avg_data)
        
        print(f"✅ Optimized index-lookup HDF5 patch tracking completed: {len(trajectory_data):,} time points")
        return trajectory_data
        
    except Exception as e:
        print(f"Error in optimized index-lookup HDF5 patch tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def track_fixed_patch_region_hdf5_optimized_multicore(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    OPTIMIZED + MULTICORE: Find patch indices once, then use multi-core processing with direct indexing.
    Best performance for large datasets (2000+ timesteps).
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries with averaged values
    """
    try:
        import h5py
        import multiprocessing as mp
        import numpy as np
        
        print(f"🚀 Using OPTIMIZED multi-core HDF5 patch tracking: {len(patch_point_pairs)} points")
        
        # Get optimal process count
        optimal_processes = get_optimal_process_count()
        
        # Calculate memory-safe chunk size per process
        memory_info = get_system_memory_info()
        memory_per_process = memory_info['available_gb'] / optimal_processes * 0.7
        chunk_size = calculate_safe_chunk_size(hdf5_file_path, target_memory_gb=memory_per_process)
        
        # Find patch point indices once at the beginning
        print("🔍 Finding patch point indices at first timestep...")
        patch_indices = None
        
        with h5py.File(hdf5_file_path, 'r') as f:
            # Get dataset info
            n_times = f['data'].shape[0]
            
            # Get properties and indices
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
            
            # Pack property indices
            property_indices = (x_idx, y_idx, z_idx, face_idx, patch_idx, pressure_idx, velocity_idx, vdotn_idx)
            
            # Find patch point indices at first timestep
            first_timestep_data = f['data'][0]
            
            patch_point_indices = []
            for patch_num, face_idx_val in patch_point_pairs:
                matches = np.where(
                    (first_timestep_data[:, patch_idx] == patch_num) & 
                    (first_timestep_data[:, face_idx] == face_idx_val)
                )[0]
                
                if len(matches) > 0:
                    patch_point_indices.append(matches[0])
                else:
                    print(f"⚠️  Warning: Point P{patch_num}F{face_idx_val} not found in first timestep")
            
            patch_indices = np.array(patch_point_indices)
        
        print(f"✅ Found {len(patch_indices)} patch point indices")
        
        if len(patch_indices) == 0:
            print("❌ No patch points found!")
            return []
        
        # Create chunk arguments with optimized indices
        chunk_args = []
        for chunk_start in range(0, n_times, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_times)
            chunk_args.append((hdf5_file_path, chunk_start, chunk_end, patch_indices, property_indices))
        
        n_chunks = len(chunk_args)
        
        print(f"📊 Optimized multi-core processing plan:")
        print(f"   Dataset: {n_times:,} timesteps")
        print(f"   Chunk size: {chunk_size:,} timesteps")
        print(f"   Number of chunks: {n_chunks}")
        print(f"   CPU cores: {optimal_processes}")
        print(f"   Memory per process: {memory_per_process:.1f}GB")
        print(f"   Optimization: Direct array indexing (no searching)")
        
        # Process chunks in parallel with optimized indexing
        trajectory_data = []
        
        start_time = time.time()
        
        with mp.Pool(processes=optimal_processes) as pool:
            results = list(tqdm(
                pool.imap(process_hdf5_chunk_optimized_parallel, chunk_args),
                total=len(chunk_args),
                desc=f"Optimized multi-core processing ({optimal_processes} cores)"
            ))
        
        # Combine results from all chunks
        for chunk_results in results:
            trajectory_data.extend(chunk_results)
        
        # Sort by time to ensure chronological order
        trajectory_data.sort(key=lambda x: x['time'])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Optimized multi-core HDF5 patch tracking completed:")
        print(f"   Tracked: {len(trajectory_data):,} time points")
        print(f"   Time: {processing_time:.1f} seconds")
        print(f"   Speed: {len(trajectory_data)/processing_time:.1f} points/second")
        print(f"   CPU cores used: {optimal_processes}")
        print(f"   Optimization: ✅ Direct indexing (no point searching)")
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error in optimized multi-core HDF5 patch tracking: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_hdf5_chunk_optimized_parallel(args):
    """
    Process a chunk using optimized direct array indexing.
    No point searching - uses pre-computed indices.
    
    Args:
        args: Tuple of (hdf5_file_path, chunk_start, chunk_end, patch_indices, property_indices)
    
    Returns:
        List of trajectory data for this chunk
    """
    hdf5_file_path, chunk_start, chunk_end, patch_indices, property_indices = args
    
    try:
        import h5py
        import numpy as np
        
        trajectory_data = []
        
        # Each process opens the file and loads only its assigned chunk
        with h5py.File(hdf5_file_path, 'r') as f:
            # Load only the chunk this process needs
            data_chunk = f['data'][chunk_start:chunk_end]
            times_chunk = f['times'][chunk_start:chunk_end]
            
            # Unpack property indices
            x_idx, y_idx, z_idx, face_idx, patch_idx = property_indices[:5]
            pressure_idx, velocity_idx, vdotn_idx = property_indices[5:]
            
            # Process each timestep in this chunk
            for local_t_idx, time in enumerate(times_chunk):
                # OPTIMIZED: Direct array indexing - no searching!
                patch_data = data_chunk[local_t_idx, patch_indices, :]
                
                # Extract coordinates and calculate average
                positions = patch_data[:, [x_idx, y_idx, z_idx]]
                avg_pos = np.mean(positions, axis=0)
                
                # Create trajectory point
                avg_data = {
                    'time_point': int(time * 1000),
                    'time': time,
                    'x': avg_pos[0],
                    'y': avg_pos[1],
                    'z': avg_pos[2],
                    'num_points_in_region': len(patch_indices),
                    'patch35_avg_pressure': 0.0,
                    'signed_velocity': 0.0,
                }
                
                # Add pressure if available
                if pressure_idx is not None:
                    pressures = patch_data[:, pressure_idx]
                    avg_data['pressure'] = np.mean(pressures)
                    avg_data['adjusted_pressure'] = avg_data['pressure']
                
                # Add velocity if available
                if velocity_idx is not None:
                    velocities = patch_data[:, velocity_idx]
                    avg_data['velocity'] = np.mean(velocities)
                    
                # Add VdotN if available
                if vdotn_idx is not None:
                    vdotns = patch_data[:, vdotn_idx]
                    avg_vdotn = np.mean(vdotns)
                    avg_data['vdotn'] = avg_vdotn
                    if 'velocity' in avg_data:
                        avg_data['signed_velocity'] = avg_data['velocity'] * np.sign(avg_vdotn)
                
                trajectory_data.append(avg_data)
        
        return trajectory_data
        
    except Exception as e:
        print(f"Error processing optimized chunk {chunk_start}-{chunk_end}: {e}")
        return []


def auto_select_optimized_hdf5_tracking_method(hdf5_file_path: str, patch_point_pairs: List[Tuple[int, int]]) -> List[Dict]:
    """
    Automatically select the best optimized HDF5 tracking method.
    Always uses the optimized approach with direct array indexing.
    
    Args:
        hdf5_file_path: Path to HDF5 cache file
        patch_point_pairs: List of (patch_number, face_index) pairs
    
    Returns:
        List of tracking data dictionaries
    """
    try:
        import h5py
        import os
        
        # Get file size and system info
        file_size_gb = os.path.getsize(hdf5_file_path) / (1024**3)
        memory_info = get_system_memory_info()
        cpu_cores = mp.cpu_count()
        
        with h5py.File(hdf5_file_path, 'r') as f:
            n_times = f['data'].shape[0]
        
        print(f"📊 Dataset: {file_size_gb:.1f}GB, {n_times:,} timesteps")
        print(f"🖥️  System: {cpu_cores} cores, {memory_info['available_gb']:.1f}GB RAM available")
        
        # Use optimized multi-core approach when beneficial
        if cpu_cores >= 4 and memory_info['available_gb'] > 8:
            print(f"🚀 Using OPTIMIZED multi-core processing ({cpu_cores} cores)")
            return track_fixed_patch_region_hdf5_optimized_multicore(hdf5_file_path, patch_point_pairs)
        else:
            print(f"⚡ Using OPTIMIZED single-thread processing")
            return track_fixed_patch_region_hdf5_optimized_index_lookup(hdf5_file_path, patch_point_pairs)
            
    except Exception as e:
        print(f"Warning: Could not determine optimal method, using standard: {e}")
        return track_fixed_patch_region_hdf5_parallel(hdf5_file_path, patch_point_pairs)


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