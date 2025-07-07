"""
Trajectory data processing module for CFD analysis.
Handles point tracking and calculation of derived quantities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import h5py
import gc
import psutil

def load_tables_to_3d_array(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False) -> Dict:
    """
    Load all tables into a 3D array format and save as HDF5.
    If HDF5 file already exists, load from it instead of processing tables again.
    Format: (time_steps, points, properties)
    
    Args:
        xyz_files: List of paths to XYZ table files with patch numbers
        save_path: Path to save the HDF5 file
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    # Check if HDF5 file already exists (unless forced to overwrite)
    if Path(save_path).exists() and not overwrite_existing:
        print(f"\nFound existing data file: {save_path}")
        # Verify the file has the expected structure
        try:
            with h5py.File(save_path, 'r') as f:
                if all(key in f for key in ['data', 'times']) and 'properties' in f.attrs:
                    print("File structure verified. Loading existing data...")
                    return {'file_path': save_path, 'properties': f.attrs['properties']}
                else:
                    print("Existing file has incorrect structure. Will recreate...")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Will recreate the data file...")
    elif Path(save_path).exists() and overwrite_existing:
        print(f"\n🔄 Force rerun enabled - recreating HDF5 file: {save_path}")
    
    print("\nLoading all tables into 3D array format...")
    
    # First read one table to get dimensions and structure
    first_df = pd.read_csv(xyz_files[0], low_memory=False)
    n_points = len(first_df)
    n_timesteps = len(xyz_files)
    
    # Use all columns from the raw table
    properties = first_df.columns.tolist()
    n_properties = len(properties)
    
    print(f"Data dimensions:")
    print(f"Time steps: {n_timesteps}")
    print(f"Points per step: {n_points}")
    print(f"Properties tracked: {n_properties}")
    print("\nProperties being tracked:")
    for prop in properties:
        print(f"  - {prop}")
    
    # Create HDF5 file
    with h5py.File(save_path, 'w') as f:
        # Create datasets with appropriate data types
        data = f.create_dataset('data', (n_timesteps, n_points, n_properties), dtype='float64')
        times = f.create_dataset('times', (n_timesteps,), dtype='float64')
        
        # Store property names exactly as they appear in the raw table
        f.attrs['properties'] = [p.encode('utf-8') for p in properties]
        
        # Load all tables
        for i, xyz_file in enumerate(tqdm(xyz_files, desc="Loading tables")):
            df = pd.read_csv(xyz_file, low_memory=False)
            # Extract timestep using robust method (handles scientific notation, decimals, etc.)
            from utils.file_processing import extract_timestep_from_filename
            table_time = extract_timestep_from_filename(xyz_file) * 0.001  # Convert to seconds
            times[i] = table_time
            
            # Store data for each property in the same order as the raw table
            for j, prop in enumerate(properties):
                try:
                    # Convert to numeric, handling any non-numeric values
                    values = pd.to_numeric(df[prop], errors='coerce').fillna(0.0)
                    data[i, :, j] = values.values
                except Exception as e:
                    print(f"Warning: Error converting column '{prop}' to numeric: {e}")
                    # Fill with zeros if conversion fails
                    data[i, :, j] = np.zeros(n_points)
    
    print(f"\nSaved 3D data to {save_path}")
    print("Properties preserved in the same order as raw table")
    return {'file_path': save_path, 'properties': properties}

def filter_data_by_breathing_cycle(data_info: Dict, subject_name: str) -> Dict:
    """
    Filter the 3D data to include only points within clean breathing cycles.
    Memory-safe version that uses chunked loading.
    
    Args:
        data_info: Dictionary containing the HDF5 file path and metadata
        subject_name: Name of the subject (e.g., 'OSAMRI007')
        
    Returns:
        dict: Updated data_info dictionary with filtered data path
    """
    # Use the new memory-safe chunked version
    from utils.parallel_csv_processing import filter_data_by_breathing_cycle_chunked
    return filter_data_by_breathing_cycle_chunked(data_info, subject_name)

def track_point_movement_3d(data_info: Dict, patch_number: int, face_index: int) -> Dict:
    """
    Track the movement of a specific point through time.
    Uses auto-selection to pick the best method (multi-core parallel for large datasets).
    
    Args:
        data_info: Dictionary containing the HDF5 file path and metadata
        patch_number: The patch number to track
        face_index: The face index within the patch to track
        
    Returns:
        dict: Dictionary containing trajectory information
    """
    # Use the new auto-selection functions that pick the best method
    from utils.parallel_csv_processing import auto_select_hdf5_point_tracking_method
    
    # Get the HDF5 file path
    hdf5_file_path = data_info['file_path']
    
    # Use auto-selection to get the best tracking method
    trajectory_data = auto_select_hdf5_point_tracking_method(hdf5_file_path, patch_number, face_index)
    
    # Convert to the expected format
    if trajectory_data:
        times = [d['time'] for d in trajectory_data]
        positions = [(d['x'], d['y'], d['z']) for d in trajectory_data]
        pressures = [d.get('pressure', 0.0) for d in trajectory_data]
        velocities = [d.get('velocity', 0.0) for d in trajectory_data]
        vdotn_values = [d.get('vdotn', 0.0) for d in trajectory_data]
        
        return {
            'times': np.array(times),
            'positions': np.array(positions),
            'total_pressures': np.array(pressures),
            'velocities': np.array(velocities),
            'vdotn': np.array(vdotn_values),
            'table_numbers': [int(t * 1000) for t in times],
            'accelerations': np.array([]),  # Will be calculated later
            'dp_dt': np.array([])  # Will be calculated later
        }
    else:
        return {}

def calculate_derived_quantities(trajectory: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate derived quantities from trajectory data.
    
    Args:
        trajectory: Dictionary containing trajectory information
        
    Returns:
        Tuple of (velocity, vdotn, acceleration, dp_dt) arrays
    """
    if not trajectory:
        print("Warning: Empty trajectory, cannot calculate derived quantities")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Return pre-calculated quantities from the trajectory
    return (
        trajectory.get('velocities', np.array([])),
        trajectory.get('vdotn', np.array([])),
        trajectory.get('accelerations', np.array([])),
        trajectory.get('dp_dt', np.array([]))
    )

def save_trajectory_data(trajectory: Dict, velocity: np.ndarray, vdotn: np.ndarray, 
                        acceleration: np.ndarray, dp_dt: np.ndarray, 
                        patch_number: int, face_index: int,
                        output_file: str) -> None:
    """
    Save trajectory data to CSV file.
    
    Args:
        trajectory: Dictionary containing trajectory data
        velocity: Array of velocity magnitudes
        vdotn: Array of velocity dot normal values
        acceleration: Array of acceleration values
        dp_dt: Array of pressure rate of change values
        patch_number: Patch number identifier
        face_index: Face index identifier
        output_file: Path to save the CSV file
    """
    df = pd.DataFrame({
        'Time (s)': trajectory['times'],
        'Table Number': trajectory['table_numbers'],
        'Total Pressure (Pa)': trajectory['total_pressures'],
        'Velocity Magnitude (m/s)': velocity,
        'VdotN': vdotn,
        'Acceleration (m/s²)': acceleration,
        'dP/dt (Pa/s)': dp_dt
    })
    df.to_csv(output_file, index=False)

def load_tables_to_3d_array_parallel(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False) -> Dict:
    """
    PARALLEL VERSION: Load all tables into a 3D array format and save as HDF5.
    Uses multiprocessing to read CSV files in parallel for dramatic speedup.
    
    Args:
        xyz_files: List of paths to XYZ table files with patch numbers
        save_path: Path to save the HDF5 file
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    
    # Check if HDF5 file already exists
    if Path(save_path).exists():
        print(f"\nFound existing data file: {save_path}")
        # Verify the file has the expected structure
        try:
            with h5py.File(save_path, 'r') as f:
                if all(key in f for key in ['data', 'times']) and 'properties' in f.attrs:
                    print("File structure verified. Loading existing data...")
                    return {'file_path': save_path, 'properties': f.attrs['properties']}
                else:
                    print("Existing file has incorrect structure. Will recreate...")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Will recreate the data file...")
    
    print("\n🚀 Loading all tables into 3D array format (PARALLEL)...")
    
    # IMPORTANT: xyz_files is already sorted in natural chronological order by main.py
    # The parallel processing preserves this order by using the original file index
    print("📋 Files are pre-sorted in natural chronological order for time-series consistency")
    
    # First read one table to get dimensions and structure
    first_df = pd.read_csv(xyz_files[0], low_memory=False)
    n_points = len(first_df)
    n_timesteps = len(xyz_files)
    
    # Use all columns from the raw table
    properties = first_df.columns.tolist()
    n_properties = len(properties)
    
    print(f"Data dimensions:")
    print(f"Time steps: {n_timesteps}")
    print(f"Points per step: {n_points}")
    print(f"Properties tracked: {n_properties}")
    print(f"🔥 Using parallel processing for {n_timesteps} CSV files...")
    
    # Get optimal process count
    optimal_processes = min(mp.cpu_count(), n_timesteps, 16)  # Cap at 16 processes
    print(f"📊 Using {optimal_processes} parallel processes")
    
    # Create HDF5 file structure
    with h5py.File(save_path, 'w') as f:
        # Create datasets with appropriate data types
        data = f.create_dataset('data', (n_timesteps, n_points, n_properties), dtype='float64')
        times = f.create_dataset('times', (n_timesteps,), dtype='float64')
        
        # Store property names exactly as they appear in the raw table
        f.attrs['properties'] = [p.encode('utf-8') for p in properties]
        
        # Process CSV files in parallel
        start_time = time.time()
        
        # Prepare arguments for parallel processing
        file_args = [(i, xyz_file, properties) for i, xyz_file in enumerate(xyz_files)]
        
        # Use parallel processing with proper progress tracking
        completed_files = 0
        
        with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_csv_file, args): args[0] for args in file_args}
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    table_data, table_time = future.result()
                    
                    # Write data to HDF5
                    data[index] = table_data
                    times[index] = table_time
                    
                    completed_files += 1
                    if completed_files % 50 == 0 or completed_files == n_timesteps:
                        progress = (completed_files / n_timesteps) * 100
                        elapsed = time.time() - start_time
                        print(f"  Progress: {progress:.1f}% ({completed_files}/{n_timesteps} files, {elapsed:.1f}s)")
                        
                except Exception as e:
                    print(f"Error processing file {index}: {e}")
                    # Fill with zeros if processing fails
                    data[index] = np.zeros((n_points, n_properties))
                    times[index] = 0.0
    
    total_time = time.time() - start_time
    print(f"\n✅ PARALLEL conversion completed in {total_time:.1f} seconds")
    print(f"📈 Performance: {n_timesteps/total_time:.1f} files/second")
    print(f"💾 Saved 3D data to {save_path}")
    
    return {'file_path': save_path, 'properties': properties}

def process_single_csv_file(args):
    """
    Process a single CSV file for parallel loading.
    
    Args:
        args: Tuple of (index, file_path, properties)
        
    Returns:
        Tuple of (processed_data, time_value)
    """
    index, xyz_file, properties = args
    
    try:
        # Read CSV file
        df = pd.read_csv(xyz_file, low_memory=False)
        
        # Extract timestep using robust method (handles scientific notation, decimals, etc.)
        from utils.file_processing import extract_timestep_from_filename
        table_time = extract_timestep_from_filename(xyz_file) * 0.001  # Convert to seconds
        
        # Process data for each property
        n_points = len(df)
        n_properties = len(properties)
        table_data = np.zeros((n_points, n_properties))
        
        for j, prop in enumerate(properties):
            try:
                # Convert to numeric, handling any non-numeric values
                values = pd.to_numeric(df[prop], errors='coerce').fillna(0.0)
                table_data[:, j] = values.values
            except Exception as e:
                print(f"Warning: Error converting column '{prop}' in file {index}: {e}")
                # Fill with zeros if conversion fails
                table_data[:, j] = np.zeros(n_points)
        
        return table_data, table_time
        
    except Exception as e:
        print(f"Error processing file {xyz_file}: {e}")
        # Return zeros if file processing fails
        return np.zeros((n_points, n_properties)), 0.0

def load_tables_to_3d_array_memory_safe_parallel(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False) -> Dict:
    """
    MEMORY-SAFE PARALLEL VERSION: For very large datasets that might not fit in memory.
    Processes CSV files in parallel chunks and writes to HDF5 incrementally.
    
    Args:
        xyz_files: List of paths to XYZ table files with patch numbers
        save_path: Path to save the HDF5 file
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time
    import psutil
    
    # Check if HDF5 file already exists (unless forced to overwrite)
    if Path(save_path).exists() and not overwrite_existing:
        print(f"\nFound existing data file: {save_path}")
        # Verify the file has the expected structure
        try:
            with h5py.File(save_path, 'r') as f:
                if all(key in f for key in ['data', 'times']) and 'properties' in f.attrs:
                    print("File structure verified. Loading existing data...")
                    return {'file_path': save_path, 'properties': f.attrs['properties']}
                else:
                    print("Existing file has incorrect structure. Will recreate...")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Will recreate the data file...")
    elif Path(save_path).exists() and overwrite_existing:
        print(f"\n🔄 Force rerun enabled - recreating HDF5 file: {save_path}")
    
    print("\n🚀 Loading all tables into 3D array format (MEMORY-SAFE PARALLEL)...")
    
    # IMPORTANT: xyz_files is already sorted in natural chronological order by main.py
    # The parallel processing preserves this order by using the original file index
    print("📋 Files are pre-sorted in natural chronological order for time-series consistency")
    
    # First read one table to get dimensions and structure
    first_df = pd.read_csv(xyz_files[0], low_memory=False)
    n_points = len(first_df)
    n_timesteps = len(xyz_files)
    
    # Use all columns from the raw table
    properties = first_df.columns.tolist()
    n_properties = len(properties)
    
    # Calculate memory usage per timestep
    memory_per_timestep_mb = (n_points * n_properties * 8) / (1024 * 1024)  # 8 bytes per float64
    
    print(f"Data dimensions:")
    print(f"Time steps: {n_timesteps}")
    print(f"Points per step: {n_points}")
    print(f"Properties tracked: {n_properties}")
    print(f"Memory per timestep: {memory_per_timestep_mb:.1f} MB")
    
    # Get system memory info
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    
    # Calculate optimal batch size (use 25% of available memory)
    target_memory_gb = available_gb * 0.25
    batch_size = max(1, int((target_memory_gb * 1024) / memory_per_timestep_mb))
    batch_size = min(batch_size, n_timesteps)  # Don't exceed total files
    
    # Get optimal process count
    optimal_processes = min(mp.cpu_count(), batch_size, 16)  # Cap at 16 processes
    
    print(f"📊 Memory-safe processing plan:")
    print(f"   Available RAM: {available_gb:.1f} GB")
    print(f"   Batch size: {batch_size} files")
    print(f"   Parallel processes: {optimal_processes}")
    print(f"   Number of batches: {(n_timesteps + batch_size - 1) // batch_size}")
    
    # Create HDF5 file structure
    with h5py.File(save_path, 'w') as f:
        # Create datasets with appropriate data types
        data = f.create_dataset('data', (n_timesteps, n_points, n_properties), dtype='float64')
        times = f.create_dataset('times', (n_timesteps,), dtype='float64')
        
        # Store property names exactly as they appear in the raw table
        f.attrs['properties'] = [p.encode('utf-8') for p in properties]
        
        # Process CSV files in parallel batches
        start_time = time.time()
        total_processed = 0
        
        for batch_start in range(0, n_timesteps, batch_size):
            batch_end = min(batch_start + batch_size, n_timesteps)
            batch_files = xyz_files[batch_start:batch_end]
            
            print(f"\n🔄 Processing batch {batch_start//batch_size + 1}: files {batch_start}-{batch_end-1}")
            
            # Prepare arguments for this batch
            file_args = [(i, xyz_file, properties) for i, xyz_file in enumerate(batch_files)]
            
            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
                # Submit all tasks in this batch
                future_to_index = {executor.submit(process_single_csv_file, args): args[0] for args in file_args}
                
                # Process completed tasks
                for future in as_completed(future_to_index):
                    local_index = future_to_index[future]
                    global_index = batch_start + local_index
                    
                    try:
                        table_data, table_time = future.result()
                        
                        # Write data to HDF5
                        data[global_index] = table_data
                        times[global_index] = table_time
                        
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing file {global_index}: {e}")
                        # Fill with zeros if processing fails
                        data[global_index] = np.zeros((n_points, n_properties))
                        times[global_index] = 0.0
                        total_processed += 1
            
            # Progress update
            progress = (total_processed / n_timesteps) * 100
            elapsed = time.time() - start_time
            print(f"  ✅ Batch completed: {progress:.1f}% total ({total_processed}/{n_timesteps} files, {elapsed:.1f}s)")
            
            # Force garbage collection between batches
            gc.collect()
    
    total_time = time.time() - start_time
    print(f"\n✅ MEMORY-SAFE PARALLEL conversion completed in {total_time:.1f} seconds")
    print(f"📈 Performance: {n_timesteps/total_time:.1f} files/second")
    print(f"💾 Saved 3D data to {save_path}")
    
    return {'file_path': save_path, 'properties': properties}

def auto_select_csv_to_hdf5_method(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False) -> Dict:
    """
    Automatically select the best CSV to HDF5 conversion method based on dataset size.
    
    Args:
        xyz_files: List of paths to XYZ table files with patch numbers
        save_path: Path to save the HDF5 file
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    import psutil
    
    n_timesteps = len(xyz_files)
    
    # Get system memory info
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    
    # Estimate memory requirements
    first_df = pd.read_csv(xyz_files[0], low_memory=False)
    n_points = len(first_df)
    n_properties = len(first_df.columns)
    
    # Memory per timestep in GB
    memory_per_timestep_gb = (n_points * n_properties * 8) / (1024**3)
    total_memory_needed_gb = memory_per_timestep_gb * n_timesteps
    
    print(f"📊 CSV to HDF5 conversion analysis:")
    print(f"   Files to process: {n_timesteps}")
    print(f"   Available RAM: {available_gb:.1f} GB")
    print(f"   Estimated memory needed: {total_memory_needed_gb:.1f} GB")
    
    # Select method based on memory requirements
    if total_memory_needed_gb > available_gb * 0.5:  # If needs more than 50% of available RAM
        print("🛡️  Using MEMORY-SAFE PARALLEL method (large dataset)")
        return load_tables_to_3d_array_memory_safe_parallel(xyz_files, save_path, overwrite_existing)
    else:
        print("🚀 Using STANDARD PARALLEL method (dataset fits in memory)")
        return load_tables_to_3d_array_parallel(xyz_files, save_path, overwrite_existing) 