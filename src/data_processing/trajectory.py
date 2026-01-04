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


def scan_csv_files_for_dimensions(xyz_files: List[Path], sample_size: int = None) -> Dict:
    """
    Scan CSV files to find maximum point count and detect remesh events.

    For remesh cases, the mesh can have different numbers of points before and after.
    This function finds the maximum to ensure HDF5 can hold all data.

    Args:
        xyz_files: List of paths to XYZ table files
        sample_size: If set, only sample this many files (first, last, and evenly spaced)
                     Set to None to scan all files (slower but accurate)

    Returns:
        Dictionary with:
        - max_points: Maximum number of points across all files
        - min_points: Minimum number of points (for info)
        - point_counts: List of point counts per file (if sample_size is None)
        - has_variable_points: True if point counts vary (indicates remesh)
        - properties: List of column names from first file
    """
    if not xyz_files:
        return {'max_points': 0, 'min_points': 0, 'has_variable_points': False, 'properties': []}

    # Determine which files to scan
    if sample_size is not None and len(xyz_files) > sample_size:
        # Sample first, last, and evenly spaced files
        indices = [0, len(xyz_files) - 1]
        step = len(xyz_files) // (sample_size - 2) if sample_size > 2 else 1
        indices.extend(range(step, len(xyz_files) - 1, step))
        indices = sorted(set(indices))[:sample_size]
        files_to_scan = [xyz_files[i] for i in indices]
    else:
        files_to_scan = xyz_files

    point_counts = []
    properties = None

    print(f"🔍 Scanning {len(files_to_scan)} CSV files for dimensions...")

    for f in tqdm(files_to_scan, desc="Scanning files", leave=False):
        try:
            # Read just the first few rows to get column count, then count rows
            df_sample = pd.read_csv(f, nrows=1, low_memory=False)
            if properties is None:
                properties = df_sample.columns.tolist()

            # Count rows efficiently
            with open(f, 'r') as file:
                row_count = sum(1 for _ in file) - 1  # Subtract header
            point_counts.append(row_count)
        except Exception as e:
            print(f"⚠️  Error scanning {f.name}: {e}")
            continue

    if not point_counts:
        return {'max_points': 0, 'min_points': 0, 'has_variable_points': False, 'properties': []}

    max_points = max(point_counts)
    min_points = min(point_counts)
    has_variable_points = max_points != min_points

    if has_variable_points:
        print(f"   ⚠️  Variable point counts detected (remesh likely):")
        print(f"      Min: {min_points:,} points, Max: {max_points:,} points")
        print(f"      Using max ({max_points:,}) for HDF5 dimensions")
    else:
        print(f"   ✅ Consistent point count: {max_points:,} points")

    return {
        'max_points': max_points,
        'min_points': min_points,
        'point_counts': point_counts if sample_size is None else None,
        'has_variable_points': has_variable_points,
        'properties': properties or []
    }


def load_tables_to_3d_array(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False) -> Dict:
    """
    Load all tables into a 3D array format and save as HDF5.
    If HDF5 file already exists, load from it instead of processing tables again.
    Format: (time_steps, max_points, properties)

    Handles remesh cases where point counts vary between timesteps by using
    the maximum point count and padding smaller files with NaN.

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
                if all(key in f for key in ['cfd_data', 'time_points']) and 'column_names' in f.attrs:
                    print("File structure verified. Loading existing data...")
                    return {'file_path': save_path, 'properties': [name.decode('utf-8') for name in f.attrs['column_names']]}
                else:
                    print("Existing file has incorrect structure. Will recreate...")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Will recreate the data file...")
    elif Path(save_path).exists() and overwrite_existing:
        print(f"\n🔄 Force rerun enabled - recreating HDF5 file: {save_path}")

    print("\nLoading all tables into 3D array format...")

    # Scan all files to find maximum point count (handles remesh cases)
    dim_info = scan_csv_files_for_dimensions(xyz_files)
    n_points = dim_info['max_points']
    has_variable_points = dim_info['has_variable_points']
    n_timesteps = len(xyz_files)

    # Read first file for properties
    first_df = pd.read_csv(xyz_files[0], low_memory=False)

    # Use all columns from the raw table, adding Patch Number if not present
    properties = first_df.columns.tolist()
    if 'Patch Number' not in properties:
        # Add Patch Number after Face Index for raw CSV files
        if 'Face Index' in properties:
            face_idx_pos = properties.index('Face Index')
            properties.insert(face_idx_pos + 1, 'Patch Number')
        else:
            properties.append('Patch Number')
        print("📋 Added 'Patch Number' to properties list (will be computed from Face Index)")

    n_properties = len(properties)

    print(f"Data dimensions:")
    print(f"Time steps: {n_timesteps}")
    print(f"Max points per step: {n_points}" + (" (variable - remesh detected)" if has_variable_points else ""))
    print(f"Properties tracked: {n_properties}")
    print("\nProperties being tracked:")
    for prop in properties:
        print(f"  - {prop}")

    # Create HDF5 file
    with h5py.File(save_path, 'w') as f:
        # Create datasets with appropriate data types
        # Use NaN for padding so we can distinguish from real zeros
        data = f.create_dataset('cfd_data', (n_timesteps, n_points, n_properties), dtype='float64', fillvalue=np.nan)
        times = f.create_dataset('time_points', (n_timesteps,), dtype='float64')
        point_counts = f.create_dataset('point_counts', (n_timesteps,), dtype='int64')

        # Store property names exactly as they appear in the raw table
        f.attrs['column_names'] = [p.encode('utf-8') for p in properties]
        f.attrs['has_variable_points'] = has_variable_points
        f.attrs['max_points'] = n_points

        # Load all tables
        for i, xyz_file in enumerate(tqdm(xyz_files, desc="Loading tables")):
            df = pd.read_csv(xyz_file, low_memory=False)
            actual_points = len(df)
            point_counts[i] = actual_points

            # Extract timestep using robust method (handles scientific notation, decimals, etc.)
            from utils.file_processing import extract_timestep_from_filename
            table_time = extract_timestep_from_filename(xyz_file) * 0.001  # Convert to seconds
            times[i] = table_time

            # Store data for each property in the same order as the raw table
            for j, prop in enumerate(properties):
                try:
                    # Convert to numeric, handling any non-numeric values
                    values = pd.to_numeric(df[prop], errors='coerce').fillna(0.0)
                    # Only fill up to actual_points, rest stays as NaN (fillvalue)
                    data[i, :actual_points, j] = values.values
                except Exception as e:
                    print(f"Warning: Error converting column '{prop}' to numeric: {e}")
                    # Fill with zeros if conversion fails
                    data[i, :actual_points, j] = np.zeros(actual_points)

    print(f"\nSaved 3D data to {save_path}")
    print("Properties preserved in the same order as raw table")
    if has_variable_points:
        print("📊 Variable point counts stored per timestep for remesh handling")
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

    Handles remesh cases where point counts vary between timesteps by using
    the maximum point count and padding smaller files with NaN.

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
                if all(key in f for key in ['cfd_data', 'time_points']) and 'column_names' in f.attrs:
                    print("File structure verified. Loading existing data...")
                    return {'file_path': save_path, 'properties': [name.decode('utf-8') for name in f.attrs['column_names']]}
                else:
                    print("Existing file has incorrect structure. Will recreate...")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print("Will recreate the data file...")

    print("\n🚀 Loading all tables into 3D array format (PARALLEL)...")

    # IMPORTANT: xyz_files is already sorted in natural chronological order by main.py
    # The parallel processing preserves this order by using the original file index
    print("📋 Files are pre-sorted in natural chronological order for time-series consistency")

    # Scan all files to find maximum point count (handles remesh cases)
    dim_info = scan_csv_files_for_dimensions(xyz_files)
    n_points = dim_info['max_points']
    has_variable_points = dim_info['has_variable_points']
    n_timesteps = len(xyz_files)

    # Read first file for properties
    first_df = pd.read_csv(xyz_files[0], low_memory=False)

    # Use all columns from the raw table, adding Patch Number if not present
    properties = first_df.columns.tolist()
    if 'Patch Number' not in properties:
        # Add Patch Number after Face Index for raw CSV files
        if 'Face Index' in properties:
            face_idx_pos = properties.index('Face Index')
            properties.insert(face_idx_pos + 1, 'Patch Number')
        else:
            properties.append('Patch Number')
        print("📋 Added 'Patch Number' to properties list (will be computed from Face Index)")

    n_properties = len(properties)

    print(f"Data dimensions:")
    print(f"Time steps: {n_timesteps}")
    print(f"Max points per step: {n_points}" + (" (variable - remesh detected)" if has_variable_points else ""))
    print(f"Properties tracked: {n_properties}")
    print(f"🔥 Using parallel processing for {n_timesteps} CSV files...")
    
    # Get optimal process count with smart scaling
    logical_cores = mp.cpu_count()  # Includes hyperthreading
    try:
        # Get physical cores (better for CPU-intensive tasks like CSV parsing)
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores is None:
            physical_cores = logical_cores // 2  # Fallback: assume hyperthreading
    except:
        physical_cores = logical_cores // 2  # Conservative fallback
    
    cpu_cores = physical_cores  # Use physical cores for CPU-intensive CSV processing
    print(f"🖥️  CPU Detection: {logical_cores} logical cores, {physical_cores} physical cores (using physical)")
    
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    
    # Smart process cap based on PHYSICAL core count and memory
    if cpu_cores >= 32 and available_gb >= 64:
        max_processes = cpu_cores  # Use all physical cores for high-end servers
    elif cpu_cores >= 16 and available_gb >= 32:
        max_processes = cpu_cores  # Use all physical cores for mid-range servers  
    elif cpu_cores >= 8 and available_gb >= 16:
        max_processes = cpu_cores  # Use all physical cores for workstations
    else:
        max_processes = min(8, cpu_cores)  # Conservative cap for smaller systems
    
    optimal_processes = min(cpu_cores, n_timesteps, max_processes)
    print(f"📊 Using {optimal_processes} parallel processes")
    
    # Create HDF5 file structure
    with h5py.File(save_path, 'w') as f:
        # Create datasets with appropriate data types
        # Use NaN for padding so we can distinguish from real zeros
        data = f.create_dataset('cfd_data', (n_timesteps, n_points, n_properties), dtype='float64', fillvalue=np.nan)
        times = f.create_dataset('time_points', (n_timesteps,), dtype='float64')
        point_counts = f.create_dataset('point_counts', (n_timesteps,), dtype='int64')

        # Store property names exactly as they appear in the raw table
        f.attrs['column_names'] = [p.encode('utf-8') for p in properties]
        f.attrs['has_variable_points'] = has_variable_points
        f.attrs['max_points'] = n_points

        # Process CSV files in parallel
        start_time = time.time()

        # Prepare arguments for parallel processing (include max_points for padding)
        file_args = [(i, xyz_file, properties, n_points) for i, xyz_file in enumerate(xyz_files)]

        # Use parallel processing with proper progress tracking
        completed_files = 0

        with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(process_single_csv_file, args): args[0] for args in file_args}

            # Process completed tasks
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    table_data, table_time, actual_points = future.result()

                    # Write data to HDF5
                    data[index] = table_data
                    times[index] = table_time
                    point_counts[index] = actual_points

                    completed_files += 1
                    if completed_files % 50 == 0 or completed_files == n_timesteps:
                        progress = (completed_files / n_timesteps) * 100
                        elapsed = time.time() - start_time
                        print(f"  Progress: {progress:.1f}% ({completed_files}/{n_timesteps} files, {elapsed:.1f}s)")

                except Exception as e:
                    print(f"Error processing file {index}: {e}")
                    # Fill with NaN if processing fails
                    data[index] = np.full((n_points, n_properties), np.nan)
                    times[index] = 0.0
                    point_counts[index] = 0

    total_time = time.time() - start_time
    print(f"\n✅ PARALLEL conversion completed in {total_time:.1f} seconds")
    print(f"📈 Performance: {n_timesteps/total_time:.1f} files/second")
    print(f"💾 Saved 3D data to {save_path}")
    if has_variable_points:
        print("📊 Variable point counts stored per timestep for remesh handling")

    return {'file_path': save_path, 'properties': properties}

def process_single_csv_file(args):
    """
    Process a single CSV file for parallel loading.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    Pads output to max_points with NaN for remesh compatibility.

    Args:
        args: Tuple of (index, file_path, properties, max_points)
              max_points can be None for backward compatibility (no padding)

    Returns:
        Tuple of (processed_data, time_value, actual_points)
    """
    # Handle both old (3-arg) and new (4-arg) formats
    if len(args) == 4:
        index, xyz_file, properties, max_points = args
    else:
        index, xyz_file, properties = args
        max_points = None

    try:
        # Read CSV file
        df = pd.read_csv(xyz_file, low_memory=False)

        # Check if we need to add patch numbers (for raw CSV files)
        if 'Patch Number' not in df.columns:
            # Add patch numbers using Face Index reset logic
            patch_numbers = []
            current_patch = 1
            prev_face_idx = -1

            for _, row in df.iterrows():
                face_idx = row['Face Index']
                # Start new patch when Face Index resets to 0 (after being > 0)
                if face_idx == 0 and prev_face_idx > 0:
                    current_patch += 1
                patch_numbers.append(current_patch)
                prev_face_idx = face_idx

            # Add Patch Number column right after Face Index
            face_idx_col = df.columns.get_loc('Face Index')
            df.insert(face_idx_col + 1, 'Patch Number', patch_numbers)

            print(f"  Added patch numbers to file {index} ({len(set(patch_numbers))} patches)")

        # Extract timestep using robust method (handles scientific notation, decimals, etc.)
        from utils.file_processing import extract_timestep_from_filename
        table_time = extract_timestep_from_filename(xyz_file) * 0.001  # Convert to seconds

        # Process data for each property (now includes Patch Number if it was added)
        actual_points = len(df)
        n_properties = len(properties)

        # Use max_points for output size if specified (for remesh padding)
        output_points = max_points if max_points is not None else actual_points
        table_data = np.full((output_points, n_properties), np.nan)  # Fill with NaN for padding

        for j, prop in enumerate(properties):
            try:
                # Handle Patch Number specially (it's integer)
                if prop == 'Patch Number':
                    if prop in df.columns:
                        table_data[:actual_points, j] = df[prop].values.astype(float)
                    else:
                        # This shouldn't happen now, but just in case
                        table_data[:actual_points, j] = np.zeros(actual_points)
                else:
                    # Convert to numeric, handling any non-numeric values
                    values = pd.to_numeric(df[prop], errors='coerce').fillna(0.0)
                    table_data[:actual_points, j] = values.values
            except Exception as e:
                print(f"Warning: Error converting column '{prop}' in file {index}: {e}")
                # Fill with zeros if conversion fails
                table_data[:actual_points, j] = np.zeros(actual_points)

        return table_data, table_time, actual_points

    except Exception as e:
        print(f"Error processing file {xyz_file}: {e}")
        # Return NaN-filled array if file processing fails
        n_properties = len(properties)
        output_points = max_points if max_points is not None else 1
        return np.full((output_points, n_properties), np.nan), 0.0, 0

def load_tables_to_3d_array_memory_safe_parallel(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False) -> Dict:
    """
    MEMORY-SAFE PARALLEL VERSION: For very large datasets that might not fit in memory.
    Processes CSV files in parallel chunks and writes to HDF5 incrementally.
    Uses intelligent resource detection to automatically calculate optimal chunks.
    
    Args:
        xyz_files: List of paths to XYZ table files with patch numbers
        save_path: Path to save the HDF5 file
        overwrite_existing: Whether to overwrite existing HDF5 file
        
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
                if all(key in f for key in ['cfd_data', 'time_points']) and 'column_names' in f.attrs:
                    print("File structure verified. Loading existing data...")
                    return {'file_path': save_path, 'properties': [name.decode('utf-8') for name in f.attrs['column_names']]}
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

    # Scan all files to find maximum point count (handles remesh cases)
    dim_info = scan_csv_files_for_dimensions(xyz_files)
    n_points = dim_info['max_points']
    has_variable_points = dim_info['has_variable_points']
    n_timesteps = len(xyz_files)

    # Read first file for properties
    first_df = pd.read_csv(xyz_files[0], low_memory=False)

    # Use all columns from the raw table, adding Patch Number if not present
    properties = first_df.columns.tolist()
    if 'Patch Number' not in properties:
        # Add Patch Number after Face Index for raw CSV files
        if 'Face Index' in properties:
            face_idx_pos = properties.index('Face Index')
            properties.insert(face_idx_pos + 1, 'Patch Number')
        else:
            properties.append('Patch Number')
        print("📋 Added 'Patch Number' to properties list (will be computed from Face Index)")

    n_properties = len(properties)

    # USER'S INTUITIVE APPROACH: Calculate actual resource usage
    print(f"Data dimensions:")
    print(f"Time steps: {n_timesteps}")
    print(f"Max points per step: {n_points}" + (" (variable - remesh detected)" if has_variable_points else ""))
    print(f"Properties tracked: {n_properties}")
    
    # Step 1: Get total size of all CSV files
    print("📊 Calculating total CSV file sizes...")
    total_csv_size_gb = sum(csv_file.stat().st_size for csv_file in xyz_files) / (1024**3)
    avg_csv_size_mb = (total_csv_size_gb * 1024) / n_timesteps
    print(f"   Total CSV files size: {total_csv_size_gb:.1f} GB")
    print(f"   Average CSV file size: {avg_csv_size_mb:.1f} MB")
    
    # Step 2: Get available memory with safety factor
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    working_memory_gb = available_gb * 0.95  # Use 95% of available RAM with 5% buffer
    
    print(f"   Available RAM: {available_gb:.1f} GB")
    print(f"   Working memory: {working_memory_gb:.1f} GB (95% safety factor)")
    
    # Step 3: Calculate chunks needed based on memory ratio
    # Account for pandas overhead (3-4x file size in memory)
    pandas_overhead_factor = 3.5  # Conservative estimate
    effective_csv_size_gb = total_csv_size_gb * pandas_overhead_factor
    
    if effective_csv_size_gb <= working_memory_gb:
        # All files fit in memory - process in one batch
        n_chunks = 1
        files_per_chunk = n_timesteps
    else:
        # Calculate how many chunks we need
        n_chunks = int(np.ceil(effective_csv_size_gb / working_memory_gb))
        files_per_chunk = int(np.ceil(n_timesteps / n_chunks))
    
    print(f"   Total effective memory needed: {effective_csv_size_gb:.1f} GB (with pandas overhead)")
    print(f"   Number of chunks needed: {n_chunks}")
    print(f"   Files per chunk: {files_per_chunk}")
    
    # Step 4: Get optimal CPU cores (physical cores for CSV processing)
    logical_cores = mp.cpu_count()
    try:
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores is None:
            physical_cores = logical_cores // 2  # Fallback: assume hyperthreading
    except:
        physical_cores = logical_cores // 2  # Conservative fallback
    
    cpu_cores = physical_cores  # Use physical cores for CPU-intensive CSV processing
    print(f"🖥️  CPU Detection: {logical_cores} logical cores, {physical_cores} physical cores (using physical)")
    
    # Step 5: Use all available cores efficiently
    optimal_processes = min(cpu_cores, files_per_chunk)  # Don't exceed files in chunk
    batch_size = files_per_chunk  # Process one chunk at a time
    
    print(f"📊 INTUITIVE resource allocation:")
    print(f"   Parallel processes: {optimal_processes}")
    print(f"   Batch size: {batch_size} files")
    print(f"   Estimated memory per batch: {(batch_size * avg_csv_size_mb * pandas_overhead_factor / 1024):.1f} GB")
    print(f"   Number of batches: {n_chunks}")
    print(f"   Memory safety: ✅ Guaranteed (chunks fit in {working_memory_gb:.1f} GB)")
    
    # Create HDF5 file structure
    with h5py.File(save_path, 'w') as f:
        # Create datasets with appropriate data types
        # Use NaN for padding so we can distinguish from real zeros
        data = f.create_dataset('cfd_data', (n_timesteps, n_points, n_properties), dtype='float64', fillvalue=np.nan)
        times = f.create_dataset('time_points', (n_timesteps,), dtype='float64')
        point_counts = f.create_dataset('point_counts', (n_timesteps,), dtype='int64')

        # Store property names exactly as they appear in the raw table
        f.attrs['column_names'] = [p.encode('utf-8') for p in properties]
        f.attrs['has_variable_points'] = has_variable_points
        f.attrs['max_points'] = n_points

        # Process CSV files in parallel batches
        start_time = time.time()
        total_processed = 0

        for chunk_idx in range(n_chunks):
            batch_start = chunk_idx * files_per_chunk
            batch_end = min(batch_start + files_per_chunk, n_timesteps)
            batch_files = xyz_files[batch_start:batch_end]

            print(f"\n🔄 Processing chunk {chunk_idx + 1}/{n_chunks}: files {batch_start}-{batch_end-1}")

            # Prepare arguments for this batch (include max_points for padding)
            file_args = [(i, xyz_file, properties, n_points) for i, xyz_file in enumerate(batch_files)]

            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
                # Submit all tasks in this batch
                future_to_index = {executor.submit(process_single_csv_file, args): args[0] for args in file_args}

                # Process completed tasks
                for future in as_completed(future_to_index):
                    local_index = future_to_index[future]
                    global_index = batch_start + local_index

                    try:
                        table_data, table_time, actual_points = future.result()

                        # Write data to HDF5
                        data[global_index] = table_data
                        times[global_index] = table_time
                        point_counts[global_index] = actual_points

                        total_processed += 1

                    except Exception as e:
                        print(f"Error processing file {global_index}: {e}")
                        # Fill with NaN if processing fails
                        data[global_index] = np.full((n_points, n_properties), np.nan)
                        times[global_index] = 0.0
                        point_counts[global_index] = 0
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
    if has_variable_points:
        print("📊 Variable point counts stored per timestep for remesh handling")

    return {'file_path': save_path, 'properties': properties}

def auto_select_csv_to_hdf5_method(xyz_files: List[Path], save_path: str = 'cfd_data.h5', overwrite_existing: bool = False,
                                   breathing_cycle_start_ms: float = None, breathing_cycle_end_ms: float = None) -> Dict:
    """
    Automatically select the best CSV to HDF5 conversion method based on dataset size.
    Uses intelligent resource detection to prevent memory overflow.

    Args:
        xyz_files: List of paths to XYZ table files with patch numbers
        save_path: Path to save the HDF5 file
        overwrite_existing: Whether to overwrite existing HDF5 file
        breathing_cycle_start_ms: Start time of breathing cycle in milliseconds (optional)
        breathing_cycle_end_ms: End time of breathing cycle in milliseconds (optional)

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
    
    # Account for Patch Number column if it will be added
    properties = first_df.columns.tolist()
    if 'Patch Number' not in properties:
        n_properties = len(properties) + 1  # +1 for Patch Number
    else:
        n_properties = len(properties)
    
    # Calculate final data size (disk space for HDF5 file)
    data_size_per_timestep_gb = (n_points * n_properties * 8) / (1024**3)
    total_data_size_gb = data_size_per_timestep_gb * n_timesteps
    
    print(f"📊 CSV to HDF5 conversion analysis:")
    print(f"   Files to process: {n_timesteps}")
    print(f"   Available RAM: {available_gb:.1f} GB")
    print(f"   Final HDF5 file size (disk): {total_data_size_gb:.1f} GB")
    
    # Select method based on dataset size (large datasets need memory-safe processing)
    if total_data_size_gb > available_gb * 0.5:  # If final data is large relative to RAM
        print("🛡️  Using MEMORY-SAFE PARALLEL method (large dataset)")
        result = load_tables_to_3d_array_memory_safe_parallel(xyz_files, save_path, overwrite_existing)
    else:
        print("🚀 Using STANDARD PARALLEL method (dataset fits in memory)")
        result = load_tables_to_3d_array_parallel(xyz_files, save_path, overwrite_existing)

    # Store breathing cycle metadata in HDF5 if provided
    if breathing_cycle_start_ms is not None and breathing_cycle_end_ms is not None:
        try:
            with h5py.File(save_path, 'a') as f:
                f.attrs['breathing_cycle_start_ms'] = breathing_cycle_start_ms
                f.attrs['breathing_cycle_end_ms'] = breathing_cycle_end_ms
                f.attrs['time_offset_ms'] = breathing_cycle_start_ms  # For normalization
                print(f"📊 Stored breathing cycle metadata: {breathing_cycle_start_ms:.2f}ms - {breathing_cycle_end_ms:.2f}ms")
        except Exception as e:
            print(f"⚠️  Warning: Could not store breathing cycle metadata: {e}")

    return result


def get_breathing_cycle_metadata(hdf5_file_path: str) -> dict:
    """
    Read breathing cycle metadata from HDF5 file.

    Args:
        hdf5_file_path: Path to the HDF5 file

    Returns:
        Dictionary with breathing cycle info:
        - breathing_cycle_start_ms: Start time in milliseconds
        - breathing_cycle_end_ms: End time in milliseconds
        - time_offset_ms: Time offset for normalization
        Returns empty dict if metadata not found.
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            metadata = {}
            if 'breathing_cycle_start_ms' in f.attrs:
                metadata['breathing_cycle_start_ms'] = float(f.attrs['breathing_cycle_start_ms'])
            if 'breathing_cycle_end_ms' in f.attrs:
                metadata['breathing_cycle_end_ms'] = float(f.attrs['breathing_cycle_end_ms'])
            if 'time_offset_ms' in f.attrs:
                metadata['time_offset_ms'] = float(f.attrs['time_offset_ms'])

            if metadata:
                print(f"📊 Loaded breathing cycle metadata from HDF5:")
                print(f"   Start: {metadata.get('breathing_cycle_start_ms', 'N/A')}ms")
                print(f"   End: {metadata.get('breathing_cycle_end_ms', 'N/A')}ms")
            return metadata
    except Exception as e:
        print(f"⚠️  Could not read breathing cycle metadata: {e}")
        return {}


def store_remesh_metadata(hdf5_file_path: str,
                          has_remesh: bool,
                          remesh_before_file: str = None,
                          remesh_after_file: str = None,
                          remesh_timestep_ms: float = None,
                          remesh_events: list = None) -> bool:
    """
    Store remesh metadata in HDF5 file.

    When CFD simulation has mesh remeshing during the run, patch_number and
    face_index change but physical coordinates stay the same. This metadata
    helps Phase 2 know how to handle tracking across mesh changes.

    Supports multiple remesh events via remesh_events parameter.

    Args:
        hdf5_file_path: Path to the HDF5 file
        has_remesh: Whether the simulation has remesh
        remesh_before_file: CSV filename of last timestep BEFORE remesh (first event, for backward compat)
        remesh_after_file: CSV filename of first timestep AFTER remesh (first event, for backward compat)
        remesh_timestep_ms: Timestamp (ms) where remesh occurs (first event, for backward compat)
        remesh_events: List of remesh events, each with {before_file, after_file, timestep_ms}

    Returns:
        True if successful, False otherwise
    """
    import json as json_module

    try:
        with h5py.File(hdf5_file_path, 'a') as f:
            f.attrs['has_remesh'] = has_remesh

            if has_remesh:
                # Store multiple remesh events as JSON string
                if remesh_events:
                    f.attrs['remesh_events_json'] = json_module.dumps(remesh_events)
                    f.attrs['remesh_event_count'] = len(remesh_events)

                    # Backward compatibility: store first event in old format
                    first_event = remesh_events[0]
                    f.attrs['remesh_before_file'] = first_event['before_file'].encode('utf-8')
                    f.attrs['remesh_after_file'] = first_event['after_file'].encode('utf-8')
                    f.attrs['remesh_timestep_ms'] = first_event['timestep_ms']

                    print(f"📊 Stored remesh metadata in HDF5:")
                    print(f"   Has remesh: {has_remesh}")
                    print(f"   Total remesh events: {len(remesh_events)}")
                    for i, event in enumerate(remesh_events, 1):
                        print(f"   #{i}: {event['before_file']} → {event['after_file']} @ {event['timestep_ms']:.1f}ms")

                elif remesh_before_file and remesh_after_file:
                    # Single event (backward compatibility)
                    f.attrs['remesh_before_file'] = remesh_before_file.encode('utf-8')
                    f.attrs['remesh_after_file'] = remesh_after_file.encode('utf-8')
                    if remesh_timestep_ms is not None:
                        f.attrs['remesh_timestep_ms'] = remesh_timestep_ms
                    f.attrs['remesh_event_count'] = 1

                    print(f"📊 Stored remesh metadata in HDF5:")
                    print(f"   Has remesh: {has_remesh}")
                    print(f"   Before file: {remesh_before_file}")
                    print(f"   After file: {remesh_after_file}")
                    if remesh_timestep_ms is not None:
                        print(f"   Remesh timestep: {remesh_timestep_ms:.2f}ms")
            else:
                print(f"📊 Stored remesh metadata: has_remesh = {has_remesh}")

        return True
    except Exception as e:
        print(f"⚠️  Could not store remesh metadata: {e}")
        return False


def get_remesh_metadata(hdf5_file_path: str) -> dict:
    """
    Read remesh metadata from HDF5 file.

    Args:
        hdf5_file_path: Path to the HDF5 file

    Returns:
        Dictionary with remesh info:
        - has_remesh: bool - Whether simulation has remesh
        - remesh_events: list - List of {before_file, after_file, timestep_ms} dicts
        - remesh_event_count: int - Number of remesh events
        # Backward compatibility (first event):
        - remesh_before_file: str - CSV filename before first remesh
        - remesh_after_file: str - CSV filename after first remesh
        - remesh_timestep_ms: float - Timestamp where first remesh occurs
        Returns dict with has_remesh=False if metadata not found.
    """
    import json as json_module

    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            metadata = {'has_remesh': False, 'remesh_events': [], 'remesh_event_count': 0}

            if 'has_remesh' in f.attrs:
                metadata['has_remesh'] = bool(f.attrs['has_remesh'])

            if metadata['has_remesh']:
                # Try to load multiple events first
                if 'remesh_events_json' in f.attrs:
                    val = f.attrs['remesh_events_json']
                    json_str = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                    metadata['remesh_events'] = json_module.loads(json_str)
                    metadata['remesh_event_count'] = len(metadata['remesh_events'])

                # Backward compatibility: load single event fields
                if 'remesh_before_file' in f.attrs:
                    val = f.attrs['remesh_before_file']
                    metadata['remesh_before_file'] = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                if 'remesh_after_file' in f.attrs:
                    val = f.attrs['remesh_after_file']
                    metadata['remesh_after_file'] = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                if 'remesh_timestep_ms' in f.attrs:
                    metadata['remesh_timestep_ms'] = float(f.attrs['remesh_timestep_ms'])
                if 'remesh_event_count' in f.attrs:
                    metadata['remesh_event_count'] = int(f.attrs['remesh_event_count'])

                # If no events list but have single event, create list for consistency
                if not metadata['remesh_events'] and metadata.get('remesh_before_file'):
                    metadata['remesh_events'] = [{
                        'before_file': metadata['remesh_before_file'],
                        'after_file': metadata['remesh_after_file'],
                        'timestep_ms': metadata.get('remesh_timestep_ms', 0)
                    }]
                    metadata['remesh_event_count'] = 1

                print(f"📊 Loaded remesh metadata from HDF5:")
                print(f"   Has remesh: {metadata['has_remesh']}")
                print(f"   Total remesh events: {metadata['remesh_event_count']}")
                for i, event in enumerate(metadata['remesh_events'], 1):
                    print(f"   #{i}: {event['before_file']} → {event['after_file']} @ {event['timestep_ms']:.1f}ms")

            return metadata
    except Exception as e:
        print(f"⚠️  Could not read remesh metadata: {e}")
        return {'has_remesh': False, 'remesh_events': [], 'remesh_event_count': 0}


def store_flow_profile(hdf5_file_path: str, flow_profile_path: str) -> bool:
    """
    Store flow profile data in HDF5 file.

    This allows Phase 2 to run without needing the original flow profile CSV.

    Args:
        hdf5_file_path: Path to the HDF5 file
        flow_profile_path: Path to the flow profile CSV file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read flow profile CSV
        flow_df = pd.read_csv(flow_profile_path)

        # Get column names (handle different naming conventions)
        time_col = None
        flow_col = None
        for col in flow_df.columns:
            col_lower = col.lower()
            if 'time' in col_lower:
                time_col = col
            elif 'mass' in col_lower or 'flow' in col_lower:
                flow_col = col

        if time_col is None or flow_col is None:
            print(f"⚠️  Could not identify time/flow columns in flow profile")
            return False

        time_data = flow_df[time_col].values
        flow_data = flow_df[flow_col].values

        with h5py.File(hdf5_file_path, 'a') as f:
            # Store flow profile data
            if 'flow_profile_time' in f:
                del f['flow_profile_time']
            if 'flow_profile_data' in f:
                del f['flow_profile_data']

            f.create_dataset('flow_profile_time', data=time_data)
            f.create_dataset('flow_profile_data', data=flow_data)

            # Store metadata
            f.attrs['flow_profile_stored'] = True
            f.attrs['flow_profile_source'] = Path(flow_profile_path).name.encode('utf-8')
            f.attrs['flow_profile_time_col'] = time_col.encode('utf-8')
            f.attrs['flow_profile_flow_col'] = flow_col.encode('utf-8')

        print(f"📊 Stored flow profile in HDF5:")
        print(f"   Source: {Path(flow_profile_path).name}")
        print(f"   Data points: {len(time_data)}")
        print(f"   Time range: {time_data[0]:.4f}s - {time_data[-1]:.4f}s")
        return True

    except Exception as e:
        print(f"⚠️  Could not store flow profile: {e}")
        return False


def get_flow_profile(hdf5_file_path: str) -> pd.DataFrame:
    """
    Read flow profile data from HDF5 file.

    Args:
        hdf5_file_path: Path to the HDF5 file

    Returns:
        DataFrame with time (s) and flow rate columns, or None if not found
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            if 'flow_profile_time' not in f or 'flow_profile_data' not in f:
                return None

            time_data = f['flow_profile_time'][:]
            flow_data = f['flow_profile_data'][:]

            # Get original column names
            time_col = 'time (s)'
            flow_col = 'Massflowrate (kg/s)'
            if 'flow_profile_time_col' in f.attrs:
                val = f.attrs['flow_profile_time_col']
                time_col = val.decode('utf-8') if isinstance(val, bytes) else str(val)
            if 'flow_profile_flow_col' in f.attrs:
                val = f.attrs['flow_profile_flow_col']
                flow_col = val.decode('utf-8') if isinstance(val, bytes) else str(val)

            df = pd.DataFrame({
                time_col: time_data,
                flow_col: flow_data
            })

            print(f"📊 Loaded flow profile from HDF5: {len(df)} data points")
            return df

    except Exception as e:
        print(f"⚠️  Could not read flow profile from HDF5: {e}")
        return None


def has_flow_profile(hdf5_file_path: str) -> bool:
    """Check if HDF5 file contains flow profile data."""
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            return 'flow_profile_time' in f and 'flow_profile_data' in f
    except:
        return False


def load_hdf5_data_for_html_plots(hdf5_file_path: str, time_point: int = None) -> pd.DataFrame:
    """
    Load data from HDF5 file for HTML plot generation.

    Args:
        hdf5_file_path: Path to the HDF5 file
        time_point: Specific time point to load (if None, loads first time point)

    Returns:
        DataFrame with the loaded data
    """
    import h5py
    import pandas as pd
    import numpy as np
    
    print(f"📊 Loading HDF5 data for HTML plots from: {hdf5_file_path}")
    
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # Handle both old and new HDF5 file formats
            if 'cfd_data' in f:
                data = f['cfd_data']
                times = f['time_points'][:]
                
                # Try new format first, then fall back to old format
                if 'column_names' in f.attrs:
                    column_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in f.attrs['column_names']]
                elif 'properties' in f.attrs:
                    column_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in f.attrs['properties']]
                else:
                    print("❌ No column information found in HDF5 file")
                    return None
            elif 'data' in f:
                # Old format compatibility
                data = f['data']
                times = f['times'][:]
                
                if 'properties' in f.attrs:
                    column_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in f.attrs['properties']]
                else:
                    print("❌ No properties found in old format HDF5 file")
                    return None
            else:
                print("❌ No data array found in HDF5 file")
                return None
            
            # Find the time point to load
            if time_point is None:
                # Load first time point
                time_idx = 0
                actual_time = times[0]
                print(f"  Loading first time point: {actual_time:.6f}s")
            else:
                # Find closest time point
                time_diffs = np.abs(times - time_point * 0.001)  # Convert ms to seconds
                time_idx = np.argmin(time_diffs)
                actual_time = times[time_idx]
                print(f"  Loading time point {time_point} (actual: {actual_time:.6f}s)")
            
            # Load data for the specific time point
            time_data = data[time_idx, :, :]
            
            # Create DataFrame
            df = pd.DataFrame(time_data, columns=column_names)
            
            # Add patch numbers based on Face Index resets (same logic as CSV processing)
            if 'Patch Number' not in df.columns:
                patch_numbers = []
                current_patch = 1
                prev_face_idx = -1
                
                for _, row in df.iterrows():
                    face_idx = row['Face Index']
                    # Start new patch when Face Index resets to 0 (after being > 0)
                    if face_idx == 0 and prev_face_idx > 0:
                        current_patch += 1
                    patch_numbers.append(current_patch)
                    prev_face_idx = face_idx
                
                # Add Patch Number column right after Face Index
                face_idx_col = df.columns.get_loc('Face Index')
                df.insert(face_idx_col + 1, 'Patch Number', patch_numbers)
                print(f"  Added Patch Number column with {len(np.unique(patch_numbers))} patches")
            else:
                print(f"  'Patch Number' column already exists, using existing values")
            
            # Get unique patch count for reporting
            unique_patches = len(df['Patch Number'].unique())
            print(f"  Loaded {len(df):,} points with {unique_patches} patches")
            
            return df
            
    except Exception as e:
        print(f"❌ Error loading HDF5 data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_hdf5_first_time_point(hdf5_file_path: str) -> float:
    """
    Get the first time point from HDF5 file.
    Handles both old and new HDF5 file formats.
    
    Args:
        hdf5_file_path: Path to the HDF5 file
        
    Returns:
        First time point in seconds
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            # Handle both old and new formats
            if 'time_points' in f:
                times = f['time_points'][:]
            elif 'times' in f:
                times = f['times'][:]
            else:
                print("❌ No time data found in HDF5 file")
                return None
            return times[0]
    except Exception as e:
        print(f"❌ Error getting first time point from HDF5: {e}")
        return None 