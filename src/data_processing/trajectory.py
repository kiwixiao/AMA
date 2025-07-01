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

def load_tables_to_3d_array(xyz_files: List[Path], save_path: str = 'cfd_data.h5') -> Dict:
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
        # Create datasets
        data = f.create_dataset('data', (n_timesteps, n_points, n_properties), dtype='float32')
        times = f.create_dataset('times', (n_timesteps,), dtype='float32')
        
        # Store property names exactly as they appear in the raw table
        f.attrs['properties'] = properties
        
        # Load all tables
        for i, xyz_file in enumerate(tqdm(xyz_files, desc="Loading tables")):
            df = pd.read_csv(xyz_file, low_memory=False)
            table_num = int(str(xyz_file).split('_')[-1].split('.')[0])
            times[i] = table_num * 0.001  # Convert to seconds
            
            # Store data for each property in the same order as the raw table
            for j, prop in enumerate(properties):
                data[i, :, j] = df[prop].values
    
    print(f"\nSaved 3D data to {save_path}")
    print("Properties preserved in the same order as raw table")
    return {'file_path': save_path, 'properties': properties}

def filter_data_by_breathing_cycle(data_info: Dict, subject_name: str) -> Dict:
    """
    Filter the 3D data to include only points within clean breathing cycles.
    
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
    
    print("Filtering data based on breathing cycle...")
    
    # Load flow profile data
    flow_data = pd.read_csv(f'{subject_name}FlowProfile.csv')
    
    # Get column names and identify time and flow columns
    columns = flow_data.columns.tolist()
    print(f"\nFlow profile columns: {columns}")
    
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
    
    print(f"\nFound {len(zero_crossings)} zero crossings")
    print(f"Time range: {cycle_times[0]:.2f} ms to {cycle_times[-1]:.2f} ms")
    
    # Load original data
    with h5py.File(data_info['file_path'], 'r') as f:
        print("\nAvailable datasets in HDF5 file:", list(f.keys()))
        
        # Load times and data
        times = f['times'][:]
        data = f['data'][:]
        
        # Check if properties exist
        properties = None
        if 'properties' in f:
            properties = f['properties'][:]
            print("Loaded properties from HDF5 file")
        else:
            print("No properties found in HDF5 file")
    
    # Find indices of times within the clean breathing cycle
    valid_indices = []
    
    # Use the entire time range between first and last zero crossing
    start_time = cycle_times[0]
    end_time = cycle_times[-1]
    
    # Find indices of times within this range
    valid_indices = np.where((times >= start_time) & (times <= end_time))[0]
    
    if len(valid_indices) == 0:
        print("Warning: No valid breathing cycles found!")
        return data_info
    
    print(f"\nFound {len(valid_indices)} timesteps within breathing cycle")
    print(f"Time range: {times[valid_indices[0]]:.2f} ms to {times[valid_indices[-1]]:.2f} ms")
    
    # Create filtered dataset
    filtered_file_path = Path(f'{subject_name}_cfd_data_filtered.h5')
    with h5py.File(filtered_file_path, 'w') as f:
        # Save filtered data
        f.create_dataset('times', data=times[valid_indices])
        f.create_dataset('data', data=data[valid_indices])
        
        # Save properties if they exist
        if properties is not None:
            f.create_dataset('properties', data=properties)
        
        # Add metadata about filtering
        f.attrs['num_original_timesteps'] = len(times)
        f.attrs['num_filtered_timesteps'] = len(valid_indices)
        f.attrs['filtering_description'] = 'Data filtered to include only complete breathing cycles'
        f.attrs['subject_name'] = subject_name
    
    print(f"\nOriginal timesteps: {len(times)}")
    print(f"Filtered timesteps: {len(valid_indices)}")
    print(f"Filtered data saved to: {filtered_file_path}")
    
    # Return updated data info
    return {
        'file_path': str(filtered_file_path),
        'times': times[valid_indices],
        'properties': properties
    }

def track_point_movement_3d(data_info: Dict, patch_number: int, face_index: int) -> Dict:
    """
    Track the movement of a specific point through time.
    
    Args:
        data_info: Dictionary containing the HDF5 file path and metadata
        patch_number: The patch number to track
        face_index: The face index within the patch to track
        
    Returns:
        dict: Dictionary containing trajectory information
    """
    import h5py
    import numpy as np
    
    print(f"\nTracking point: Patch {patch_number}, Face {face_index}")
    
    with h5py.File(data_info['file_path'], 'r') as f:
        data = f['data'][:]
        times = f['times'][:]
        
        # Get properties from HDF5 file
        if 'properties' in f.attrs:
            properties = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f.attrs['properties']]
            print(f"\nAvailable properties: {properties}")
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
            
            print("\nFound property indices:")
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
    
    # Find points matching patch and face
    trajectory = []
    velocities = []
    accelerations = []
    pressures = []
    vdotn_values = []
    
    prev_pos = None
    prev_vel = None
    prev_time = None
    
    for t_idx, time in enumerate(times):
        # Find the point with matching patch and face
        matches = np.where(
            (data[t_idx, :, patch_idx] == patch_number) & 
            (data[t_idx, :, face_idx] == face_index)
        )[0]
        
        if len(matches) == 0:
            print(f"Warning: Point not found at time {time}")
            continue
        elif len(matches) > 1:
            print(f"Warning: Multiple matches found at time {time}, using first")
        
        point_idx = matches[0]
        
        # Get position
        pos = data[t_idx, point_idx, [x_idx, y_idx, z_idx]]
        trajectory.append((time, pos))
        
        # Get pressure if available
        if pressure_idx is not None:
            pressure = data[t_idx, point_idx, pressure_idx]
            pressures.append((time, pressure))
        
        # Get VdotN if available
        if vdotn_idx is not None:
            vdotn = data[t_idx, point_idx, vdotn_idx]
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
    
    return result

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