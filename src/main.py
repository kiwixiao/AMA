"""
Main script for CFD data analysis.
Processes trajectory data and creates visualization plots.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from typing import Tuple, List
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
import json
# Import additional dependencies for improved filtering
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Handle imports for both direct execution and module execution
import sys
from pathlib import Path

# Add src directory to path for direct execution
if __name__ == '__main__':
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Now import modules consistently
from utils.file_processing import (
    load_tracking_locations,
    preprocess_all_tables,
    preprocess_all_tables_parallel,
    find_breathing_cycle_bounds,
    filter_xyz_files_by_time,
    extract_timestep_from_filename,
    extract_base_subject
)
from utils.parallel_csv_processing import (
    track_point_parallel,
    track_patch_region_parallel,
    track_fixed_patch_region_csv_parallel,
    track_point_hdf5_parallel,
    track_fixed_patch_region_hdf5_parallel,
    auto_select_hdf5_tracking_method,
    auto_select_hdf5_point_tracking_method,
    track_fixed_patch_region_hdf5_memory_safe_multicore,
    track_point_hdf5_memory_safe_multicore,
    track_fixed_patch_region_hdf5_optimized_multicore,
    track_fixed_patch_region_hdf5_optimized_index_lookup,
    auto_select_optimized_hdf5_tracking_method,
    find_initial_region_points_hdf5_safe
)

# Import surface plots function with fallback
try:
    from visualization.surface_plots import plot_3d_interactive_all_patches
    SURFACE_PLOTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import surface plots: {e}")
    SURFACE_PLOTS_AVAILABLE = False
    def plot_3d_interactive_all_patches(*args, **kwargs):
        print("⚠️  Surface plots not available - skipping visualization")
        return None

from analysis.patch_region_analysis import (
    find_points_in_circle_enhanced,
    calculate_patch_statistics,
    analyze_multi_radius_patches
)

# Import visualization function (handle import gracefully)
try:
    from visualization.patch_visualization import visualize_patch_regions
    VISUALIZATION_AVAILABLE = True
    VISUALIZE_PATCH_REGIONS_FUNC = visualize_patch_regions
except ImportError as e:
    print(f"Warning: Could not import patch visualization: {e}")
    VISUALIZATION_AVAILABLE = False
    VISUALIZE_PATCH_REGIONS_FUNC = None

# Import CFD analysis functions
try:
    from visualization.cfd_analysis_3x3 import (
        create_cfd_analysis_3x3_panel,
        create_cfd_analysis_3x3_panel_with_markers,
        load_cfd_data_for_analysis
    )
    CFD_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import CFD analysis functions: {e}")
    CFD_ANALYSIS_AVAILABLE = False

def calculate_surface_normal(points: np.ndarray, center_idx: int, k: int = 10) -> np.ndarray:
    """
    Calculate surface normal at a point using local neighborhood PCA.
    
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

def find_connected_points_with_normal_filter(df: pd.DataFrame, center_point: Tuple[float, float, float], 
                                           radius: float, connectivity_threshold: float = 0.001,
                                           normal_angle_threshold: float = 60.0) -> pd.DataFrame:
    """
    Find connected points within radius that also satisfy surface normal constraints.
    
    Args:
        df: DataFrame with airway surface points
        center_point: (x, y, z) coordinates of center
        radius: Maximum distance from center
        connectivity_threshold: Maximum distance between connected points
        normal_angle_threshold: Maximum angle difference from reference normal (degrees)
    
    Returns:
        DataFrame with filtered connected points
    """
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
            reference_normal = calculate_surface_normal(ref_points, ref_center_idx)
            
            # Calculate normals for all connected points and filter by angle
            valid_indices = []
            for i, point_coords in enumerate(coords_connected):
                if ref_mask[i]:  # Always keep reference points
                    valid_indices.append(i)
                else:
                    # Calculate normal at this point
                    point_normal = calculate_surface_normal(coords_connected, i, k=8)
                    
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

def find_points_in_circle(df: pd.DataFrame, center_point: Tuple[float, float, float], radius: float, patch_number: int) -> pd.DataFrame:
    """
    Find all points within a circular region around a center point, considering only points
    whose normal vectors align with the center point's normal vector.
    
    Args:
        df: DataFrame containing point coordinates and data
        center_point: (x, y, z) coordinates of the center point
        radius: Radius of the circular region
        patch_number: The patch number of the center point
        
    Returns:
        DataFrame containing only the points within the circular region on the same side
    """
    x, y, z = center_point
    
    # Get the center point's data to determine its normal direction
    center_data = df[(df['X (m)'] == x) & (df['Y (m)'] == y) & (df['Z (m)'] == z)]
    if len(center_data) == 0:
        raise ValueError("Center point not found in data")
    
    # First filter to only include points from same patch and adjacent patches
    nearby_patches = [patch_number - 1, patch_number, patch_number + 1]
    df_nearby = df[df['Patch Number'].isin(nearby_patches)]
    
    # Get the area components and normalize them to create a unit normal vector for center point
    center_area = np.array([
        center_data['Area[i] (m^2)'].iloc[0],
        center_data['Area[j] (m^2)'].iloc[0],
        center_data['Area[k] (m^2)'].iloc[0]
    ])
    center_normal = center_area / np.linalg.norm(center_area)  # Normalize to unit vector
    
    # Create two orthogonal vectors in the tangent plane
    # First, find any vector not parallel to the normal
    if abs(center_normal[2]) < 0.9:
        tangent1 = np.array([0, 0, 1])
    else:
        tangent1 = np.array([1, 0, 0])
    # Make it perpendicular to normal
    tangent1 = tangent1 - np.dot(tangent1, center_normal) * center_normal
    tangent1 = tangent1 / np.linalg.norm(tangent1)
    # Get second tangent vector by cross product
    tangent2 = np.cross(center_normal, tangent1)
    
    # Calculate vectors from center to all points
    vectors_to_points = np.column_stack((
        df_nearby['X (m)'] - x,
        df_nearby['Y (m)'] - y,
        df_nearby['Z (m)'] - z
    ))
    
    # Project these vectors onto the tangent plane
    proj1 = np.dot(vectors_to_points, tangent1)
    proj2 = np.dot(vectors_to_points, tangent2)
    
    # Calculate radial distances in the tangent plane
    distances_in_plane = np.sqrt(proj1**2 + proj2**2)
    
    # Get normal vectors for points by normalizing their area components
    point_areas = np.column_stack((
        df_nearby['Area[i] (m^2)'],
        df_nearby['Area[j] (m^2)'],
        df_nearby['Area[k] (m^2)']
    ))
    # Normalize each row to get unit normal vectors
    magnitudes = np.linalg.norm(point_areas, axis=1)
    point_normals = point_areas / magnitudes[:, np.newaxis]
    
    # Calculate dot product between center normal and each point's normal
    # If dot product is positive, normals are pointing in similar direction
    normal_alignment = np.sum(point_normals * center_normal, axis=1)
    same_side = normal_alignment > 0.7  # Make alignment criteria more strict
    
    # Filter points that are both within radius and have aligned normals
    return df_nearby[(distances_in_plane <= radius) & same_side]

def calculate_patch_average_pressure(file_path: Path, patch_number: int = 35) -> float:
    """Calculate average Total Pressure for a given patch number.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Check if we need to add patch numbers (for raw CSV files)
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
        
        # Add Patch Number column
        df['Patch Number'] = patch_numbers
    
    patch_data = df[df['Patch Number'] == patch_number]
    if len(patch_data) == 0:
        return None
    return float(patch_data['Total Pressure (Pa)'].mean())

def find_initial_region_points(xyz_file: Path, patch_number: int, face_index: int, radius: float = 0.002, 
                             normal_angle_threshold: float = 60.0) -> list:
    """
    Find all points in the circular region around a center point in the first time step.
    Returns a list of (patch_number, face_index) pairs for all points in the region.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    
    Args:
        xyz_file: Path to the first time step XYZ table file
        patch_number: Patch number of the center point
        face_index: Face index of the center point
        radius: Radius of the circular region in meters (default: 2mm)
        
    Returns:
        List of (patch_number, face_index) tuples for all points in the region
    """
    df = pd.read_csv(xyz_file, low_memory=False)
    
    # Check if we need to add patch numbers (for raw CSV files)
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
        
        # Add Patch Number column
        df['Patch Number'] = patch_numbers
    
    # Get center point coordinates
    center_data = df[(df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)]
    if len(center_data) == 0:
        return []
    
    center_point = (
        float(center_data['X (m)'].iloc[0]),
        float(center_data['Y (m)'].iloc[0]),
        float(center_data['Z (m)'].iloc[0])
    )
    
    # Find points in the region using improved filtering
    region_points = find_connected_points_with_normal_filter(
        df, center_point, radius, 
        connectivity_threshold=0.001,
        normal_angle_threshold=normal_angle_threshold
    )
    
    # Get patch/face pairs for all points in the region
    point_pairs = list(zip(region_points['Patch Number'], region_points['Face Index']))
    
    print(f"Found {len(point_pairs)} connected points in {radius*1000:.1f}mm radius around Patch {patch_number}, Face {face_index} (normal filter: {normal_angle_threshold}°)")
    return point_pairs

def track_point_in_file(file_path: Path, patch_number: int, face_index: int) -> dict:
    """Track a specific point in a single time step file.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Check if we need to add patch numbers (for raw CSV files)
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
        
        # Add Patch Number column
        df['Patch Number'] = patch_numbers
    
    # Get point data
    point_data = df[(df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)]
    if len(point_data) == 0:
        return None
    
    if len(point_data) > 1:
        print(f"Warning: Multiple matches found in {file_path.name}, using first")
        point_data = point_data.iloc[0:1]
    
    # Calculate average pressure for patch 35 (skipping for now)
    patch_avg_pressure = 0.0  # Temporarily set to 0
    
    # Get time point from filename using robust extraction
    timestep = extract_timestep_from_filename(file_path)
    
    # Determine time unit based on filename format
    if 'e+' in file_path.stem or 'e-' in file_path.stem or '.' in file_path.stem:
        # Scientific notation or decimal format - likely in seconds
        time_sec = timestep
        time_point = int(timestep * 1000)  # Convert to milliseconds
    else:
        # Integer format - likely in milliseconds
        time_sec = timestep * 0.001  # Convert to seconds
        time_point = int(timestep)  # Already in milliseconds
    
    # Get velocity vector
    velocity_vector = np.array([
        float(point_data['Velocity[i] (m/s)'].iloc[0]),
        float(point_data['Velocity[j] (m/s)'].iloc[0]),
        float(point_data['Velocity[k] (m/s)'].iloc[0])
    ])
    
    # Calculate velocity magnitude
    velocity = float(np.linalg.norm(velocity_vector))
    
    # Get area vector and normalize to get normal vector
    area_vector = np.array([
        float(point_data['Area[i] (m^2)'].iloc[0]),
        float(point_data['Area[j] (m^2)'].iloc[0]),
        float(point_data['Area[k] (m^2)'].iloc[0])
    ])
    normal_vector = area_vector / np.linalg.norm(area_vector)
    
    # Calculate dot product between velocity and normal vectors
    vdotn = float(np.dot(velocity_vector, normal_vector))
    
    # Calculate signed velocity using dot product sign
    signed_velocity = velocity * np.sign(vdotn)
    
    point_pressure = float(point_data['Total Pressure (Pa)'].iloc[0])
    
    return {
        'time': time_sec,
        'time_point': time_point,
        'x': float(point_data['X (m)'].iloc[0]),
        'y': float(point_data['Y (m)'].iloc[0]),
        'z': float(point_data['Z (m)'].iloc[0]),
        'pressure': point_pressure,
        'adjusted_pressure': point_pressure - patch_avg_pressure,
        'velocity': velocity,
        'velocity_i': float(point_data['Velocity[i] (m/s)'].iloc[0]),
        'velocity_j': float(point_data['Velocity[j] (m/s)'].iloc[0]),
        'velocity_k': float(point_data['Velocity[k] (m/s)'].iloc[0]),
        'area_i': float(point_data['Area[i] (m^2)'].iloc[0]),
        'area_j': float(point_data['Area[j] (m^2)'].iloc[0]),
        'area_k': float(point_data['Area[k] (m^2)'].iloc[0]),
        'vdotn': vdotn,
        'signed_velocity': signed_velocity,
        'patch35_avg_pressure': patch_avg_pressure
    }

def track_patch_region_in_file(file_path: Path, patch_number: int, face_index: int, radius: float) -> dict:
    """Track a patch region around a center point in a single time step file.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Check if we need to add patch numbers (for raw CSV files)
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
        
        # Add Patch Number column
        df['Patch Number'] = patch_numbers
    
    # Find center point coordinates
    center_data = df[(df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)]
    if len(center_data) == 0:
        return None
    
    center_point = (
        float(center_data['X (m)'].iloc[0]),
        float(center_data['Y (m)'].iloc[0]),
        float(center_data['Z (m)'].iloc[0])
    )
    
    # Find points in the patch region
    region_df = find_points_in_circle_enhanced(df, center_point, radius, patch_number)
    
    if len(region_df) == 0:
        return None
    
    # Calculate patch statistics
    stats = calculate_patch_statistics(region_df)
    
    # Get time point from filename using robust extraction
    timestep = extract_timestep_from_filename(file_path)
    
    # Determine time unit based on filename format
    if 'e+' in file_path.stem or 'e-' in file_path.stem or '.' in file_path.stem:
        # Scientific notation or decimal format - likely in seconds
        time_sec = timestep
    else:
        # Integer format - likely in milliseconds
        time_sec = timestep * 0.001  # Convert to seconds
    
    # Return data in same format as single point tracking
    return {
        'Time (s)': time_sec,
        'X (m)': center_point[0],  # Use center point coordinates
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

def track_region_in_file(file_path: Path, point_pairs: list) -> dict:
    """Track all points in a region for a single time step file.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Check if we need to add patch numbers (for raw CSV files)
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
        
        # Add Patch Number column
        df['Patch Number'] = patch_numbers
    
    # Create boolean mask for all points in the region
    mask = pd.DataFrame({'Patch Number': df['Patch Number'], 'Face Index': df['Face Index']})
    mask['in_region'] = mask.apply(lambda x: (x['Patch Number'], x['Face Index']) in point_pairs, axis=1)
    region_points = df[mask['in_region']]
    
    if len(region_points) == 0:
        return None
    
    # Calculate average pressure for patch 35
    patch_avg_pressure = calculate_patch_average_pressure(file_path)
    if patch_avg_pressure is None:
        print(f"Warning: No data found for patch 35 in {file_path.name}")
        patch_avg_pressure = 0.0
    
    # Get time point from filename using robust extraction
    timestep = extract_timestep_from_filename(file_path)
    
    # Determine time unit based on filename format
    if 'e+' in file_path.stem or 'e-' in file_path.stem or '.' in file_path.stem:
        # Scientific notation or decimal format - likely in seconds
        time_sec = timestep
        time_point = int(timestep * 1000)  # Convert to milliseconds
    else:
        # Integer format - likely in milliseconds
        time_sec = timestep * 0.001  # Convert to seconds
        time_point = int(timestep)  # Already in milliseconds
    
    # Calculate average velocity vector
    avg_velocity_vector = np.array([
        float(region_points['Velocity[i] (m/s)'].mean()),
        float(region_points['Velocity[j] (m/s)'].mean()),
        float(region_points['Velocity[k] (m/s)'].mean())
    ])
    
    # Calculate average velocity magnitude
    avg_velocity = float(np.linalg.norm(avg_velocity_vector))
    
    # Calculate average area vector and normalize to get normal vector
    avg_area_vector = np.array([
        float(region_points['Area[i] (m^2)'].mean()),
        float(region_points['Area[j] (m^2)'].mean()),
        float(region_points['Area[k] (m^2)'].mean())
    ])
    avg_normal_vector = avg_area_vector / np.linalg.norm(avg_area_vector)
    
    # Calculate dot product between average velocity and normal vectors
    vdotn = float(np.dot(avg_velocity_vector, avg_normal_vector))
    
    # Calculate signed velocity using dot product sign
    signed_velocity = avg_velocity * np.sign(vdotn)
    
    # Calculate centroid and average pressure
    avg_pressure = float(region_points['Total Pressure (Pa)'].mean())
    centroid = (
        float(region_points['X (m)'].mean()),
        float(region_points['Y (m)'].mean()),
        float(region_points['Z (m)'].mean())
    )
    
    return {
        'time': time_sec,
        'time_point': time_point,
        'x': centroid[0],
        'y': centroid[1],
        'z': centroid[2],
        'pressure': avg_pressure,
        'adjusted_pressure': avg_pressure - patch_avg_pressure,
        'velocity': avg_velocity,
        'velocity_i': region_points['Velocity[i] (m/s)'].mean(),
        'velocity_j': region_points['Velocity[j] (m/s)'].mean(),
        'velocity_k': region_points['Velocity[k] (m/s)'].mean(),
        'area_i': region_points['Area[i] (m^2)'].mean(),
        'area_j': region_points['Area[j] (m^2)'].mean(),
        'area_k': region_points['Area[k] (m^2)'].mean(),
        'vdotn': vdotn,
        'signed_velocity': signed_velocity,
        'patch35_avg_pressure': patch_avg_pressure,
        'num_points_in_region': len(region_points)
    }

def calculate_derived_quantities(trajectory_data):
    """Calculate velocity and acceleration from position data."""
    times = np.array([p['time'] for p in trajectory_data])  # Now in seconds
    positions = np.array([[p['x'], p['y'], p['z']] for p in trajectory_data])
    pressures = np.array([p['pressure'] for p in trajectory_data])
    velocities = np.array([p['velocity'] for p in trajectory_data])
    vdotn = np.array([p['vdotn'] for p in trajectory_data])
    
    # Calculate signed velocity (velocity magnitude with direction based on VdotN)
    signed_velocities = velocities * np.sign(vdotn)
    
    # Calculate accelerations using velocity magnitude - now dt is constant at 0.001 seconds
    dt = 0.001  # Fixed time step in seconds
    dv = np.diff(velocities)  # Using original velocity magnitude for acceleration
    accelerations = dv / dt  # No need to convert dt since it's already in seconds
    
    return signed_velocities, accelerations, vdotn, pressures

def save_trajectory_data(trajectory_data, subject_name, patch_number, face_index, description, is_region=False):
    """Save trajectory data to CSV file in a dedicated folder."""
    # Create tracked_points directory if it doesn't exist
    results_dir = Path(f'{subject_name}_results')
    output_dir = results_dir / 'tracked_points'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with meaningful information
    suffix = '_r2mm' if is_region else ''
    output_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}{suffix}.csv"
    
    # Sort trajectory data by time
    # Handle both 'time' and 'Time (s)' keys for compatibility
    if trajectory_data and 'Time (s)' in trajectory_data[0]:
        trajectory_data = sorted(trajectory_data, key=lambda x: x['Time (s)'])
    else:
        trajectory_data = sorted(trajectory_data, key=lambda x: x['time'])
    
    # Remove duplicates based on time points
    seen_times = set()
    unique_trajectory_data = []
    
    for data_point in trajectory_data:
        time_point = data_point['time_point']
        if time_point not in seen_times:
            seen_times.add(time_point)
            unique_trajectory_data.append(data_point)
        else:
            print(f"Warning: Duplicate time point {time_point} found and removed")
    
    if len(trajectory_data) != len(unique_trajectory_data):
        print(f"Removed {len(trajectory_data) - len(unique_trajectory_data)} duplicate time points")
    
    df = pd.DataFrame({
        'Time Point': [p['time_point'] for p in unique_trajectory_data],
        'Time (s)': [p['time'] for p in unique_trajectory_data],
        'X (m)': [p['x'] for p in unique_trajectory_data],
        'Y (m)': [p['y'] for p in unique_trajectory_data],
        'Z (m)': [p['z'] for p in unique_trajectory_data],
        'Total Pressure (Pa)': [p['pressure'] for p in unique_trajectory_data],
        'Patch35 Avg Pressure (Pa)': [p['patch35_avg_pressure'] for p in unique_trajectory_data],
        'Adjusted Pressure (Pa)': [p['adjusted_pressure'] for p in unique_trajectory_data],
        'Velocity: Magnitude (m/s)': [p['velocity'] for p in unique_trajectory_data],
        'Velocity[i] (m/s)': [p.get('velocity_i', 0) for p in unique_trajectory_data],
        'Velocity[j] (m/s)': [p.get('velocity_j', 0) for p in unique_trajectory_data],
        'Velocity[k] (m/s)': [p.get('velocity_k', 0) for p in unique_trajectory_data],
        'Area[i] (m^2)': [p.get('area_i', 0) for p in unique_trajectory_data],
        'Area[j] (m^2)': [p.get('area_j', 0) for p in unique_trajectory_data],
        'Area[k] (m^2)': [p.get('area_k', 0) for p in unique_trajectory_data],
        'VdotN': [p['vdotn'] for p in unique_trajectory_data],
        'Signed Velocity (m/s)': [p['signed_velocity'] for p in unique_trajectory_data]
    })
    
    if is_region:
        df['Num Points in Region'] = [p.get('num_points_in_region', 0) for p in unique_trajectory_data]
        # Add patch-specific statistics if available
        if 'pressure_std' in unique_trajectory_data[0]:
            df['Pressure Std (Pa)'] = [p.get('pressure_std', 0) for p in unique_trajectory_data]
        if 'velocity_std' in unique_trajectory_data[0]:
            df['Velocity Std (m/s)'] = [p.get('velocity_std', 0) for p in unique_trajectory_data]
        if 'vdotn_std' in unique_trajectory_data[0]:
            df['VdotN Std'] = [p.get('vdotn_std', 0) for p in unique_trajectory_data]
    
    df.to_csv(output_file, index=False)
    print(f"Saved trajectory data to {output_file}")
    return output_file

def compute_combination_mean(subject_name, combination_config, tracking_locations):
    """
    Compute mean tracking results from a combination of individual tracking points.
    
    Args:
        subject_name: Name of the subject
        combination_config: Dictionary with combination configuration
        tracking_locations: List of individual tracking locations
        
    Returns:
        Path to the saved combination CSV file, or None if failed
    """
    combination_name = combination_config['name']
    combination_description = combination_config['description']
    point_indices = combination_config['point_indices']
    
    print(f"\nComputing combination: {combination_name}")
    print(f"Description: {combination_description}")
    print(f"Combining points with indices: {point_indices}")
    
    # Load data for each point in the combination
    dataframes = []
    point_info = []
    
    results_dir = Path(f'{subject_name}_results')
    output_dir = results_dir / 'tracked_points'
    
    for idx in point_indices:
        if idx >= len(tracking_locations):
            print(f"Warning: Point index {idx} is out of range (max: {len(tracking_locations)-1})")
            continue
            
        location = tracking_locations[idx]
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        
        # Load the individual point data
        data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
        
        if not data_file.exists():
            print(f"Warning: Data file not found for point {idx}: {description}")
            continue
            
        df = pd.read_csv(data_file)
        dataframes.append(df)
        point_info.append(f"P{patch_number}F{face_index}")
        print(f"  - Loaded: {description} (Patch {patch_number}, Face {face_index})")
    
    if len(dataframes) == 0:
        print(f"Error: No valid data files found for combination {combination_name}")
        return None
    
    # Ensure all dataframes have the same time points
    common_times = dataframes[0]['Time (s)'].values
    for i, df in enumerate(dataframes[1:], 1):
        if not np.allclose(df['Time (s)'].values, common_times, rtol=1e-6):
            print(f"Warning: Time points don't match for point {point_indices[i]}. Interpolating...")
            # In practice, this shouldn't happen if all data comes from the same simulation
    
    # Compute means for each column
    print(f"Computing means across {len(dataframes)} points...")
    
    # Start with time columns (same for all)
    mean_data = {
        'Time Point': dataframes[0]['Time Point'].values,
        'Time (s)': dataframes[0]['Time (s)'].values,
    }
    
    # Columns to average
    columns_to_average = [
        'X (m)', 'Y (m)', 'Z (m)',
        'Total Pressure (Pa)', 'Patch35 Avg Pressure (Pa)', 'Adjusted Pressure (Pa)',
        'Velocity: Magnitude (m/s)', 'Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)',
        'Area[i] (m^2)', 'Area[j] (m^2)', 'Area[k] (m^2)',
        'VdotN', 'Signed Velocity (m/s)'
    ]
    
    # Compute means
    for col in columns_to_average:
        if col in dataframes[0].columns:
            values = np.array([df[col].values for df in dataframes])
            mean_data[col] = np.mean(values, axis=0)
        else:
            print(f"Warning: Column {col} not found in data")
    
    # Create combined DataFrame
    combined_df = pd.DataFrame(mean_data)
    
    # Save the combination results
    combination_filename = f"{subject_name}_combination_{combination_name}.csv"
    combination_file = output_dir / combination_filename
    combined_df.to_csv(combination_file, index=False)
    
    print(f"Saved combination results to {combination_file}")
    print(f"Combined points: {' + '.join(point_info)}")
    
    return combination_file

def process_all_combinations(subject_name, tracking_locations):
    """
    Process all combinations defined in the tracking_locations.json file.
    
    Args:
        subject_name: Name of the subject
        tracking_locations: Loaded tracking locations configuration
        
    Returns:
        List of paths to saved combination CSV files
    """
    # Check if combinations are defined
    if 'combinations' not in tracking_locations or not tracking_locations['combinations']:
        print("\nNo combinations defined in tracking_locations.json")
        return []
    
    combinations = tracking_locations['combinations']
    print(f"\nProcessing {len(combinations)} combinations...")
    
    combination_files = []
    individual_locations = tracking_locations['locations']
    
    for combination in combinations:
        combination_file = compute_combination_mean(subject_name, combination, individual_locations)
        if combination_file:
            combination_files.append(combination_file)
    
    return combination_files

def create_airway_surface_velocity_plot(df, subject_name, description, patch_number, face_index, pdf, smoothing_window=20):
    """Create a plot of airway surface velocity over time."""
    # Compute resultant velocity
    velocity = np.sqrt(
        df['Velocity[i] (m/s)']**2 + 
        df['Velocity[j] (m/s)']**2 + 
        df['Velocity[k] (m/s)']**2
    )
    
    # Convert to mm/s for display
    velocity_mm = velocity * 1000
    vdotn_mm = df['VdotN'] * 1000
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Get flow profile for breathing cycle markers
    try:
        base_subject = extract_base_subject(subject_name)
        flow_profile = pd.read_csv(f'{base_subject}FlowProfile.csv')
        
        # Apply smoothing in memory
        def apply_smoothing(data, window_size=smoothing_window):
            if len(data) < window_size:
                return data.copy()
            smoothed = data.copy()
            half_window = window_size // 2
            for i in range(half_window, len(data) - half_window):
                smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
            return smoothed
        
        flow_times = flow_profile['time (s)'].values
        flow_rates = apply_smoothing(flow_profile['Massflowrate (kg/s)'].values)
        
        # Find zero crossings in flow profile for breathing cycle transitions
        zero_crossings = np.where(np.diff(np.signbit(flow_rates)))[0]
        cycle_markers = []
        
        if len(zero_crossings) > 0:
            for i, idx in enumerate(zero_crossings):
                t = flow_times[idx]
                if i == 0:
                    label = f"Start Inhale (t={t:.3f}s)"
                elif i == 1:
                    label = f"Inhale→Exhale (t={t:.3f}s)"
                else:
                    label = f"Marker {i+1} (t={t:.3f}s)"
                cycle_markers.append({'time': t, 'label': label})
    except Exception as e:
        print(f"Warning: Could not load flow profile: {e}")
        cycle_markers = []
    
    # Plot 1: Total velocity
    ax1.plot(df['Time (s)'], velocity_mm, 'r-', linewidth=2, label='|v⃗|')
    ax1.plot(df['Time (s)'], vdotn_mm, 'b-', linewidth=2, label='v⃗·n⃗')
    
    # Add breathing cycle markers if available
    for marker in cycle_markers:
        ax1.axvline(x=marker['time'], color='k', linestyle=':', alpha=0.7)
        ax1.text(marker['time'], ax1.get_ylim()[1]*0.9, marker['label'], 
                rotation=90, verticalalignment='top', fontsize=10)
    
    ax1.set_ylabel('Velocity (mm/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Airway Surface Velocity at {description}\nPatch: {patch_number}, Face: {face_index}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: Individual velocity components
    ax2.plot(df['Time (s)'], df['Velocity[i] (m/s)'] * 1000, 'r-', linewidth=1.5, label='v_x')
    ax2.plot(df['Time (s)'], df['Velocity[j] (m/s)'] * 1000, 'g-', linewidth=1.5, label='v_y')
    ax2.plot(df['Time (s)'], df['Velocity[k] (m/s)'] * 1000, 'b-', linewidth=1.5, label='v_z')
    
    # Add breathing cycle markers if available
    for marker in cycle_markers:
        ax2.axvline(x=marker['time'], color='k', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|v⃗| (mm/s)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to PDF
    if pdf is not None:
        pdf.savefig(fig)
        plt.close()
    else:
        plt.show()

def create_local_coordinate_system(normal_vector):
    """Create local coordinate system based on surface normal and z-direction.
    
    Args:
        normal_vector: Unit vector normal to surface
        
    Returns:
        Tuple of three orthonormal vectors:
        - normal_vector: Surface normal direction
        - z_aligned_vector: Direction in surface plane closest to positive z
        - third_vector: Direction orthogonal to both above vectors
    """
    # Ensure normal vector is normalized
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Create z-direction vector
    z_direction = np.array([0, 0, 1])
    
    # Project z onto plane perpendicular to normal
    z_proj = z_direction - np.dot(z_direction, normal_vector) * normal_vector
    z_proj_mag = np.linalg.norm(z_proj)
    
    # If z_proj is too small (normal almost parallel to z), use x or y instead
    if z_proj_mag < 1e-6:
        if abs(normal_vector[0]) < 0.9:  # If normal not close to x-axis
            temp_vec = np.array([1, 0, 0])
        else:  # Use y-axis
            temp_vec = np.array([0, 1, 0])
        z_proj = temp_vec - np.dot(temp_vec, normal_vector) * normal_vector
        z_proj_mag = np.linalg.norm(z_proj)
    
    # Normalize z-aligned vector
    z_aligned_vector = z_proj / z_proj_mag
    
    # Get third vector using cross product
    third_vector = np.cross(normal_vector, z_aligned_vector)
    
    return normal_vector, z_aligned_vector, third_vector

def decompose_velocity(velocity_vector, normal_vector):
    """Decompose velocity into local coordinate system components.
    
    Args:
        velocity_vector: The velocity vector to decompose
        normal_vector: Surface normal unit vector
        
    Returns:
        Dictionary containing:
        - v_normal: Component in surface normal direction (v⋅n)
        - v_z_aligned: Component in surface plane closest to z-direction
        - v_third: Component in third orthogonal direction
    """
    # Create local coordinate system
    normal, z_aligned, third = create_local_coordinate_system(normal_vector)
    
    # Calculate components by projecting onto each direction
    v_normal = np.dot(velocity_vector, normal)
    v_z_aligned = np.dot(velocity_vector, z_aligned)
    v_third = np.dot(velocity_vector, third)
    
    return {
        'v_normal': v_normal,
        'v_z_aligned': v_z_aligned,
        'v_third': v_third
    }

def compute_windowed_cross_correlation(signal1, signal2, times, window_size=0.2):
    """Compute windowed cross-correlation between two signals.
    
    Args:
        signal1: First signal array
        signal2: Second signal array
        times: Time points array
        window_size: Window size in seconds (default: 0.2s)
        
    Returns:
        Dictionary containing:
        - correlation_values: Array of correlation coefficients
        - window_centers: Array of window center times
        - lags: Array of lag times
        - max_correlations: Maximum correlation for each window
        - optimal_lags: Lag time at maximum correlation for each window
    """
    n_points = len(times)
    
    # Check if we have enough data for windowed correlation
    if n_points < 5:
        print(f"Warning: Only {n_points} time points available. Skipping windowed correlation analysis.")
        # Return minimal data structure
        return {
            'correlation_values': np.array([[0.0]]),
            'window_centers': np.array([times[n_points//2]]),
            'lags': np.array([0.0]),
            'max_correlations': np.array([0.0]),
            'optimal_lags': np.array([0.0])
        }
    
    # Calculate time step
    dt = times[1] - times[0] if n_points > 1 else 0.001
    
    # Adapt window size based on available data
    max_possible_window_size = (n_points - 1) * dt
    if window_size > max_possible_window_size:
        window_size = max_possible_window_size * 0.8  # Use 80% of available data
        print(f"Warning: Requested window size too large. Reduced to {window_size:.3f}s")
    
    # Convert window size from seconds to number of samples
    window_samples = int(window_size / dt)
    
    # Ensure minimum window size
    window_samples = max(window_samples, 3)  # At least 3 samples
    
    # Ensure we don't exceed available data
    if window_samples >= n_points:
        window_samples = n_points - 1
        print(f"Warning: Window size adjusted to {window_samples} samples due to limited data")
    
    # Ensure window size is odd
    if window_samples % 2 == 0:
        window_samples += 1
    
    half_window = window_samples // 2
    
    # Initialize arrays for results
    n_windows = len(times) - window_samples + 1
    
    # If we still don't have enough windows, use a simpler approach
    if n_windows <= 0:
        print(f"Warning: Insufficient data for windowed analysis. Using simple correlation.")
        # Calculate simple correlation for the entire signal
        try:
            simple_corr = np.corrcoef(signal1, signal2)[0, 1]
            if np.isnan(simple_corr):
                simple_corr = 0.0
        except:
            simple_corr = 0.0
            
        return {
            'correlation_values': np.array([[simple_corr]]),
            'window_centers': np.array([times[n_points//2]]),
            'lags': np.array([0.0]),
            'max_correlations': np.array([simple_corr]),
            'optimal_lags': np.array([0.0])
        }
    
    # Limit maximum lag to prevent issues with small datasets
    max_lag = min(window_samples // 4, 10, n_points // 4)  # Conservative limits
    if max_lag < 1:
        max_lag = 1
        
    lags = np.arange(-max_lag, max_lag + 1) * dt
    correlation_values = np.zeros((n_windows, len(lags)))
    
    # Calculate window centers
    if n_windows == 1:
        window_centers = np.array([times[n_points//2]])
    else:
        window_centers = times[half_window:half_window + n_windows]
    
    # Compute correlation for each window
    for i in range(n_windows):
        # Extract window data
        start_idx = i
        end_idx = i + window_samples
        
        s1_window = signal1[start_idx:end_idx]
        s2_window = signal2[start_idx:end_idx]
        
        # Check for valid data
        if len(s1_window) < 3 or len(s2_window) < 3:
            continue
            
        # Normalize signals within window
        s1_std = np.std(s1_window)
        s2_std = np.std(s2_window)
        
        if s1_std < 1e-10 or s2_std < 1e-10:
            # No variation in the signal
            correlation_values[i, :] = 0.0
            continue
            
        s1_norm = (s1_window - np.mean(s1_window)) / s1_std
        s2_norm = (s2_window - np.mean(s2_window)) / s2_std
        
        # For each lag, compute normalized correlation coefficient
        for j, lag in enumerate(range(-max_lag, max_lag + 1)):
            try:
                if lag < 0:
                    # Shift s1 left relative to s2
                    if len(s1_norm[-lag:]) > 0 and len(s2_norm[:lag]) > 0:
                        corr = np.corrcoef(s1_norm[-lag:], s2_norm[:lag])[0, 1]
                    else:
                        corr = 0.0
                elif lag > 0:
                    # Shift s1 right relative to s2
                    if len(s1_norm[:-lag]) > 0 and len(s2_norm[lag:]) > 0:
                        corr = np.corrcoef(s1_norm[:-lag], s2_norm[lag:])[0, 1]
                    else:
                        corr = 0.0
                else:
                    # No shift
                    corr = np.corrcoef(s1_norm, s2_norm)[0, 1]
                
                correlation_values[i, j] = corr if not np.isnan(corr) else 0.0
                
            except (ValueError, IndexError):
                correlation_values[i, j] = 0.0
    
    # Find maximum correlation and optimal lag for each window
    max_correlations = np.max(correlation_values, axis=1)
    optimal_lags = lags[np.argmax(correlation_values, axis=1)]
    
    print(f"Windowed correlation completed: {n_windows} windows, {len(lags)} lags, window size: {window_samples} samples")
    
    return {
        'correlation_values': correlation_values,
        'window_centers': window_centers,
        'lags': lags,
        'max_correlations': max_correlations,
        'optimal_lags': optimal_lags
    }

def create_correlation_analysis_plot(df, subject_name, description, patch_number, face_index, pdf_50ms, pdf_100ms):
    """Create plots showing windowed cross-correlation analysis between pressure and motion."""
    
    # Check if we have enough data for correlation analysis
    n_points = len(df)
    if n_points < 5:
        print(f"Warning: Only {n_points} time points available for {description}. Skipping correlation analysis.")
        return
    
    # For small datasets, provide a warning and use simplified analysis
    if n_points < 10:
        print(f"Warning: Limited data ({n_points} points) for {description}. Correlation analysis may be less reliable.")
    
    # Create plots for both window sizes
    for window_size in [0.05, 0.1]:  # 50ms and 100ms
        # Create figure with 2x3 layout with more space for titles and labels
        fig = plt.figure(figsize=(24, 16))  # Increased figure size
        
        # Create gridspec with more space for titles and between subplots
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4,
                            top=0.85,  # More space for main title
                            bottom=0.08,  # More space for x-labels
                            left=0.08,  # More space for y-labels
                            right=0.92)  # More space for colorbars
        
        # Get time points and signals
        times = df['Time (s)'].values
        pressure = df['Total Pressure (Pa)'].values
        
        # Get velocity components and calculate metrics as before
        velocity_vectors = np.column_stack((
            df['Velocity[i] (m/s)'],
            df['Velocity[j] (m/s)'],
            df['Velocity[k] (m/s)']
        ))
        velocity_mag = np.linalg.norm(velocity_vectors, axis=1)
        
        area_vectors = np.column_stack((
            df['Area[i] (m^2)'],
            df['Area[j] (m^2)'],
            df['Area[k] (m^2)']
        ))
        normal_vectors = area_vectors / np.linalg.norm(area_vectors, axis=1)[:, np.newaxis]
        
        v_components = []
        for vel, norm in zip(velocity_vectors, normal_vectors):
            v_components.append(decompose_velocity(vel, norm))
        
        v_normal = np.array([comp['v_normal'] for comp in v_components])
        v_z_aligned = np.array([comp['v_z_aligned'] for comp in v_components])
        v_third = np.array([comp['v_third'] for comp in v_components])
        
        # Calculate acceleration
        dt = times[1] - times[0]
        dvdotn = np.diff(v_normal)
        adotn = np.append(dvdotn / dt, dvdotn[-1] / dt)
        
        # Define metrics and their styles
        metrics = {
            'Normal Acceleration': adotn,
            'Normal Velocity': v_normal,
            'Z-Aligned Velocity': v_z_aligned,
            'Third Component': v_third,
            'Total Velocity': velocity_mag
        }
        
        metric_styles = {
            'Normal Acceleration': {'color': '#9467bd', 'linestyle': '-'},  # purple
            'Normal Velocity': {'color': '#ff7f0e', 'linestyle': '--'},  # orange
            'Z-Aligned Velocity': {'color': '#2ca02c', 'linestyle': ':'},  # green
            'Third Component': {'color': '#d62728', 'linestyle': '-.'},  # red
            'Total Velocity': {'color': '#1f77b4', 'linestyle': '-'}  # blue
        }
        
        # Panel 1: Raw data overlay (top-left)
        ax_raw = fig.add_subplot(gs[0, 0])
        
        def normalize_signal(signal):
            return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
        
        # Plot normalized signals
        ax_raw.plot(times, normalize_signal(pressure), color='k', linestyle='-', 
                    label='Pressure', linewidth=2.5)
        
        for name, data in metrics.items():
            style = metric_styles[name]
            ax_raw.plot(times, normalize_signal(data),
                       label=name,
                       color=style['color'],
                       linestyle=style['linestyle'],
                       linewidth=1.5)
        
        ax_raw.set_xlabel('Time (s)', fontsize=12, labelpad=10)
        ax_raw.set_ylabel('Normalized Values', fontsize=12, labelpad=10)
        ax_raw.set_title(f'Raw Data Overlay (Normalized)\n{description} - Patch {patch_number}, Face {face_index}', 
                        fontsize=14, pad=20)
        ax_raw.grid(True, alpha=0.3)
        ax_raw.set_ylim(-1.2, 1.2)
        
        # Adjust legend position to prevent overlap
        legend = ax_raw.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', 
                              fontsize=10, borderaxespad=0.)
        
        # Create axes for correlation plots
        axes = [
            fig.add_subplot(gs[0, 1]),  # top middle
            fig.add_subplot(gs[0, 2]),  # top right
            fig.add_subplot(gs[1, 0]),  # bottom left
            fig.add_subplot(gs[1, 1]),  # bottom middle
            fig.add_subplot(gs[1, 2])   # bottom right
        ]
        
        # Plot correlations for each metric
        for (name, data), ax in zip(metrics.items(), axes):
            style = metric_styles[name]
            corr = compute_windowed_cross_correlation(pressure, data, times, window_size=window_size)
            ax.plot(corr['window_centers'], corr['max_correlations'],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    alpha=0.8,
                    linewidth=2,
                    label=f'Pressure-{name}')
            
            ax.set_xlabel('Time (s)', fontsize=12, labelpad=10)
            ax.set_ylabel('Correlation', fontsize=12, labelpad=10)
            ax.set_title(f'Pressure vs {name}\n{description} - Patch {patch_number}, Face {face_index}', 
                        fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.1, 1.1)
            
            # Add reference lines
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axhline(y=0.7, color='k', linestyle=':', alpha=0.2)
            ax.axhline(y=-0.7, color='k', linestyle=':', alpha=0.2)
            
            # Add legend with adjusted position
            ax.legend(loc='upper right', fontsize=10, borderaxespad=2)
        
        # Add inhale-exhale transition lines to all plots
        base_subject = extract_base_subject(subject_name)
        flow_profile = pd.read_csv(f'{base_subject}FlowProfile.csv')
        # Apply smoothing in memory
        def apply_smoothing(data, window_size=20):
            return data.rolling(window=window_size, center=True, min_periods=1).mean()
        
        flow_profile_smoothed = flow_profile.copy()
        flow_profile_smoothed['Massflowrate (kg/s)'] = apply_smoothing(flow_profile['Massflowrate (kg/s)'])
        
        zero_crossings = np.where(np.diff(np.signbit(flow_profile_smoothed['Massflowrate (kg/s)'])))[0]
        if len(zero_crossings) >= 2:
            inhale_exhale = flow_profile_smoothed['time (s)'].iloc[zero_crossings[1]]
            for ax in [ax_raw] + axes:
                ax.axvline(x=inhale_exhale, color='k', linestyle=':', alpha=0.5,
                          label=f'Inhale→Exhale')
        
        # Add overall title with more padding
        window_ms = int(window_size * 1000)
        fig.suptitle(f'Pressure-Motion Analysis ({window_ms}ms window)\n{description} - Patch {patch_number}, Face {face_index}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save to appropriate PDF with tight layout and extra padding
        if window_size == 0.05:
            pdf_50ms.savefig(fig, bbox_inches='tight', bbox_extra_artists=[legend], pad_inches=0.2)
        else:
            pdf_100ms.savefig(fig, bbox_inches='tight', bbox_extra_artists=[legend], pad_inches=0.2)
        
        # Also save as PNG
        results_dir = Path(f'{subject_name}_results')
        figures_dir = results_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        png_filename = figures_dir / f"{subject_name}_correlation_analysis_{description.lower().replace(' ', '_')}_patch{patch_number}_face{face_index}_{window_ms}ms.png"
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        
        plt.close()

def create_dpdt_vs_dadt_plot(df, subject_name, description, patch_number, face_index, pdf):
    """Create a plot showing pressure rate of change vs acceleration rate of change."""
    # Get data
    times = df['Time (s)'].values
    vdotn = df['VdotN'].values
    pressure = df['Total Pressure (Pa)'].values
    
    # Apply smoothing with Savitzky-Golay filter - handle small datasets
    data_length = len(times)
    
    if data_length <= 5:
        # Too few points, no smoothing
        print(f"Too few data points ({data_length}) for smoothing, using original data")
        vdotn_smooth = vdotn.copy()
        pressure_smooth = pressure.copy()
    else:
        # Calculate appropriate window size
        max_window = data_length if data_length % 2 == 1 else data_length - 1
        min_window = 5  # Minimum for polyorder=3
        
        # Try to use 20% of data length, but respect constraints
        desired_window = max(min_window, int(0.2 * data_length))
        window = min(desired_window, max_window)
        
        # Ensure odd
        if window % 2 == 0:
            window -= 1
        
        try:
            vdotn_smooth = savgol_filter(vdotn, window, 3, deriv=0)
            pressure_smooth = savgol_filter(pressure, window, 3, deriv=0)
        except ValueError as e:
            print(f"Savgol filter failed, falling back to moving average: {e}")
            # Fallback to simple moving average
            def simple_smooth(data, window):
                return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
            
            vdotn_smooth = simple_smooth(vdotn, min(5, data_length))
            pressure_smooth = simple_smooth(pressure, min(5, data_length))
    
    # Calculate derivatives using actual time differences
    dt_array = np.diff(times)
    
    # Check for zero time differences and fix them
    zero_dt_mask = dt_array == 0
    if np.any(zero_dt_mask):
        print(f"Warning: Found {np.sum(zero_dt_mask)} zero time differences. Fixing with interpolation.")
        # Replace zero time differences with the mean of non-zero values
        non_zero_dt = dt_array[~zero_dt_mask]
        if len(non_zero_dt) > 0:
            mean_dt = np.mean(non_zero_dt)
            dt_array[zero_dt_mask] = mean_dt
        else:
            dt_array = np.full_like(dt_array, 0.001)  # Default 1ms timestep
    
    # Acceleration = d(vdotn)/dt - compute from smoothed data
    dvdotn = np.diff(vdotn_smooth)
    adotn_smooth = np.append(dvdotn / dt_array, dvdotn[-1] / dt_array[-1])
    
    # Second derivative of velocity = d²(vdotn)/dt² = d(adotn)/dt
    dadotn = np.diff(adotn_smooth)
    dadt = np.append(dadotn / dt_array, dadotn[-1] / dt_array[-1])
    
    # Rate of change of pressure = dp/dt
    dpressure = np.diff(pressure_smooth)
    dpdt = np.append(dpressure / dt_array, dpressure[-1] / dt_array[-1])
    
    # Print statistics
    print(f"\nStatistics for {description}:")
    print("Acceleration (AdotN):")
    # Convert to mm/s² for display
    adotn_smooth_mm = adotn_smooth * 1000
    print(f"  Mean: {np.mean(adotn_smooth_mm):.2f} mm/s²")
    print(f"  Std: {np.std(adotn_smooth_mm):.2f} mm/s²")
    print(f"  Min: {np.min(adotn_smooth_mm):.2f} mm/s²")
    print(f"  Max: {np.max(adotn_smooth_mm):.2f} mm/s²")
    
    print("Rate of change of acceleration (dAdotN/dt):")
    # Convert to mm/s³ for display
    dadt_mm = dadt * 1000
    print(f"  Mean: {np.mean(dadt_mm):.2f} mm/s³")
    print(f"  Std: {np.std(dadt_mm):.2f} mm/s³")
    print(f"  Min: {np.min(dadt_mm):.2f} mm/s³")
    print(f"  Max: {np.max(dadt_mm):.2f} mm/s³")
    
    print("Rate of change of pressure (dP/dt):")
    print(f"  Mean: {np.mean(dpdt):.2f} Pa/s")
    print(f"  Std: {np.std(dpdt):.2f} Pa/s")
    print(f"  Min: {np.min(dpdt):.2f} Pa/s")
    print(f"  Max: {np.max(dpdt):.2f} Pa/s")
    
    # Create the actual plot
    fig = plt.figure(figsize=(12, 8))
    
    # Convert units for plotting
    dpdt_converted = dpdt  # Keep Pa/s
    dadt_converted = dadt * 1000  # Convert to mm/s³
    
    # Create scatter plot
    plt.scatter(dadt_converted, dpdt_converted, alpha=0.6, s=20, c=times, cmap='viridis')
    
    # Add colorbar for time
    cbar = plt.colorbar()
    cbar.set_label('Time (s)', fontsize=12)
    
    # Add labels and title
    plt.xlabel('Rate of change of acceleration (dA/dt) [mm/s³]', fontsize=12, fontweight='bold')
    plt.ylabel('Rate of change of pressure (dP/dt) [Pa/s]', fontsize=12, fontweight='bold')
    plt.title(f'Rate of Change Analysis: dP/dt vs dA/dt\n{description} - Patch {patch_number}, Face {face_index}', 
              fontsize=14, fontweight='bold')
    
    # Add zero reference lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_metrics_vs_time_plot(df, subject_name, description, patch_number, face_index, pdf):
    """Create a plot showing various metrics over time."""
    # Common settings
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Get data
    times = df['Time (s)'].values
    vdotn = df['VdotN'].values
    pressure = df['Total Pressure (Pa)'].values
    velocity = df['Velocity: Magnitude (m/s)'].values
    
    # Apply smoothing to reduce noise - handle small datasets
    data_length = len(times)
    
    if data_length <= 5:
        # Too few points, no smoothing
        print(f"Too few data points ({data_length}) for smoothing in metrics plot, using original data")
        vdotn_smooth = vdotn.copy()
        pressure_smooth = pressure.copy()
    else:
        # Calculate appropriate window size
        max_window = data_length if data_length % 2 == 1 else data_length - 1
        min_window = 5  # Minimum for polyorder=3
        
        # Try to use 20% of data length, but respect constraints
        desired_window = max(min_window, int(0.2 * data_length))
        window = min(desired_window, max_window)
        
        # Ensure odd
        if window % 2 == 0:
            window -= 1
        
        try:
            vdotn_smooth = savgol_filter(vdotn, window, 3)
            pressure_smooth = savgol_filter(pressure, window, 3)
        except ValueError as e:
            print(f"Savgol filter failed in metrics plot, falling back to moving average: {e}")
            # Fallback to simple moving average
            def simple_smooth(data, window):
                return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
            
            vdotn_smooth = simple_smooth(vdotn, min(5, data_length))
            pressure_smooth = simple_smooth(pressure, min(5, data_length))
    
    # Calculate acceleration from vdotn
    dt_array = np.diff(times)
    
    # Check for zero time differences and fix them
    zero_dt_mask = dt_array == 0
    if np.any(zero_dt_mask):
        print(f"Warning: Found {np.sum(zero_dt_mask)} zero time differences in time series. Fixing with interpolation.")
        # Replace zero time differences with the mean of non-zero values
        non_zero_dt = dt_array[~zero_dt_mask]
        if len(non_zero_dt) > 0:
            mean_dt = np.mean(non_zero_dt)
            dt_array[zero_dt_mask] = mean_dt
        else:
            dt_array = np.full_like(dt_array, 0.001)  # Default 1ms timestep
    
    dvdotn = np.diff(vdotn)
    adotn = np.append(dvdotn / dt_array, dvdotn[-1] / dt_array[-1])
    
    # Apply same smoothing logic to acceleration
    if data_length <= 5:
        adotn_smooth = adotn.copy()
    else:
        try:
            adotn_smooth = savgol_filter(adotn, window, 3)
        except ValueError:
            # Fallback to simple moving average
            def simple_smooth(data, window):
                return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
            adotn_smooth = simple_smooth(adotn, min(5, data_length))
    
    # Find zero crossings in velocity (where vdotn changes sign)
    vel_zero_crossings = np.where(np.diff(np.signbit(vdotn)))[0]
    vel_crossing_times = []
    for idx in vel_zero_crossings:
        t = times[idx:idx+2].mean()
        vel_crossing_times.append(t)
    
    if vel_crossing_times:
        print(f"\nVelocity sign changes for {description} at times (seconds):")
        for i, t in enumerate(vel_crossing_times):
            print(f"  Crossing {i+1}: {t:.3f}s")
    
    # Find zero crossings in acceleration (where adotn_smooth changes sign)
    acc_zero_crossings = np.where(np.diff(np.signbit(adotn_smooth)))[0]
    acc_crossing_times = []
    for idx in acc_zero_crossings:
        t = times[idx:idx+2].mean()
        acc_crossing_times.append(t)
    
    if acc_crossing_times:
        print(f"Acceleration sign changes for {description} at times (seconds):")
        for i, t in enumerate(acc_crossing_times):
            print(f"  Crossing {i+1}: {t:.3f}s")
    
    # Create figure with 5 subplots stacked vertically
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.3)
    
    # Get inhale-exhale transition from flow profile
    base_subject = extract_base_subject(subject_name)
    flow_profile = pd.read_csv(f'{base_subject}FlowProfile.csv')
    
    # Apply smoothing in memory
    def apply_smoothing(data, window_size=20):
        if len(data) < window_size:
            return data.copy()
        smoothed = data.copy()
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
        return smoothed
    
    flow_times = flow_profile['time (s)'].values
    flow_rates = apply_smoothing(flow_profile['Massflowrate (kg/s)'].values)
    
    # Find zero crossings in flow profile for breathing cycle transitions
    zero_crossings = np.where(np.diff(np.signbit(flow_rates)))[0]
    inhale_exhale_label = ""
    
    if len(zero_crossings) >= 2:
        inhale_exhale_transition = flow_times[zero_crossings[1]]
        inhale_exhale_label = f"Inhale→Exhale (t={inhale_exhale_transition:.3f}s)"
    else:
        inhale_exhale_transition = None
    
    # Convert to mm/s and mm/s² for plotting
    vdotn_mm = vdotn * 1000  # Convert from m/s to mm/s
    vdotn_smooth_mm = vdotn_smooth * 1000  # Convert from m/s to mm/s
    adotn_smooth_mm = adotn_smooth * 1000  # Convert from m/s² to mm/s²
    
    # Store all axes for shared properties
    axs = []
    
    # 1. Plot Velocity (VdotN)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, vdotn_mm, 'b-', linewidth=2, label='Normal Velocity')
    if inhale_exhale_transition:
        ax1.axvline(x=inhale_exhale_transition, color='k', linestyle=':', alpha=0.7, label=inhale_exhale_label)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add vertical lines for velocity zero crossings
    for t in vel_crossing_times:
        ax1.axvline(x=t, color='b', linestyle='--', alpha=0.3)
    
    ax1.set_ylabel('VdotN (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
    ax1.set_title('Normal Velocity (v⃗·n⃗) Over Time', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    axs.append(ax1)
    
    # 2. Plot Acceleration (AdotN)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(times, adotn_smooth_mm, 'r-', linewidth=2, label='Normal Acceleration')
    if inhale_exhale_transition:
        ax2.axvline(x=inhale_exhale_transition, color='k', linestyle=':', alpha=0.7, label=inhale_exhale_label)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add vertical lines for acceleration zero crossings
    for t in acc_crossing_times:
        ax2.axvline(x=t, color='r', linestyle='--', alpha=0.3)
    
    ax2.set_ylabel('AdotN (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
    ax2.set_title('Normal Acceleration (a⃗·n⃗) Over Time', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    axs.append(ax2)
    
    # 3. Plot Pressure
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(times, pressure, 'g-', linewidth=2, label='Total Pressure')
    if inhale_exhale_transition:
        ax3.axvline(x=inhale_exhale_transition, color='k', linestyle=':', alpha=0.7, label=inhale_exhale_label)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.set_title('Total Pressure Over Time', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=10)
    axs.append(ax3)
    
    # 4. Plot Velocity Magnitude
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(times, velocity, 'm-', linewidth=2, label='Velocity Magnitude')
    if inhale_exhale_transition:
        ax4.axvline(x=inhale_exhale_transition, color='k', linestyle=':', alpha=0.7, label=inhale_exhale_label)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Velocity (m/s)', fontsize=LABEL_SIZE, fontweight='bold')
    ax4.set_title('Velocity Magnitude Over Time', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)
    axs.append(ax4)
    
    # 5. Plot Flow Rate
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    # Interpolate flow rate to match the time points if necessary
    from scipy.interpolate import interp1d
    if not np.array_equal(times, flow_times):
        flow_interp = interp1d(flow_times, flow_rates, bounds_error=False, fill_value="extrapolate")
        interp_flow_rates = flow_interp(times)
        ax5.plot(times, interp_flow_rates, 'm-', linewidth=2, label='Mass Flow Rate')
    else:
        ax5.plot(flow_times, flow_rates, 'm-', linewidth=2, label='Mass Flow Rate')
    
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.axvline(x=inhale_exhale_transition, color='k', linestyle=':', alpha=0.7, label=inhale_exhale_label)
    ax5.set_xlabel('Time (s)', fontsize=LABEL_SIZE, fontweight='bold')
    ax5.set_ylabel('Flow Rate (kg/s)', fontsize=LABEL_SIZE, fontweight='bold')
    ax5.set_title('Mass Flow Rate', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right', fontsize=10)
    axs.append(ax5)
    
    # Only show x-label on the bottom plot
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    # Add overall title
    fig.suptitle(f'Metrics vs Time Analysis\n{description} - Patch {patch_number}, Face {face_index}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to PDF
    pdf.savefig(fig, bbox_inches='tight')
    
    # Also save as PNG
    results_dir = Path(f'{subject_name}_results')
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_filename = figures_dir / f"{subject_name}_metrics_vs_time_{description.lower().replace(' ', '_')}_patch{patch_number}_face{face_index}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_symmetric_comparison_panel_clean(dfs, subject_name, pdf, pdfs_dir=None):
    """
    Create a symmetric 3x3 panel comparison of pressure, velocity, and acceleration across three anatomical points.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.
    This version does not include zero-crossing markers.
    
    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to (can be None if standalone_output=True)
    """
    print(f"\nGenerating clean symmetric comparison panel (without markers)...")
    
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define the locations and their descriptions - these should match the keys in dfs
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find mutual ranges for each variable across all points
    v_max = 0
    a_max = 0
    p_max = 0
    
    # Find the global minimum time to renormalize time to start from 0
    global_time_min = float('inf')
    for loc in locations:
        df = dfs[loc['key']]
        times = df['Time (s)'].values
        global_time_min = min(global_time_min, times.min())
    
    # The original inhale-exhale transition time
    original_inhale_exhale = 1.034  # This is the value from the command output
    
    # Calculate the normalized inhale-exhale transition time
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    print(f"Renormalizing time for clean plot: Original range started at {global_time_min:.3f}s")
    print(f"Inhale-exhale transition: Original at {original_inhale_exhale:.3f}s, Normalized at {normalized_inhale_exhale:.3f}s")
    
    # Process each dataset to find ranges and precompute derived values
    processed_data = {}
    for loc in locations:
        df = dfs[loc['key']]
        
        # Get the data
        times = df['Time (s)'].values
        vdotn = df['VdotN'].values * 1000  # Convert from m/s to mm/s
        pressure = df['Total Pressure (Pa)'].values
        
        # Normalize time to start from 0
        normalized_times = times - global_time_min
        
        # Calculate acceleration
        dt = times[1] - times[0]  # Use original time for dt calculation
        dvdotn = np.diff(vdotn)
        adotn = np.append(dvdotn / dt, dvdotn[-1] / dt)  # Already in mm/s² since vdotn is in mm/s
        
        # Update max values
        v_max = max(v_max, np.max(np.abs(vdotn)))
        a_max = max(a_max, np.max(np.abs(adotn)))
        p_max = max(p_max, np.max(np.abs(pressure)))
        
        # Store processed data for easy access during plotting
        processed_data[loc['key']] = {
            'normalized_times': normalized_times,
            'times': times,
            'vdotn': vdotn,
            'pressure': pressure,
            'adotn': adotn
        }
    
    # Add a small margin to prevent data from being exactly on the edge
    v_max *= 1.05
    a_max *= 1.05
    p_max *= 1.05
    
    # Common settings for all subplots
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Create a custom colormap that transitions at the normalized inhale-exhale point
    normalized_time_max = max([data['normalized_times'].max() for data in processed_data.values()])
    norm = plt.Normalize(0, normalized_time_max)  # Start from 0 for normalized time
    transition_norm = normalized_inhale_exhale / normalized_time_max
    
    # Ensure color points are in increasing order
    # Clamp transition points to valid range [0, 1]
    transition_norm = max(0.1, min(0.9, transition_norm))  # Keep transition between 10% and 90%
    
    colors = [
        (0, 'darkblue'),
        (max(0.01, transition_norm - 0.05), 'blue'),
        (max(0.02, transition_norm - 0.005), 'lightblue'),
        (transition_norm, 'white'),
        (min(0.98, transition_norm + 0.005), 'pink'),
        (min(0.99, transition_norm + 0.05), 'red'),
        (1, 'darkred')
    ]
    
    # Sort colors by position to ensure increasing order
    colors = sorted(colors, key=lambda x: x[0])
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        data = processed_data[loc['key']]
        
        # Get the processed data
        normalized_times = data['normalized_times']
        vdotn = data['vdotn']
        pressure = data['pressure']
        adotn = data['adotn']
        
        # Convert to mm/s and mm/s² for plotting
        vdotn_mm = vdotn * 1000  # Convert from m/s to mm/s
        adotn_mm = adotn * 1000  # Convert from m/s² to mm/s²
        v_max_mm = v_max * 1000  # Convert m/s to mm/s
        a_max_mm = a_max * 1000  # Convert m/s² to mm/s²
        
        # 1. Plot p vs v
        ax1 = fig.add_subplot(gs[i, 0])
        scatter1 = ax1.scatter(vdotn_mm, pressure, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlim(-v_max_mm, v_max_mm)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot p vs a
        ax2 = fig.add_subplot(gs[i, 1])
        scatter2 = ax2.scatter(adotn_mm, pressure, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlim(-a_max_mm, a_max_mm)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Plot v vs a
        ax3 = fig.add_subplot(gs[i, 2])
        scatter3 = ax3.scatter(adotn_mm, vdotn_mm, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlim(-a_max_mm, a_max_mm)
        ax3.set_ylim(-v_max_mm, v_max_mm)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Add a colorbar for time
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
    
    # Add a note about the normalized time
    fig.text(0.5, 0.01, f'Note: Time has been normalized to start at 0s. Original data started at {global_time_min:.3f}s.\nInhale-exhale transition at {normalized_inhale_exhale:.2f}s.', 
            fontsize=10, ha='center', va='bottom')
    
    # Add overall title
    fig.suptitle(f'Comparative Analysis of Pressure, Velocity and Acceleration\nSymmetric Plots Centered at Origin (Clean Version)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # [left, bottom, right, top]
    
    # Save to main PDF if provided
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    
    # Save a standalone PDF version to pdfs directory
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_3x3_panel_clean.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_3x3_panel_clean.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    
    plt.close()

def create_symmetric_comparison_panel(dfs, subject_name, pdf, pdfs_dir=None):
    """
    Create a symmetric 3x3 panel comparison of pressure, velocity, and acceleration across three anatomical points.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.
    
    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to (can be None if standalone_output=True)
    """
    print(f"\nGenerating symmetric comparison panel...")
    
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define the locations and their descriptions - these should match the keys in dfs
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find mutual ranges for each variable across all points
    v_max = 0
    a_max = 0
    p_max = 0
    
    # Find the global minimum time to renormalize time to start from 0
    global_time_min = float('inf')
    for loc in locations:
        df = dfs[loc['key']]
        times = df['Time (s)'].values
        global_time_min = min(global_time_min, times.min())
    
    # The original inhale-exhale transition time
    original_inhale_exhale = 1.034  # This is the value from the command output
    
    # Calculate the normalized inhale-exhale transition time
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    print(f"Renormalizing time: Original range started at {global_time_min:.3f}s")
    print(f"Inhale-exhale transition: Original at {original_inhale_exhale:.3f}s, Normalized at {normalized_inhale_exhale:.3f}s")
    
    # Process each dataset to find ranges, zero crossings, and precompute derived values
    processed_data = {}
    for loc in locations:
        df = dfs[loc['key']]
        
        # Get the data
        times = df['Time (s)'].values
        vdotn = df['VdotN'].values * 1000  # Convert from m/s to mm/s
        pressure = df['Total Pressure (Pa)'].values
        
        # Normalize time to start from 0
        normalized_times = times - global_time_min
        
        # Calculate acceleration
        dt = times[1] - times[0]  # Use original time for dt calculation
        dvdotn = np.diff(vdotn)
        adotn = np.append(dvdotn / dt, dvdotn[-1] / dt)  # Already in mm/s² since vdotn is in mm/s
        
        # Find zero-crossings and other relevant points
        # Index of velocity crossing zero (from negative to positive)
        vel_zero_idxs = np.where(np.diff(np.signbit(vdotn)))[0]
        vel_zero_times = times[vel_zero_idxs]
        vel_zero_normalized = normalized_times[vel_zero_idxs]
        
        # Index of acceleration crossing zero
        acc_zero_idxs = np.where(np.diff(np.signbit(adotn)))[0]
        acc_zero_times = times[acc_zero_idxs]
        acc_zero_normalized = normalized_times[acc_zero_idxs]
        
        # Update max values
        v_max = max(v_max, np.max(np.abs(vdotn)))
        a_max = max(a_max, np.max(np.abs(adotn)))
        p_max = max(p_max, np.max(np.abs(pressure)))
        
        # Store processed data for easy access during plotting
        processed_data[loc['key']] = {
            'normalized_times': normalized_times,
            'times': times,
            'vdotn': vdotn,
            'pressure': pressure,
            'adotn': adotn,
            'vel_zero_normalized': vel_zero_normalized,
            'acc_zero_normalized': acc_zero_normalized
        }
        
        # Print zero crossing info
        print(f"\nZero crossings for {loc['description']}:")
        print(f"  Velocity zero crossings: {len(vel_zero_times)} at times: {', '.join([f'{t:.3f}s' for t in vel_zero_times])}")
        print(f"  Acceleration zero crossings: {len(acc_zero_times)} at times: {', '.join([f'{t:.3f}s' for t in acc_zero_times])}")
    
    # Add a small margin to prevent data from being exactly on the edge
    v_max *= 1.05
    a_max *= 1.05
    p_max *= 1.05
    
    # Common settings for all subplots
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Create a custom colormap that transitions at the normalized inhale-exhale point
    normalized_time_max = max([data['normalized_times'].max() for data in processed_data.values()])
    norm = plt.Normalize(0, normalized_time_max)  # Start from 0 for normalized time
    transition_norm = normalized_inhale_exhale / normalized_time_max
    
    # Ensure color points are in increasing order
    # Clamp transition points to valid range [0, 1]
    transition_norm = max(0.1, min(0.9, transition_norm))  # Keep transition between 10% and 90%
    
    colors = [
        (0, 'darkblue'),
        (max(0.01, transition_norm - 0.05), 'blue'),
        (max(0.02, transition_norm - 0.005), 'lightblue'),
        (transition_norm, 'white'),
        (min(0.98, transition_norm + 0.005), 'pink'),
        (min(0.99, transition_norm + 0.05), 'red'),
        (1, 'darkred')
    ]
    
    # Sort colors by position to ensure increasing order
    colors = sorted(colors, key=lambda x: x[0])
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        data = processed_data[loc['key']]
        
        # Get the processed data
        normalized_times = data['normalized_times']
        vdotn = data['vdotn']
        pressure = data['pressure']
        adotn = data['adotn']
        vel_zero_normalized = data['vel_zero_normalized']
        acc_zero_normalized = data['acc_zero_normalized']
        
        # Convert to mm/s and mm/s² for plotting
        vdotn_mm = vdotn * 1000  # Convert from m/s to mm/s
        adotn_mm = adotn * 1000  # Convert from m/s² to mm/s²
        v_max_mm = v_max * 1000  # Convert m/s to mm/s
        a_max_mm = a_max * 1000  # Convert m/s² to mm/s²
        
        # 1. Plot p vs v
        ax1 = fig.add_subplot(gs[i, 0])
        scatter1 = ax1.scatter(vdotn_mm, pressure, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlim(-v_max_mm, v_max_mm)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add markers for zero crossings
        for t_normalized in vel_zero_normalized:
            # Find closest data point
            idx = np.abs(normalized_times - t_normalized).argmin()
            ax1.plot(vdotn_mm[idx], pressure[idx], 'o', color='lime', markersize=8, alpha=0.7)
        
        for t_normalized in acc_zero_normalized:
            # Find closest data point
            idx = np.abs(normalized_times - t_normalized).argmin()
            ax1.plot(vdotn_mm[idx], pressure[idx], 's', color='yellow', markersize=8, alpha=0.7)
        
        # Mark inhale-exhale transition point
        inhale_exhale_idx = np.abs(normalized_times - normalized_inhale_exhale).argmin()
        ax1.plot(vdotn_mm[inhale_exhale_idx], pressure[inhale_exhale_idx], '*', color='black', markersize=12)
        
        # 2. Plot p vs a
        ax2 = fig.add_subplot(gs[i, 1])
        scatter2 = ax2.scatter(adotn_mm, pressure, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlim(-a_max_mm, a_max_mm)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add markers for zero crossings
        for t_normalized in vel_zero_normalized:
            # Find closest data point
            idx = np.abs(normalized_times - t_normalized).argmin()
            ax2.plot(adotn_mm[idx], pressure[idx], 'o', color='lime', markersize=8, alpha=0.7)
        
        for t_normalized in acc_zero_normalized:
            # Find closest data point
            idx = np.abs(normalized_times - t_normalized).argmin()
            ax2.plot(adotn_mm[idx], pressure[idx], 's', color='yellow', markersize=8, alpha=0.7)
        
        # Mark inhale-exhale transition point
        ax2.plot(adotn_mm[inhale_exhale_idx], pressure[inhale_exhale_idx], '*', color='black', markersize=12)
        
        # 3. Plot v vs a
        ax3 = fig.add_subplot(gs[i, 2])
        scatter3 = ax3.scatter(adotn_mm, vdotn_mm, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlim(-a_max_mm, a_max_mm)
        ax3.set_ylim(-v_max_mm, v_max_mm)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add markers for zero crossings
        for t_normalized in vel_zero_normalized:
            # Find closest data point
            idx = np.abs(normalized_times - t_normalized).argmin()
            ax3.plot(adotn_mm[idx], vdotn_mm[idx], 'o', color='lime', markersize=8, alpha=0.7)
        
        for t_normalized in acc_zero_normalized:
            # Find closest data point
            idx = np.abs(normalized_times - t_normalized).argmin()
            ax3.plot(adotn_mm[idx], vdotn_mm[idx], 's', color='yellow', markersize=8, alpha=0.7)
        
        # Mark inhale-exhale transition point
        ax3.plot(adotn_mm[inhale_exhale_idx], vdotn_mm[inhale_exhale_idx], '*', color='black', markersize=12)
    
    # Add a legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=8, alpha=0.7, linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=8, alpha=0.7, linestyle='None'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12, linestyle='None')
    ]
    labels = ['Velocity Zero Crossing', 'Acceleration Zero Crossing', 'Inhale-Exhale Transition']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
    
    # Add a colorbar for time
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
    
    # Add a note about the normalized time
    fig.text(0.5, 0.01, f'Note: Time has been normalized to start at 0s. Original data started at {global_time_min:.3f}s.\nInhale-exhale transition at {normalized_inhale_exhale:.2f}s.', 
            fontsize=10, ha='center', va='bottom')
    
    # Add overall title
    fig.suptitle(f'Comparative Analysis of Pressure, Velocity and Acceleration\nSymmetric Plots Centered at Origin', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # [left, bottom, right, top]
    
    # Save to main PDF if provided
    if pdf is not None:
        pdf.savefig(fig)
    
    # Save a standalone PDF version to pdfs directory
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_3x3_panel.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_3x3_panel.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    
    plt.close()

def create_symmetric_comparison_panel_smooth(dfs, subject_name, pdf, pdfs_dir=None):
    """
    Create a symmetric 3x3 panel comparison of pressure, velocity, and acceleration across three anatomical points.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.
    This version applies smoothing to the data.
    
    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to (can be None if standalone_output=True)
    """
    print(f"\nGenerating smoothed symmetric comparison panel...")
    
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define moving average function for smoothing
    def moving_average(data, window_size=20):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Define the locations and their descriptions - these should match the keys in dfs
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find mutual ranges for each variable across all points
    v_max = 0
    a_max = 0
    p_max = 0
    
    # Find the global minimum time to renormalize time to start from 0
    global_time_min = float('inf')
    for loc in locations:
        df = dfs[loc['key']]
        times = df['Time (s)'].values
        global_time_min = min(global_time_min, times.min())
    
    # The original inhale-exhale transition time
    original_inhale_exhale = 1.034  # This is the value from the command output
    
    # Calculate the normalized inhale-exhale transition time
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    print(f"Renormalizing time for smoothed plot: Original range started at {global_time_min:.3f}s")
    print(f"Inhale-exhale transition: Original at {original_inhale_exhale:.3f}s, Normalized at {normalized_inhale_exhale:.3f}s")
    
    # Process each dataset to find ranges and precompute derived values
    processed_data = {}
    for loc in locations:
        df = dfs[loc['key']]
        
        # Get the data
        times = df['Time (s)'].values
        vdotn = df['VdotN'].values * 1000  # Convert from m/s to mm/s
        pressure = df['Total Pressure (Pa)'].values
        
        # Normalize time to start from 0
        normalized_times = times - global_time_min
        
        # Apply smoothing - handle small datasets gracefully
        data_length = len(times)
        
        if data_length <= 5:
            # Too few points, no smoothing
            print(f"Too few data points ({data_length}) for {loc['description']}, skipping smoothing")
            vdotn_smooth = vdotn.copy()
            pressure_smooth = pressure.copy()
        else:
            # Calculate appropriate window size
            # For savgol_filter: window_size must be odd and <= data_length
            # Also need window_size >= polyorder + 1 (we use polyorder=3, so >= 4)
            max_window = data_length if data_length % 2 == 1 else data_length - 1
            min_window = 5  # Minimum for polyorder=3 + buffer
            
            # Try to use 20% of data length, but respect constraints
            desired_window = max(min_window, int(0.2 * data_length))
            window_size = min(desired_window, max_window)
            
            # Ensure odd
            if window_size % 2 == 0:
                window_size -= 1
            
            print(f"Applying smoothing with window size {window_size} for {loc['description']} (data length: {data_length})")
            
            try:
                # Apply Savitzky-Golay filter for smoothing
                vdotn_smooth = savgol_filter(vdotn, window_size, 3)
                pressure_smooth = savgol_filter(pressure, window_size, 3)
            except ValueError as e:
                print(f"Savgol filter failed for {loc['description']}, falling back to moving average: {e}")
                # Fallback to simple moving average
                def simple_smooth(data, window):
                    return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
                
                vdotn_smooth = simple_smooth(vdotn, min(5, data_length))
                pressure_smooth = simple_smooth(pressure, min(5, data_length))
        
        # Calculate acceleration from smoothed velocity
        dt = times[1] - times[0]  # Should be constant
        dvdotn = np.diff(vdotn_smooth)
        adotn_smooth = np.append(dvdotn / dt, dvdotn[-1] / dt)
        
        # Update max values based on smooth data
        v_max = max(v_max, np.max(np.abs(vdotn_smooth)))
        a_max = max(a_max, np.max(np.abs(adotn_smooth)))
        p_max = max(p_max, np.max(np.abs(pressure_smooth)))
        
        # Store processed data for easy access during plotting
        processed_data[loc['key']] = {
            'normalized_times': normalized_times,
            'vdotn_smooth': vdotn_smooth,
            'pressure_smooth': pressure_smooth,
            'adotn_smooth': adotn_smooth
        }
    
    # Add a small margin to prevent data from being exactly on the edge
    v_max *= 1.05
    a_max *= 1.05
    p_max *= 1.05
    
    # Common settings for all subplots
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Create a custom colormap that transitions at the normalized inhale-exhale point
    normalized_time_max = max([data['normalized_times'].max() for data in processed_data.values()])
    norm = plt.Normalize(0, normalized_time_max)  # Start from 0 for normalized time
    transition_norm = normalized_inhale_exhale / normalized_time_max
    
    # Ensure color points are in increasing order
    # Clamp transition points to valid range [0, 1]
    transition_norm = max(0.1, min(0.9, transition_norm))  # Keep transition between 10% and 90%
    
    colors = [
        (0, 'darkblue'),
        (max(0.01, transition_norm - 0.05), 'blue'),
        (max(0.02, transition_norm - 0.005), 'lightblue'),
        (transition_norm, 'white'),
        (min(0.98, transition_norm + 0.005), 'pink'),
        (min(0.99, transition_norm + 0.05), 'red'),
        (1, 'darkred')
    ]
    
    # Sort colors by position to ensure increasing order
    colors = sorted(colors, key=lambda x: x[0])
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        data = processed_data[loc['key']]
        
        # Get the smoothed data
        normalized_times = data['normalized_times']
        vdotn_smooth = data['vdotn_smooth']
        pressure_smooth = data['pressure_smooth']
        adotn_smooth = data['adotn_smooth']
        
        # Convert to mm/s and mm/s² for plotting
        vdotn_smooth_mm = vdotn_smooth * 1000  # Convert from m/s to mm/s
        adotn_smooth_mm = adotn_smooth * 1000  # Convert from m/s² to mm/s²
        v_max_mm = v_max * 1000  # Convert m/s to mm/s
        a_max_mm = a_max * 1000  # Convert m/s² to mm/s²
        
        # 1. Plot p vs v (smoothed)
        ax1 = fig.add_subplot(gs[i, 0])
        scatter1 = ax1.scatter(vdotn_smooth_mm, pressure_smooth, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlim(-v_max_mm, v_max_mm)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]} (Smoothed)', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Mark inhale-exhale transition point
        inhale_exhale_idx = np.abs(normalized_times - normalized_inhale_exhale).argmin()
        ax1.plot(vdotn_smooth_mm[inhale_exhale_idx], pressure_smooth[inhale_exhale_idx], '*', color='black', markersize=12)
        
        # 2. Plot p vs a (smoothed)
        ax2 = fig.add_subplot(gs[i, 1])
        scatter2 = ax2.scatter(adotn_smooth_mm, pressure_smooth, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlim(-a_max_mm, a_max_mm)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]} (Smoothed)', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Mark inhale-exhale transition point
        ax2.plot(adotn_smooth_mm[inhale_exhale_idx], pressure_smooth[inhale_exhale_idx], '*', color='black', markersize=12)
        
        # 3. Plot v vs a (smoothed)
        ax3 = fig.add_subplot(gs[i, 2])
        scatter3 = ax3.scatter(adotn_smooth_mm, vdotn_smooth_mm, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlim(-a_max_mm, a_max_mm)
        ax3.set_ylim(-v_max_mm, v_max_mm)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]} (Smoothed)', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Mark inhale-exhale transition point
        ax3.plot(adotn_smooth_mm[inhale_exhale_idx], vdotn_smooth_mm[inhale_exhale_idx], '*', color='black', markersize=12)
    
    # Add a legend for the inhale-exhale transition point
    handles = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=12, linestyle='None')
    ]
    labels = ['Inhale-Exhale Transition']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=1, fontsize=12)
    
    # Add a colorbar for time
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
    
    # Add a note about the normalized time and smoothing
    fig.text(0.5, 0.01, f'Note: Time has been normalized to start at 0s. Original data started at {global_time_min:.3f}s.\nInhale-exhale transition at {normalized_inhale_exhale:.2f}s.', 
            fontsize=10, ha='center', va='bottom')
    
    # Add overall title
    fig.suptitle(f'Comparative Analysis of Pressure, Velocity and Acceleration\nSmoothed, Symmetric Plots Centered at Origin', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # [left, bottom, right, top]
    
    # Save to main PDF if provided
    if pdf is not None:
        pdf.savefig(fig)
    
    # Save a standalone PDF version to pdfs directory
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_3x3_panel_smooth.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_3x3_panel_smooth.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    
    plt.close()

def create_symmetric_comparison_panel_smooth_with_markers(dfs, subject_name, pdf, pdfs_dir=None):
    """
    Create a symmetric 3x3 panel comparison of pressure, velocity, and acceleration across three anatomical points.
    This version applies a moving average smoothing filter with window size 20 to all data,
    and also adds markers and text labels for zero crossings.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.
    
    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to
    """
    print(f"\nGenerating smoothed symmetric comparison panel with zero-crossing markers...")
    
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define the locations and their descriptions - these should match the keys in dfs
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find mutual ranges for each variable across all points
    v_max = 0
    a_max = 0
    p_max = 0
    
    # Find the global minimum time to renormalize time to start from 0
    global_time_min = float('inf')
    for loc in locations:
        df = dfs[loc['key']]
        times = df['Time (s)'].values
        global_time_min = min(global_time_min, times.min())
    
    # The original inhale-exhale transition time
    original_inhale_exhale = 1.034  # This is the value from the command output
    
    # Calculate the normalized inhale-exhale transition time
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    print(f"Renormalizing time for smoothed plot with markers: Original range started at {global_time_min:.3f}s")
    print(f"Inhale-exhale transition: Original at {original_inhale_exhale:.3f}s, Normalized at {normalized_inhale_exhale:.3f}s")
    
    # Define the improved moving average smoothing function that handles edge cases
    def moving_average(data, window_size=20):
        """
        Apply a moving average filter to smooth the data, properly handling edge cases.
        For edge points where a full window is not available, the original data values are kept.
        """
        if len(data) < window_size:
            return data.copy()  # Return a copy of original data if too short
            
        # Initialize the output array with original data (will keep edges unchanged)
        smoothed = data.copy()
        
        # Only smooth the points where we have enough data for a full window
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            # Calculate average only where we have a full window
            smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
            
        return smoothed
    
    # Process each dataset to find ranges, zero crossings, and precompute derived values
    processed_data = {}
    for loc in locations:
        df = dfs[loc['key']]
        
        # Get the data
        times = df['Time (s)'].values
        vdotn = df['VdotN'].values * 1000  # Convert from m/s to mm/s
        pressure = df['Total Pressure (Pa)'].values
        
        # Normalize time to start from 0
        normalized_times = times - global_time_min
        
        # Calculate acceleration
        dt = times[1] - times[0]  # Use original time for dt calculation
        dvdotn = np.diff(vdotn)
        adotn = np.append(dvdotn / dt, dvdotn[-1] / dt)  # Already in mm/s² since vdotn is in mm/s
        
        # Find sign changes in original data before smoothing (for accurate zero crossings)
        v_crossings = find_zero_crossings(normalized_times, vdotn)
        a_crossings = find_zero_crossings(normalized_times, adotn)
        p_crossings = find_zero_crossings(normalized_times, pressure)

        # Apply smoothing to all data
        vdotn_smooth = moving_average(vdotn)
        adotn_smooth = moving_average(adotn)
        pressure_smooth = moving_average(pressure)
        
        # Update max values
        v_max = max(v_max, np.max(np.abs(vdotn_smooth)))
        a_max = max(a_max, np.max(np.abs(adotn_smooth)))
        p_max = max(p_max, np.max(np.abs(pressure_smooth)))
        
        # Store processed data for easy access during plotting
        processed_data[loc['key']] = {
            'normalized_times': normalized_times,
            'times': times,
            'vdotn': vdotn_smooth,
            'pressure': pressure_smooth,
            'adotn': adotn_smooth,
            'v_crossings': v_crossings,
            'a_crossings': a_crossings,
            'p_crossings': p_crossings
        }
    
    # Add a small margin to prevent data from being exactly on the edge
    v_max *= 1.05
    a_max *= 1.05
    p_max *= 1.05
    
    # Common settings for all subplots
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Create a custom colormap that transitions at the normalized inhale-exhale point
    normalized_time_max = max([data['normalized_times'].max() for data in processed_data.values()])
    norm = plt.Normalize(0, normalized_time_max)  # Start from 0 for normalized time
    transition_norm = normalized_inhale_exhale / normalized_time_max
    
    # Ensure color points are in increasing order
    # Clamp transition points to valid range [0, 1]
    transition_norm = max(0.1, min(0.9, transition_norm))  # Keep transition between 10% and 90%
    
    colors = [
        (0, 'darkblue'),
        (max(0.01, transition_norm - 0.05), 'blue'),
        (max(0.02, transition_norm - 0.005), 'lightblue'),
        (transition_norm, 'white'),
        (min(0.98, transition_norm + 0.005), 'pink'),
        (min(0.99, transition_norm + 0.05), 'red'),
        (1, 'darkred')
    ]
    
    # Sort colors by position to ensure increasing order
    colors = sorted(colors, key=lambda x: x[0])
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        data = processed_data[loc['key']]
        
        # Get the processed data
        normalized_times = data['normalized_times']
        vdotn = data['vdotn']
        pressure = data['pressure']
        adotn = data['adotn']
        v_crossings = data['v_crossings']
        a_crossings = data['a_crossings']
        p_crossings = data['p_crossings']
        
        # 1. Plot p vs v
        ax1 = fig.add_subplot(gs[i, 0])
        scatter1 = ax1.scatter(vdotn, pressure, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark the zero crossings in the p vs v plot
        for v_cross in v_crossings:
            t_idx = np.abs(normalized_times - v_cross).argmin()
            # Mark v=0 crossings with vertical lines
            ax1.axvline(x=0, ymin=0.45, ymax=0.55, color='red', linewidth=2)
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(normalized_times - v_cross)).argmin()
            p_at_cross = pressure[idx_at_cross]
            ax1.annotate(f"{v_cross:.2f}s", 
                       xy=(0, p_at_cross), 
                       xytext=(0.1*v_max, p_at_cross + 0.1*p_max),
                       arrowprops=dict(arrowstyle="->", color='red'),
                       color='red', fontsize=10)
        
        for p_cross in p_crossings:
            t_idx = np.abs(normalized_times - p_cross).argmin()
            # Mark p=0 crossings with horizontal lines
            ax1.axhline(y=0, xmin=0.45, xmax=0.55, color='blue', linewidth=2)
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(normalized_times - p_cross)).argmin()
            v_at_cross = vdotn[idx_at_cross]
            ax1.annotate(f"{p_cross:.2f}s", 
                       xy=(v_at_cross, 0), 
                       xytext=(v_at_cross - 0.1*v_max, -0.1*p_max),
                       arrowprops=dict(arrowstyle="->", color='blue'),
                       color='blue', fontsize=10)
        
        ax1.set_xlim(-v_max, v_max)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot p vs a
        ax2 = fig.add_subplot(gs[i, 1])
        scatter2 = ax2.scatter(adotn, pressure, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark the zero crossings in the p vs a plot
        for a_cross in a_crossings:
            t_idx = np.abs(normalized_times - a_cross).argmin()
            # Mark a=0 crossings with vertical lines
            ax2.axvline(x=0, ymin=0.45, ymax=0.55, color='red', linewidth=2)
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(normalized_times - a_cross)).argmin()
            p_at_cross = pressure[idx_at_cross]
            ax2.annotate(f"{a_cross:.2f}s", 
                       xy=(0, p_at_cross), 
                       xytext=(0.1*a_max, p_at_cross + 0.1*p_max),
                       arrowprops=dict(arrowstyle="->", color='red'),
                       color='red', fontsize=10)
        
        for p_cross in p_crossings:
            t_idx = np.abs(normalized_times - p_cross).argmin()
            # Mark p=0 crossings with horizontal lines
            ax2.axhline(y=0, xmin=0.45, xmax=0.55, color='blue', linewidth=2)
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(normalized_times - p_cross)).argmin()
            a_at_cross = adotn[idx_at_cross]
            ax2.annotate(f"{p_cross:.2f}s", 
                       xy=(a_at_cross, 0), 
                       xytext=(a_at_cross - 0.1*a_max, -0.1*p_max),
                       arrowprops=dict(arrowstyle="->", color='blue'),
                       color='blue', fontsize=10)
        
        ax2.set_xlim(-a_max, a_max)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Plot v vs a
        ax3 = fig.add_subplot(gs[i, 2])
        scatter3 = ax3.scatter(adotn, vdotn, c=normalized_times, cmap=custom_cmap, norm=norm)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark the zero crossings in the v vs a plot
        for v_cross in v_crossings:
            t_idx = np.abs(normalized_times - v_cross).argmin()
            # Mark v=0 crossings with horizontal lines
            ax3.axhline(y=0, xmin=0.45, xmax=0.55, color='green', linewidth=2)
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(normalized_times - v_cross)).argmin()
            a_at_cross = adotn[idx_at_cross]
            ax3.annotate(f"{v_cross:.2f}s", 
                       xy=(a_at_cross, 0), 
                       xytext=(a_at_cross + 0.1*a_max, 0.1*v_max),
                       arrowprops=dict(arrowstyle="->", color='green'),
                       color='green', fontsize=10)
        
        for a_cross in a_crossings:
            t_idx = np.abs(normalized_times - a_cross).argmin()
            # Mark a=0 crossings with vertical lines
            ax3.axvline(x=0, ymin=0.45, ymax=0.55, color='purple', linewidth=2)
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(normalized_times - a_cross)).argmin()
            v_at_cross = vdotn[idx_at_cross]
            ax3.annotate(f"{a_cross:.2f}s", 
                       xy=(0, v_at_cross), 
                       xytext=(-0.1*a_max, v_at_cross - 0.1*v_max),
                       arrowprops=dict(arrowstyle="->", color='purple'),
                       color='purple', fontsize=10)
        
        ax3.set_xlim(-a_max, a_max)
        ax3.set_ylim(-v_max, v_max)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Add a colorbar for time
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
    
    # Add a note about the normalized time and smoothing
    fig.text(0.5, 0.01, f'Note: Time has been normalized to start at 0s. Original data started at {global_time_min:.3f}s.\nInhale-exhale transition at {normalized_inhale_exhale:.2f}s. Data smoothed with moving average (window size: 20).', 
            fontsize=10, ha='center', va='bottom')
    
    # Add overall title
    fig.suptitle(f'Comparative Analysis of Pressure, Velocity and Acceleration\nSymmetric Plots Centered at Origin (Smoothed Version with Zero-Crossing Markers)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # [left, bottom, right, top]
    
    # Save to main PDF if provided
    if pdf is not None:
        pdf.savefig(fig)
    
    # Save a standalone PDF version to pdfs directory
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_3x3_panel_smoothed_with_markers.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_3x3_panel_smoothed_with_markers.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    
    plt.close()
    print("Smoothed symmetric comparison panel with markers completed.")

# Helper function to find zero-crossings with interpolation for more precise times
def find_zero_crossings(times, values):
    """Find the times when values cross zero.
    
    Args:
        times: Array of time points
        values: Array of values corresponding to time points
        
    Returns:
        List of times when values cross zero
    """
    # Find indices where the value changes sign
    zero_crossings = np.where(np.diff(np.signbit(values)))[0]
    crossing_times = []
    
    for idx in zero_crossings:
        # For each zero crossing, interpolate to find a more precise time
        if idx + 1 < len(times) and idx >= 0:
            t0, t1 = times[idx], times[idx + 1]
            v0, v1 = values[idx], values[idx + 1]
            
            # Linear interpolation to find t where v = 0
            # v = v0 + (v1-v0)*(t-t0)/(t1-t0) = 0
            # Solve for t: t = t0 - v0 * (t1-t0)/(v1-v0)
            if v1 != v0:  # Avoid division by zero
                t_cross = t0 - v0 * (t1 - t0) / (v1 - v0)
                crossing_times.append(t_cross)
    
    return crossing_times

def create_airway_surface_analysis_plot(df, subject_name, description, patch_number, face_index, pdf, output_dir=None, smoothing_window=20):
    """Create a plot showing analysis for a single point, including pressure vs velocity/acceleration relationships."""
    # Convert DataFrame to trajectory_data structure
    trajectory_data = []
    for _, row in df.iterrows():
                data_point = {
                    'time': row['Time (s)'],
                    'time_point': row['Time Point'],
                    'x': row['X (m)'],
                    'y': row['Y (m)'],
                    'z': row['Z (m)'],
                    'pressure': row['Total Pressure (Pa)'],
                    'velocity': row['Velocity: Magnitude (m/s)'],
                    'velocity_i': row['Velocity[i] (m/s)'],
                    'velocity_j': row['Velocity[j] (m/s)'],
                    'velocity_k': row['Velocity[k] (m/s)'],
                    'area_i': row['Area[i] (m^2)'],
                    'area_j': row['Area[j] (m^2)'],
                    'area_k': row['Area[k] (m^2)'],
                    'vdotn': row['VdotN'],
                    'adjusted_pressure': row['Adjusted Pressure (Pa)'],
                    'patch35_avg_pressure': row['Patch35 Avg Pressure (Pa)']
                }
                trajectory_data.append(data_point)
            
            # Sort trajectory data by time
    trajectory_data = sorted(trajectory_data, key=lambda x: x['time'])

            # Create plots with 2x3 layout
    fig = plt.figure(figsize=(24, 16))  # Adjusted figure size for wider layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Total Pressure vs VdotN
    ax2 = fig.add_subplot(gs[0, 1])  # Total Pressure vs AdotN
    ax3 = fig.add_subplot(gs[0, 2])  # VdotN vs AdotN (moved from subplot 6)
    ax4 = fig.add_subplot(gs[1, 0])  # dp/dt vs VdotN
    ax5 = fig.add_subplot(gs[1, 1])  # dp/dt vs AdotN
    ax6 = fig.add_subplot(gs[1, 2])  # Time series plot with flow rate and velocities
            
            # Common style settings for all subplots
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce number of x-ticks
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce number of y-ticks
                ax.xaxis.label.set_fontsize(LABEL_SIZE * 1.1)
                ax.xaxis.label.set_fontweight('bold')
                ax.yaxis.label.set_fontsize(LABEL_SIZE * 1.1)
                ax.yaxis.label.set_fontweight('bold')
                ax.title.set_fontsize(TITLE_SIZE * 1.1)
                ax.title.set_fontweight('bold')
            
    fig.suptitle(f'Point Movement Analysis - {description}\nPatch {patch_number}, Face {face_index}', 
                        fontsize=16, fontweight='bold')
            
            # Load flow profile to find inhale-exhale transition
    base_subject = extract_base_subject(subject_name)
    flow_profile = pd.read_csv(f'{base_subject}FlowProfile.csv')
    
    # Apply smoothing in memory
    def apply_smoothing(data, window_size=smoothing_window):
        if len(data) < window_size:
            return data.copy()
        smoothed = data.copy()
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
        return smoothed
    
    # Smooth the flow rates for analysis
    flow_profile.iloc[:, 1] = apply_smoothing(flow_profile.iloc[:, 1].values)
            # Find zero crossings in the flow profile (where y=0)
    zero_crossings = np.where(np.diff(np.signbit(flow_profile['Massflowrate (kg/s)'])))[0]
            # Get the second zero crossing time (inhale to exhale transition)
    if len(zero_crossings) >= 2:
                inhale_exhale_transition = flow_profile['time (s)'].iloc[zero_crossings[1]]
                print(f"Inhale to exhale transition detected at {inhale_exhale_transition:.3f} seconds")
    else:
                print("Warning: Could not find second zero crossing in flow profile")
    inhale_exhale_transition = np.mean([p['time'] for p in trajectory_data])  # Fallback to middle of time range
            
    times = np.array([p['time'] for p in trajectory_data])
    pressures = np.array([p['pressure'] for p in trajectory_data])
            
            # Calculate smoothed pressure using moving average  
    window_size = smoothing_window * 5  # Use 5x smoothing window for pressure (was 100 with default 20)
    window = np.bartlett(window_size)
    window = window / window.sum()
    pad_width = window_size // 2
    padded_pressure = np.pad(pressures, pad_width, mode='edge')
    smoothed_pressure = np.convolve(padded_pressure, window, mode='valid')
    if len(smoothed_pressure) > len(pressures):
        smoothed_pressure = smoothed_pressure[:len(pressures)]
    elif len(smoothed_pressure) < len(pressures):
        smoothed_pressure = np.pad(smoothed_pressure, (0, len(pressures) - len(smoothed_pressure)), mode='edge')
            
            # Calculate dp/dt from smoothed pressure
    dt = np.diff(times)
    dp = np.diff(smoothed_pressure)
    dpdt = np.append(dp / dt, dp[-1] / dt[-1])  # Add last point to match array length
            
            # Get velocity and acceleration components
    velocities = np.array([p['velocity'] for p in trajectory_data])
    vdotn = np.array([p.get('vdotn', 0) for p in trajectory_data])
            
            # Calculate AdotN using acceleration and normal vectors
    normal_vectors = np.array([[p.get('area_i', 0), p.get('area_j', 0), p.get('area_k', 0)] 
                                     for p in trajectory_data])
            # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(normal_vectors, axis=1)
    norms = np.where(norms < 1e-10, 1e-10, norms)  # Replace zero norms with small epsilon
    normal_vectors = normal_vectors / norms[:, np.newaxis]
            
            # Calculate acceleration vectors from velocity changes
    velocity_vectors = np.array([[p.get('velocity_i', 0), p.get('velocity_j', 0), p.get('velocity_k', 0)] 
                                for p in trajectory_data])
    dv_vectors = np.diff(velocity_vectors, axis=0)
    dt = np.diff(times)  # Calculate actual time differences
            
            # Print detailed time step information
    inconsistent_steps = np.where(np.abs(dt - 0.001) > 1e-6)[0]
    if len(inconsistent_steps) > 0:
        print("\nFound inconsistent time steps at indices:")
        for idx in inconsistent_steps:
            print(f"Between t={times[idx]:.3f}s and t={times[idx+1]:.3f}s: dt={dt[idx]:.6f}s")
            
            # Calculate adotn in two ways:
            # 1. Using velocity vector differences
    acceleration_vectors = dv_vectors / dt[:, np.newaxis]  # Using actual time differences
    adotn_from_vectors = np.sum(acceleration_vectors * normal_vectors[1:], axis=1)
    adotn_from_vectors = np.append(adotn_from_vectors, adotn_from_vectors[-1])
            
            # 2. Using forward difference of vdotn
    dvdotn = np.diff(vdotn)
    adotn_from_vdotn = dvdotn / dt  # Using actual time differences
    adotn_from_vdotn = np.append(adotn_from_vdotn, adotn_from_vdotn[-1])
            
            # Compare the two methods
    max_diff = np.max(np.abs(adotn_from_vectors - adotn_from_vdotn))
    print(f"\nMaximum difference between adotn calculation methods: {max_diff:.6f} m/s²")
            
            # Use the vdotn difference method for adotn
    adotn = adotn_from_vdotn
            
            # Save the data for inspection with point description in filename
    # Create debug directory at the same level as results
    debug_dir = Path(f'{subject_name}_results') / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_filename = debug_dir / f'{subject_name}_debug_acceleration_{description.lower().replace(" ", "_")}_patch{patch_number}_face{face_index}.csv'
            
            # Ensure all arrays are the same length by padding
    dt_padded = np.append(dt, dt[-1])
    dvdotn_padded = np.append(dvdotn, dvdotn[-1])
            
    debug_data = pd.DataFrame({
                'Time (s)': times,
                'dt (s)': dt_padded,
                'VdotN': vdotn,
                'dVdotN': dvdotn_padded,  # Raw difference in VdotN
                'AdotN_from_VdotN': adotn_from_vdotn,  # a·n calculated from VdotN differences
                'AdotN_from_vectors': adotn_from_vectors,  # a·n calculated from velocity vector differences
        'Velocity_i': velocity_vectors[:, 0],
        'Velocity_j': velocity_vectors[:, 1],
        'Velocity_k': velocity_vectors[:, 2],
                'Normal_i': normal_vectors[:, 0],
                'Normal_j': normal_vectors[:, 1],
                'Normal_k': normal_vectors[:, 2],
                'Total_Pressure': pressures
            })
    debug_data.to_csv(debug_filename, index=False)
    print(f"Saved debug data to {debug_filename}")
            
            # Create custom colormap that transitions at inhale-exhale point
    norm = plt.Normalize(times.min(), times.max())
    transition_norm = (inhale_exhale_transition - times.min()) / (times.max() - times.min())
    colors = [
                (0, 'darkblue'),
                (transition_norm - 0.05, 'blue'),
                (transition_norm - 0.005, 'lightblue'),
                (transition_norm, 'white'),
                (transition_norm + 0.005, 'pink'),
                (transition_norm + 0.05, 'red'),
                (1, 'darkred')
            ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors)
    
    # Convert to mm/s and mm/s² for plotting
    vdotn_mm = vdotn * 1000  # Convert from m/s to mm/s
    adotn_mm = adotn * 1000  # Convert from m/s² to mm/s²
            
            # Plot 1: Total Pressure vs VdotN
    scatter1 = ax1.scatter(vdotn_mm, pressures, c=times, cmap=custom_cmap, norm=norm)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('v⃗·n⃗ (mm/s)')
    ax1.set_ylabel('Total Pressure (Pa)')
    ax1.set_title('Total Pressure vs v⃗·n⃗')
    ax1.grid(True)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
            
            # Plot 2: Total Pressure vs AdotN
    scatter2 = ax2.scatter(adotn_mm, pressures, c=times, cmap=custom_cmap, norm=norm)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('a⃗·n⃗ (mm/s²)')
    ax2.set_ylabel('Total Pressure (Pa)')
    ax2.set_title('Total Pressure vs a⃗·n⃗')
    ax2.grid(True)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
            
            # Plot 3: VdotN vs AdotN (moved from subplot 6)
    scatter3 = ax3.scatter(adotn_mm, vdotn_mm, c=times, cmap=custom_cmap, norm=norm)
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('a⃗·n⃗ (mm/s²)')
    ax3.set_ylabel('v⃗·n⃗ (mm/s)')
    ax3.set_title('v⃗·n⃗ vs a⃗·n⃗')
    ax3.grid(True)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
            
            # Plot 4: dp/dt vs VdotN
    scatter4 = ax4.scatter(vdotn_mm, dpdt, c=times, cmap=custom_cmap, norm=norm)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('v⃗·n⃗ (mm/s)')
    ax4.set_ylabel('dp/dt (Pa/s)')
    ax4.set_title('Pressure Derivative vs v⃗·n⃗')
    ax4.grid(True)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
            
            # Plot 5: dp/dt vs AdotN
    scatter5 = ax5.scatter(adotn_mm, dpdt, c=times, cmap=custom_cmap, norm=norm)
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('a⃗·n⃗ (mm/s²)')
    ax5.set_ylabel('dp/dt (Pa/s)')
    ax5.set_title('Pressure Derivative vs a⃗·n⃗')
    ax5.grid(True)
    cbar5 = plt.colorbar(scatter5, ax=ax5)
    cbar5.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
            
            # Plot 6: Time series plot with flow rate and velocities
    ax6_flow = ax6  # Primary y-axis for flow rate
    ax6_vel = ax6_flow.twinx()  # Secondary y-axis for velocities
            
            # Load and plot flow profile
    base_subject = extract_base_subject(subject_name)
    flow_profile = pd.read_csv(f'{base_subject}FlowProfile.csv')
    
    # Apply smoothing in memory
    def apply_smoothing(data, window_size=20):
        if len(data) < window_size:
            return data.copy()
        smoothed = data.copy()
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
        return smoothed
    
    # Smooth the flow rates for analysis
    flow_profile.iloc[:, 1] = apply_smoothing(flow_profile.iloc[:, 1].values)
    flow_line = ax6_flow.plot(flow_profile['time (s)'], flow_profile['Massflowrate (kg/s)'],
                                    'k-', label='Flow Rate', linewidth=2)
    ax6_flow.set_xlabel('Time (s)')
    ax6_flow.set_ylabel('Mass Flow Rate (kg/s)', color='k')
    ax6_flow.tick_params(axis='y', labelcolor='k')
            
            # Calculate and plot signed velocity
    signed_velocity = velocities * np.sign(vdotn)
    vel_line = ax6_vel.plot(times, signed_velocity, 'b--', 
                                  label='Signed Velocity', linewidth=1.5)
    vdot_line = ax6_vel.plot(times, vdotn, 'r:', 
                                   label='v⃗·n⃗', linewidth=1.5)
            
    ax6_vel.set_ylabel('Velocity (m/s)', color='b')
    ax6_vel.tick_params(axis='y', labelcolor='b')
            
            # Add grid and title
    ax6_flow.grid(True, alpha=0.3)
    ax6_flow.set_title('Flow Rate and Point Velocities')
            
            # Combine legends from both axes
    lines = flow_line + vel_line + vdot_line
    labels = [l.get_label() for l in lines]
    ax6_flow.legend(lines, labels, loc='upper right')
            
            # Add vertical line for inhale-exhale transition
    if len(zero_crossings) >= 2:
        ax6_flow.axvline(x=inhale_exhale_transition, color='k', 
                               linestyle=':', alpha=0.7,
                               label=f'Inhale→Exhale')
            
            # Save to PDF correctly
    if pdf is not None:
            pdf.savefig(fig)
            
            # Also save as PNG in figures directory
    results_dir = Path(f'{subject_name}_results')
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_filename = figures_dir / f"{subject_name}_point_analysis_{description.lower().replace(' ', '_')}_patch{patch_number}_face{face_index}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            
    plt.close()

def create_clean_flow_profile_plot(subject_name, output_dir=None, pdfs_dir=None):
    """
    Create a clean, smoothed flow profile plot showing just one breathing cycle.
    Clips exactly at zero crossings and converts flow rate from kg/s to L/min.
    
    Arguments:
        subject_name: Name of the subject
        output_dir: Directory to save output files (optional)
    """
    print(f"\nGenerating clean flow profile visualization for {subject_name}...")
    
    # Load flow profile data using smart resolution
    from utils.file_processing import find_flow_profile_file, extract_base_subject
    
    # Try to find smoothed flow profile first
    base_subject = extract_base_subject(subject_name)
    smoothed_candidates = [
        f"{subject_name}FlowProfile_smoothed.csv",
        f"{base_subject}FlowProfile_smoothed.csv"
    ]
    
    flow_profile_path = None
    for candidate in smoothed_candidates:
        if Path(candidate).exists():
            flow_profile_path = candidate
            break
    
    if flow_profile_path is None:
        print(f"❌ No smoothed flow profile found for {subject_name}")
        return
    
    print(f"📊 Using flow profile: {flow_profile_path}")
    flow_profile = pd.read_csv(flow_profile_path)
    flow_times = flow_profile['time (s)'].values
    flow_rates = flow_profile['Massflowrate (kg/s)'].values
    
    # Find zero crossings in the flow profile (where flow_rate = 0)
    zero_crossings_idx = np.where(np.diff(np.signbit(flow_rates)))[0]
    
    if len(zero_crossings_idx) < 3:
        print("Warning: Could not find enough zero crossings in flow profile to isolate a clean breathing cycle")
        return
    
    # We need at least 3 zero crossings to define one full breathing cycle
    # 1. Zero crossing: Start of inhale (flow crosses from negative to positive)
    # 2. Zero crossing: End of inhale/start of exhale (flow crosses from positive to negative)
    # 3. Zero crossing: End of exhale/start of next inhale (flow crosses from negative to positive)
    
    # For a clean cycle, clip exactly at the first and third zero crossings
    start_idx = zero_crossings_idx[0]  # First zero crossing
    end_idx = zero_crossings_idx[2]    # Third zero crossing
    
    # Find the exact zero crossing times using interpolation
    zero_times = []
    for idx in [start_idx, zero_crossings_idx[1], end_idx]:
        t0, t1 = flow_times[idx], flow_times[idx + 1]
        v0, v1 = flow_rates[idx], flow_rates[idx + 1]
        
        # Linear interpolation to find t where v = 0
        if v1 != v0:  # Avoid division by zero
            t_cross = t0 - v0 * (t1 - t0) / (v1 - v0)
        else:
            t_cross = t0
        zero_times.append(t_cross)
    
    # Extract the clean cycle (including exactly one full breathing cycle)
    # Use binary search to find the indices closest to our interpolated zero crossing times
    start_idx = np.searchsorted(flow_times, zero_times[0])
    inhale_exhale_idx = np.searchsorted(flow_times, zero_times[1])
    end_idx = np.searchsorted(flow_times, zero_times[2])
    
    clean_times = flow_times[start_idx:end_idx+1]
    clean_rates = flow_rates[start_idx:end_idx+1]
    
    # Apply additional smoothing with a moving average filter
    def moving_average(data, window_size=20):
        """Apply a moving average filter to smooth the data, properly handling edge cases."""
        if len(data) < window_size:
            return data.copy()
            
        # Initialize the output array with original data (will keep edges unchanged)
        smoothed = data.copy()
        
        # Only smooth the points where we have enough data for a full window
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            # Calculate average only where we have a full window
            smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
            
        return smoothed
    
    # Apply smoothing
    smoothed_rates = moving_average(clean_rates, window_size=15)
    
    # Convert from kg/s to L/min
    # Assuming air density of approximately 1.225 kg/m³ at standard conditions
    air_density = 1.225  # kg/m³
    # 1 kg/s ÷ density = m³/s, then × 1000 for L/s, × 60 for L/min
    conversion_factor = (1 / air_density) * 1000 * 60  # kg/s to L/min
    smoothed_rates_lpm = smoothed_rates * conversion_factor
    clean_rates_lpm = clean_rates * conversion_factor
    
    # Renormalize time to start from 0
    normalized_times = clean_times - clean_times[0]
    
    # Get the inhale-exhale transition time
    inhale_exhale_time = zero_times[1] - zero_times[0]
    
    # Create the plot
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Plot the smoothed flow profile
    ax.plot(normalized_times, smoothed_rates_lpm, 'b-', linewidth=2.5, label='Smoothed Flow Rate')
    
    # Add the original data as a lighter line for reference
    ax.plot(normalized_times, clean_rates_lpm, 'lightblue', linewidth=1, alpha=0.5, label='Original Flow Rate')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add a vertical line at the inhale-exhale transition
    ax.axvline(x=inhale_exhale_time, color='r', linestyle='-', linewidth=2, 
               label=f'Inhale→Exhale Transition (t = {inhale_exhale_time:.3f}s)')
    
    # Add inhale/exhale labels at appropriate positions
    inhale_midpoint = inhale_exhale_time / 2
    exhale_midpoint = inhale_exhale_time + (normalized_times[-1] - inhale_exhale_time) / 2
    
    max_rate = np.max(smoothed_rates_lpm)
    min_rate = np.min(smoothed_rates_lpm)
    
    ax.annotate('INHALE', xy=(inhale_midpoint, max_rate/2), xytext=(inhale_midpoint, max_rate*0.7),
                ha='center', va='center', fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    
    ax.annotate('EXHALE', xy=(exhale_midpoint, min_rate/2), xytext=(exhale_midpoint, min_rate*0.7),
                ha='center', va='center', fontsize=14, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # Add labels and title
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flow Rate (L/min)', fontsize=14, fontweight='bold')
    ax.set_title(f'Clean Breathing Cycle Flow Profile - {subject_name}', fontsize=16, fontweight='bold')
    
    # Add a note about the normalized time
    original_start_time = clean_times[0]
    fig.text(0.5, 0.01, f'Note: Time has been normalized to start at 0s. Original data started at {original_start_time:.3f}s.', 
            fontsize=10, ha='center', va='bottom')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Save as a standalone PDF to pdfs directory
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_clean_flow_profile.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_clean_flow_profile.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    
    # If we have an output directory, also save there
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_filename = output_dir / f"{subject_name}_clean_flow_profile.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved PNG: {output_filename}")
    
    plt.close()
    
    return standalone_filename

def detect_available_subjects() -> List[str]:
    """
    Auto-detect available subjects based on existing folder patterns.
    
    Returns:
        List of detected subject names
    """
    current_dir = Path('.')
    subjects = []
    
    # Look for patterns like: SUBJECT_xyz_tables, subject11_xyz_tables, etc.
    xyz_dirs = list(current_dir.glob('*_xyz_tables'))
    
    for xyz_dir in xyz_dirs:
        # Extract subject name by removing '_xyz_tables' suffix
        subject_name = xyz_dir.name.replace('_xyz_tables', '')
        subjects.append(subject_name)
    
    return sorted(subjects)

def auto_detect_subject() -> str:
    """
    Auto-detect the subject to process.
    
    Returns:
        Subject name to process
    """
    subjects = detect_available_subjects()
    
    if not subjects:
        print("❌ No subjects detected! Looking for folders matching pattern: *_xyz_tables")
        print("Available directories:")
        current_dir = Path('.')
        for item in current_dir.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}")
        raise ValueError("No subject folders found")
    
    print(f"🔍 Detected {len(subjects)} available subject(s): {subjects}")
    
    if len(subjects) == 1:
        selected_subject = subjects[0]
        print(f"✅ Auto-selected subject: {selected_subject}")
        return selected_subject
    else:
        # Multiple subjects found, use the first one but inform user
        selected_subject = subjects[0]
        print(f"⚠️  Multiple subjects found. Auto-selected: {selected_subject}")
        print(f"   To specify a different subject, use: --subject {' | '.join(subjects)}")
        return selected_subject

def track_fixed_patch_region_in_file(file_path: Path, point_pairs: list) -> dict:
    """Track a fixed set of points (determined from first time step) in a single time step file.
    Handles both raw CSV files (without Patch Number) and patched CSV files.
    """
    df = pd.read_csv(file_path, low_memory=False)
    
    # Check if we need to add patch numbers (for raw CSV files)
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
        
        # Add Patch Number column
        df['Patch Number'] = patch_numbers
    
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
            'mean': region_df['VdotN'].mean(),
            'std': region_df['VdotN'].std(),
        }
    }
    
    # Get time point from filename using robust extraction
    timestep = extract_timestep_from_filename(file_path)
    
    # Determine time unit based on filename format
    if 'e+' in file_path.stem or 'e-' in file_path.stem or '.' in file_path.stem:
        # Scientific notation or decimal format - likely in seconds
        time_sec = timestep
        time_point = int(timestep * 1000)  # Convert to milliseconds
    else:
        # Integer format - likely in milliseconds
        time_sec = timestep * 0.001  # Convert to seconds
        time_point = int(timestep)  # Already in milliseconds
    
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

# Smoothed flow profile generation functions removed - smoothing is now done in-memory during analysis

def auto_detect_visualization_timestep(subject_name: str) -> Tuple[float, int]:
    """
    Auto-detect the appropriate timestep for visualization using the same logic as patch highlighting.
    
    Returns:
        Tuple of (timestep_value, display_timestep_ms) where:
        - timestep_value: Raw timestep value to use for visualization 
        - display_timestep_ms: Timestep converted to milliseconds for display
    """
    # Check for patched XYZ files first
    xyz_dir = Path(f'{subject_name}_xyz_tables_with_patches')
    if not xyz_dir.exists():
        # Fall back to raw files
        xyz_dir = Path(f'{subject_name}_xyz_tables')
        if not xyz_dir.exists():
            print("⚠️  Warning: No XYZ directory found, using default 100ms")
            return 100.0, 100
    
    # Get available timesteps
    available_files = list(xyz_dir.glob('*XYZ_Internal_Table_table_*.csv'))
    if not available_files:
        print("⚠️  Warning: No XYZ files found, using default 100ms")
        return 100.0, 100
    
    # Sort files in natural chronological order
    timestep_file_pairs = []
    for file_path in available_files:
        try:
            timestep = extract_timestep_from_filename(file_path)
            timestep_file_pairs.append((timestep, file_path))
        except ValueError:
            continue
    
    if not timestep_file_pairs:
        print("⚠️  Warning: No valid timesteps found, using default 100ms")
        return 100.0, 100
    
    # Sort by timestep value (natural chronological order)
    timestep_file_pairs.sort(key=lambda x: x[0])
    sorted_files = [file_path for timestep, file_path in timestep_file_pairs]
    
    # Filter files to only include those within the breathing cycle (same as tracking)
    start_time, end_time = find_breathing_cycle_bounds(subject_name)
    if start_time is not None and end_time is not None:
        filtered_files = filter_xyz_files_by_time(sorted_files, start_time, end_time)
        if filtered_files:
            # Use the first file from the breathing cycle (same as tracking analysis)
            first_file = filtered_files[0]
            visualization_timestep = extract_timestep_from_filename(first_file)
            
            # Convert to milliseconds for display message only
            if 'e+' in first_file.stem or 'e-' in first_file.stem:
                # Scientific notation - likely in seconds
                display_timestep = int(visualization_timestep * 1000)
            else:
                # Likely already in milliseconds
                display_timestep = int(visualization_timestep)
            
            print(f"🎯 Auto-detected timestep {display_timestep}ms for visualization (first file in breathing cycle)")
            return visualization_timestep, display_timestep
        else:
            print("⚠️  Warning: No files found within breathing cycle, using first available file")
    else:
        print("⚠️  Warning: Could not determine breathing cycle bounds, using first available file")
    
    # Use first available file as fallback
    first_timestep = timestep_file_pairs[0][0]
    first_file = timestep_file_pairs[0][1]
    visualization_timestep = first_timestep
    
    # Convert to milliseconds for display message only
    if 'e+' in first_file.stem or 'e-' in first_file.stem:
        display_timestep = int(first_timestep * 1000)
    else:
        display_timestep = int(first_timestep)
    
    print(f"🎯 Auto-detected timestep {display_timestep}ms for visualization (first available file)")
    return visualization_timestep, display_timestep

def main(overwrite_existing: bool = False,
         enable_patch_analysis: bool = True,
         patch_radii: List[float] = None,
         normal_angle_threshold: float = 60.0,
         enable_patch_visualization: bool = True,
         subject_name: str = None,
         raw_dir: str = None,
         highlight_patches: bool = False,
         patch_timestep: int = 100,
         raw_surface: bool = False,
         surface_timestep: int = 1,
         interactive_selector: bool = False,
         selector_timestep: int = 100,
         smoothing_window: int = 20):
    """
    Main function to process CFD data and generate analysis plots.
    """
    # Determine subject name
    if subject_name is None:
        subject_name = auto_detect_subject()
    else:
        # Validate provided subject name
        available_subjects = detect_available_subjects()
        if subject_name not in available_subjects:
            print(f"❌ Subject '{subject_name}' not found!")
            if available_subjects:
                print(f"Available subjects: {available_subjects}")
            else:
                print("No subjects detected in current directory.")
            raise ValueError(f"Subject '{subject_name}' not found")
        print(f"✅ Using specified subject: {subject_name}")
    
    print(f"\n🎯 Processing subject: {subject_name}")
    base_subject = extract_base_subject(subject_name)
    if base_subject != subject_name:
        print(f"🔬 Mesh variant detected: {subject_name} (base: {base_subject})")
    
    # Handle highlight-patches mode
    if highlight_patches:
        # Use consistent timestep detection logic
        if patch_timestep == 100:  # Only auto-detect if using default
            patch_timestep, display_timestep = auto_detect_visualization_timestep(subject_name)
            print(f"\n🎨 Highlight-patches mode: {display_timestep}ms")
        else:
            print(f"\n🎨 Highlight-patches mode: Using specified timestep {patch_timestep}ms")
        
        # Set default patch radii if not provided
        if patch_radii is None:
            patch_radii = [0.001, 0.002, 0.005]  # 1mm, 2mm, 5mm
        
        # Import visualization function
        visualize_patch_regions = None
        if VISUALIZATION_AVAILABLE and VISUALIZE_PATCH_REGIONS_FUNC is not None:
            visualize_patch_regions = VISUALIZE_PATCH_REGIONS_FUNC
        else:
            try:
                from .visualization.patch_visualization import visualize_patch_regions
            except ImportError:
                # Handle direct execution or different import context
                import sys
                import os
                src_dir = os.path.dirname(__file__)
                if str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))
                try:
                    from visualization.patch_visualization import visualize_patch_regions as visualize_func
                except ImportError as e:
                    print("❌ Error: Could not import visualization module")
                    print(f"Import error: {e}")
                    print("Make sure the visualization dependencies are installed")
                    return
        
        # Run visualization
        try:
            fig = visualize_patch_regions(
                subject_name=subject_name,
                time_step=patch_timestep,
                patch_radii=patch_radii,
                use_pipeline_data=True,
                normal_angle_threshold=normal_angle_threshold,
                hdf5_file_path=f"{subject_name}_cfd_data.h5"  # Use HDF5 data instead of CSV
            )
            
            if fig:
                print(f"✅ Patch highlighting completed successfully!")
                print(f"📁 Interactive visualization saved as: {subject_name}_patch_regions_t{patch_timestep}.html")
            else:
                print("❌ Patch highlighting failed!")
                
        except Exception as e:
            print(f"❌ Error during patch highlighting: {e}")
            import traceback
            traceback.print_exc()
        
        return  # Exit early for highlight-patches mode
    
    # Handle raw-surface mode
    if raw_surface:
        print(f"\n🌊 Raw-surface mode: Displaying complete airway surface for manual point selection at timestep {surface_timestep}")
        
        # Check if surface visualization is available
        if not SURFACE_PLOTS_AVAILABLE:
            print("❌ Error: Surface visualization module not available")
            print("Make sure the visualization dependencies are installed")
            return
        
        # Load tracking locations
        tracking_config = load_tracking_locations(subject_name=subject_name)
        tracking_locations = tracking_config['locations']
        
        # Import enhanced file resolution
        from utils.file_processing import find_closest_xyz_file, get_xyz_file_info
        
        # Find the XYZ file for the specified timestep using smart resolution
        xyz_file = find_closest_xyz_file(subject_name, surface_timestep)
        
        if xyz_file is None:
            print(f"❌ No XYZ file found for timestep {surface_timestep}ms")
            
            # Show available files and file info
            file_info = get_xyz_file_info(subject_name)
            if file_info['files_found'] > 0:
                print(f"📁 Found {file_info['files_found']} files in {file_info['directory']}")
                print(f"🕐 Time unit: {file_info['time_unit']}")
                print(f"📝 Naming convention: {file_info['naming_convention']}")
                print(f"⏱️  Available time range: {file_info['time_range']}")
                
                # Show sample files
                xyz_dir = Path(file_info['directory'])
                available_files = list(xyz_dir.glob('*XYZ_Internal_Table_table_*.csv'))
                print("Available files:")
                for f in sorted(available_files)[:10]:  # Show first 10
                    print(f"   - {f.name}")
                if len(available_files) > 10:
                    print(f"   ... and {len(available_files) - 10} more files")
                    
                # Suggest appropriate timestep
                if file_info['time_unit'] == 's':
                    suggested_timestep = int(file_info['timesteps'][0] * 1000) if file_info['timesteps'] else 250
                    print(f"💡 Try: --surface-timestep {suggested_timestep}")
                else:
                    suggested_timestep = int(file_info['timesteps'][0]) if file_info['timesteps'] else 1
                    print(f"💡 Try: --surface-timestep {suggested_timestep}")
            else:
                print("   No XYZ files found")
            return
        
        print(f"📁 Using XYZ file: {xyz_file}")
        xyz_dir = xyz_file.parent
        
        # Create results directory
        results_dir = Path(f'{subject_name}_results')
        interactive_dir = results_dir / 'interactive'
        interactive_dir.mkdir(parents=True, exist_ok=True)
        
        # Run surface visualization
        try:
            print(f"📁 Loading data from: {xyz_file}")
            plot_3d_interactive_all_patches(xyz_file, tracking_locations, subject_name, interactive_dir)
            
            print(f"✅ Raw surface visualization completed successfully!")
            print(f"📁 Use the interactive visualization to manually select new tracking points")
            print(f"📁 Interactive surface saved as: {subject_name}_surface_patches_interactive.html")
            print(f"📁 Also saved to: {interactive_dir}/{subject_name}_surface_patches_interactive.html")
                
        except Exception as e:
            print(f"❌ Error during raw surface visualization: {e}")
            import traceback
            traceback.print_exc()
        
        return  # Exit early for raw-surface mode
    
    # Handle interactive selector mode
    if interactive_selector:
        print(f"\n🎯 Interactive Selector Mode: Launching 3D point selector for timestep {selector_timestep}")
        
        # Import the interactive selector
        try:
            from .visualization.interactive_point_selector import run_interactive_selector
        except ImportError:
            # Handle direct execution or different import context
            import sys
            import os
            src_dir = os.path.dirname(__file__)
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            try:
                from visualization.interactive_point_selector import run_interactive_selector
            except ImportError as e:
                print(f"❌ Error importing interactive selector: {e}")
                print("💡 Make sure PyVista is installed: pip install pyvista")
                return
        
        # Launch the interactive selector
        selected_points = run_interactive_selector(subject_name, selector_timestep)
        
        print(f"✅ Interactive selector completed!")
        if selected_points:
            print(f"📊 {len(selected_points)} points were selected and saved to tracking locations")
        else:
            print("📊 No points were selected")
        
        return  # Exit early for interactive selector mode
    
    # Import smart file resolution functions
    from utils.file_processing import (
        find_flow_profile_file, 
        validate_subject_files, 
        create_variant_tracking_locations
    )
    
    # For mesh variants, always create a subject-specific tracking locations file
    if base_subject != subject_name:
        print(f"📋 Creating subject-specific tracking locations file...")
        
        success = create_variant_tracking_locations(subject_name, force_create=False)
        if success:
            print(f"✅ Created: {subject_name}_tracking_locations.json")
            print(f"⚠️  IMPORTANT: Update patch numbers using interactive visualization:")
            print(f"   python src/main.py --raw-surface --subject {subject_name}")
        else:
            print(f"❌ Failed to create tracking locations file for {subject_name}")
    
    # Verify required files exist using smart resolution
    validation_results = validate_subject_files(subject_name)
    
    if not all(validation_results.values()):
        missing = [k for k, v in validation_results.items() if not v]
        print(f"❌ Missing required files for subject {subject_name}: {missing}")
        raise FileNotFoundError(f"Missing required files for subject {subject_name}")
    
    print(f"✅ All required files found for {subject_name}")
    
    # Note: Flow profile smoothing is now done in-memory during analysis
    print(f"\n✅ Flow profile smoothing will be applied in-memory during analysis")
    
    # Create organized results directory structure
    results_dir = Path(f'{subject_name}_results')
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different types of outputs
    output_dir = results_dir / 'tracked_points'  # CSV data files
    figures_dir = results_dir / 'figures'        # PNG images
    pdfs_dir = results_dir / 'reports'           # PDF reports
    interactive_dir = results_dir / 'interactive' # HTML files
    
    # Create all subdirectories
    for dir_path in [output_dir, figures_dir, pdfs_dir, interactive_dir]:
        dir_path.mkdir(exist_ok=True)
    
    print(f"Results will be saved in: {results_dir}")
    print(f"  - CSV data: {output_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - PDF reports: {pdfs_dir}")
    print(f"  - Interactive files: {interactive_dir}")
    
    # Check if we need to process the data
    tracking_config = load_tracking_locations(subject_name=subject_name)
    tracking_locations = tracking_config['locations']  # Extract just the locations for compatibility
    all_files_exist = True
    
    # Check if tracked point files exist (single point only)
    for location in tracking_locations:
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        single_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
        if not single_file.exists():
            all_files_exist = False
            break
    
    # Set HDF5 file path before processing (needed for visualization)
    hdf5_file_path = f"{subject_name}_cfd_data.h5"
    
    # Only process raw data if needed
    if overwrite_existing or not all_files_exist:
        
        if Path(hdf5_file_path).exists() and not overwrite_existing:
            print(f"\n🚀 Found existing HDF5 cache: {hdf5_file_path}")
            print("Using cached data for fast tracking (skipping CSV processing)...")
            
            # Load the existing HDF5 data info
            from data_processing.trajectory import auto_select_csv_to_hdf5_method
            data_info = {'file_path': hdf5_file_path, 'properties': None}
            
            # Skip CSV processing entirely when using HDF5 cache
            # No need for patched CSV files - HDF5 contains all the data
            xyz_files = []  # Empty list - using HDF5 directly
            
            # No CSV files needed - HDF5 contains all data already filtered
            print("✅ Using HDF5 cache directly (no CSV files needed)")
            xyz_files = []
        
        else:
            # Original CSV processing path
            # Try to get patched XYZ table files first
            xyz_dir = Path(f'{subject_name}_xyz_tables_with_patches')
            xyz_files = list(xyz_dir.glob('patched_XYZ_Internal_Table_table_*.csv'))
            
            # If no patched files found, look for raw files and process them
            if not xyz_files:
                print(f"No patched XYZ table files found in {xyz_dir}")
                print("Looking for raw XYZ table files...")
                
                # Determine raw directory name
                if raw_dir is not None:
                    raw_xyz_dir = Path(raw_dir)
                    print(f"Using custom raw directory: {raw_dir}")
                else:
                    raw_xyz_dir = Path(f'{subject_name}_xyz_tables')
                    print(f"Using default raw directory: {raw_xyz_dir}")
                
                raw_xyz_files = list(raw_xyz_dir.glob('XYZ_Internal_Table_table_*.csv'))
                
                if raw_xyz_files:
                    print(f"Found {len(raw_xyz_files)} raw XYZ files")
                    print(f"⚠️  SKIPPING patched CSV creation (saving disk space - using HDF5 instead)")
                    # Sort raw files using natural chronological order based on timestep values
                    timestep_file_pairs = []
                    for file_path in raw_xyz_files:
                        try:
                            timestep = extract_timestep_from_filename(file_path)
                            timestep_file_pairs.append((timestep, file_path))
                        except ValueError:
                            print(f"Warning: Could not parse timestep from {file_path.name}")
                            continue
                    
                    # Sort by timestep value (natural chronological order)
                    timestep_file_pairs.sort(key=lambda x: x[0])
                    xyz_files = [file_path for timestep, file_path in timestep_file_pairs]
                    
                    # Use raw files directly for HDF5 conversion (skip patched CSV creation)
                    print(f"✅ Using {len(xyz_files)} raw files directly for HDF5 conversion")
                else:
                    if raw_dir is not None:
                        print(f"No XYZ table files found in custom directory {raw_dir} or patched directory {xyz_dir}")
                    else:
                        print(f"No XYZ table files found in either {xyz_dir} or {raw_xyz_dir}")
                    return
            else:
                print(f"Found {len(xyz_files)} existing patched XYZ table files")
            
            # Sort using natural chronological order based on timestep values
            timestep_file_pairs = []
            for file_path in xyz_files:
                try:
                    timestep = extract_timestep_from_filename(file_path)
                    timestep_file_pairs.append((timestep, file_path))
                except ValueError:
                    print(f"Warning: Could not parse timestep from {file_path.name}")
                    continue
            
            # Sort by timestep value (natural chronological order)
            timestep_file_pairs.sort(key=lambda x: x[0])
            xyz_files = [file_path for timestep, file_path in timestep_file_pairs]
            
            print(f"Found {len(xyz_files)} XYZ table files")
            
            # Find breathing cycle bounds from flow profile
            start_time, end_time = find_breathing_cycle_bounds(subject_name)
            if start_time is None or end_time is None:
                print("Error: Could not determine breathing cycle bounds!")
                return
            
            # Filter files to only include those within the breathing cycle
            xyz_files = filter_xyz_files_by_time(xyz_files, start_time, end_time)
            if not xyz_files:
                print("Error: No files found within breathing cycle bounds!")
                return
            
            # STEP 3: Convert CSV files to HDF5 format for faster processing (PARALLEL)
            print(f"\n🚀 Converting CSV files to HDF5 format for faster processing (PARALLEL)...")
            from data_processing.trajectory import auto_select_csv_to_hdf5_method
            
            # Convert to HDF5 using optimized parallel method (respects --forcererun flag)
            data_info = auto_select_csv_to_hdf5_method(xyz_files, hdf5_file_path, overwrite_existing)
            print(f"✅ HDF5 data ready: {data_info['file_path']}")
        
        # Note: For now, we'll continue using CSV files for tracking
        # In the future, we can modify the tracking functions to use HDF5 directly
        
        # Create interactive visualization using consistent timestep detection
        print("\nCreating interactive visualization...")
        
        # Use the same timestep detection as patch highlighting
        try:
            visualization_timestep, display_timestep = auto_detect_visualization_timestep(subject_name)
            print(f"Using consistent timestep: {display_timestep}ms")
            
            # Try HDF5 first with the detected timestep
            if Path(hdf5_file_path).exists():
                print(f"📊 Using HDF5 data source: {hdf5_file_path}")
                plot_3d_interactive_all_patches(hdf5_file_path, tracking_locations, subject_name, interactive_dir, time_point=display_timestep)
            elif xyz_files:
                print("📁 Using CSV data source")
                plot_3d_interactive_all_patches(xyz_files[0], tracking_locations, subject_name, interactive_dir)
            else:
                print("❌ No data source available for interactive visualization")
        except Exception as e:
            print(f"❌ Error creating interactive visualization: {e}")
            # Fallback to CSV if available
            if xyz_files:
                print("📁 Falling back to CSV for interactive visualization...")
                plot_3d_interactive_all_patches(xyz_files[0], tracking_locations, subject_name, interactive_dir)
        
        # Process each tracking point
        print("\nProcessing tracking points...")
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            print(f"\nProcessing {description}")
            print(f"Tracking point: Patch {patch_number}, Face {face_index}")
            
            # Process single point
            print("\nProcessing single point...")
            
            # Check if we should use HDF5 cache or CSV files
            if Path(hdf5_file_path).exists() and len(xyz_files) == 0:
                # Use HDF5 cache for tracking (much faster) with auto-selection
                single_point_data = auto_select_hdf5_point_tracking_method(hdf5_file_path, patch_number, face_index)
            else:
                # Use parallel processing for single point tracking from CSV
                single_point_data = track_point_parallel(xyz_files, patch_number, face_index)
            
            if single_point_data:
                print(f"Tracked single point through {len(single_point_data)} time steps")
                save_trajectory_data(single_point_data, subject_name, patch_number, face_index, description, is_region=False)
            else:
                print(f"Warning: No trajectory found for single point")
            
            # Process patch regions if enabled
            if enable_patch_analysis:
                if patch_radii is None:
                    patch_radii = [0.002]  # Default: 2mm only (faster processing)
                
                print(f"\nProcessing patch regions...")
                print(f"Patch radii: {[f'{r*1000:.1f}mm' for r in patch_radii]}")
                print(f"Normal angle threshold: {normal_angle_threshold}° (surface normal filtering)")
                
                for radius in patch_radii:
                    radius_mm = radius * 1000
                    print(f"\n  Processing {radius_mm:.1f}mm patch...")
                    
                    # STEP 1: Find patch points in FIRST time step
                    # Try HDF5 first, then fall back to CSV
                    if Path(hdf5_file_path).exists():
                        print(f"  🚀 Using HDF5 cache for spatial analysis: {hdf5_file_path}")
                        print(f"  Finding patch points using HDF5 data...")
                        patch_point_pairs = find_initial_region_points_hdf5_safe(hdf5_file_path, patch_number, face_index, radius, normal_angle_threshold)
                    else:
                        print(f"  📁 Using CSV files for spatial analysis (HDF5 not available)")
                        if len(xyz_files) == 0:
                            print(f"  Warning: No CSV files available for patch analysis")
                            continue
                        first_file = xyz_files[0]
                        print(f"  Finding patch points in first time step: {first_file.name}")
                        patch_point_pairs = find_initial_region_points(first_file, patch_number, face_index, radius, normal_angle_threshold)
                    
                    if not patch_point_pairs:
                        print(f"  Warning: No points found in {radius_mm:.1f}mm patch")
                        continue
                    
                    print(f"  Found {len(patch_point_pairs)} points in {radius_mm:.1f}mm patch")
                    
                    # STEP 2: Track these SAME points through all time steps
                    # Use HDF5 or CSV based on availability
                    if Path(hdf5_file_path).exists():
                        print(f"  🚀 Using HDF5 cache for patch tracking (10-20x faster)")
                        patch_data = auto_select_hdf5_tracking_method(hdf5_file_path, patch_point_pairs)
                    else:
                        print(f"  📁 Using CSV files for patch tracking")
                        patch_data = track_fixed_patch_region_csv_parallel(xyz_files, patch_point_pairs)
                    
                    if patch_data:
                        print(f"  Tracked {radius_mm:.1f}mm patch through {len(patch_data)} time steps")
                        # Save with modified description to indicate patch analysis
                        patch_description = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                        save_trajectory_data(patch_data, subject_name, patch_number, face_index, patch_description, is_region=True)
                    else:
                        print(f"  Warning: No trajectory found for {radius_mm:.1f}mm patch")
        
        # Process combinations after individual points are done
        print("\nProcessing combinations...")
        combination_files = process_all_combinations(subject_name, tracking_config)
        if combination_files:
            print(f"Successfully processed {len(combination_files)} combinations")
        
    else:
        print("\nUsing existing tracked point data...")
        # Still process combinations if they don't exist
        combination_files = process_all_combinations(subject_name, tracking_config)
    
    # Extract key time points for visualization
    print("\nExtracting key time points for visualization...")
    
    # Load and smooth flow profile to find inhale-exhale transition
    # Use base subject name for flow profile files
    base_subject = extract_base_subject(subject_name)
    flow_profile = pd.read_csv(f'{base_subject}FlowProfile.csv')
    
    # Apply smoothing in memory (same as moving_average function)
    def apply_smoothing(data, window_size=20):
        if len(data) < window_size:
            return data.copy()
        smoothed = data.copy()
        half_window = window_size // 2
        for i in range(half_window, len(data) - half_window):
            smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
        return smoothed
    
    # Smooth the flow rates for analysis
    flow_profile.iloc[:, 1] = apply_smoothing(flow_profile.iloc[:, 1].values)
    flow_times = flow_profile['time (s)'].values
    flow_rates = flow_profile['Massflowrate (kg/s)'].values
    
    # Find zero crossings in the flow profile (where flow_rate = 0)
    zero_crossings_idx = np.where(np.diff(np.signbit(flow_rates)))[0]
    
    if len(zero_crossings_idx) >= 2:
        inhale_exhale_transition = flow_times[zero_crossings_idx[1]]
        print(f"Inhale to exhale transition detected at {inhale_exhale_transition:.3f} seconds")
    else:
        print("Warning: Could not find inhale-exhale transition in flow profile")
        inhale_exhale_transition = None
    
    # Find the posterior border of soft palate data
    soft_palate_data = None
    for location in tracking_locations:
        if location['description'] == 'Posterior border of soft palate':
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if data_file.exists():
                soft_palate_data = pd.read_csv(data_file)
                print(f"Loaded data for posterior border of soft palate")
            else:
                print(f"Warning: No data file found for posterior border of soft palate")
            break
    
    # Initialize variables to store key time points
    key_time_points = {
        'inhalation_max_negative_adotn': None,
        'exhalation_max_pressure': None
    }
    
    # Find time point during inhalation with maximum negative AdotN at the posterior border of the soft palate
    if soft_palate_data is not None and inhale_exhale_transition is not None:
        # Calculate AdotN from VdotN
        times = soft_palate_data['Time (s)'].values
        vdotn = soft_palate_data['VdotN'].values * 1000  # Convert from m/s to mm/s
        
        # Calculate acceleration using forward difference
        dt = np.diff(times)
        
        # Check for zero time differences and fix them
        zero_dt_mask = dt == 0
        if np.any(zero_dt_mask):
            print(f"Warning: Found {np.sum(zero_dt_mask)} zero time differences in key time points calculation. Fixing with interpolation.")
            # Replace zero time differences with the mean of non-zero values
            non_zero_dt = dt[~zero_dt_mask]
            if len(non_zero_dt) > 0:
                mean_dt = np.mean(non_zero_dt)
                dt[zero_dt_mask] = mean_dt
            else:
                dt = np.full_like(dt, 0.001)  # Default 1ms timestep
        
        dvdotn = np.diff(vdotn)
        adotn = np.append(dvdotn / dt, dvdotn[-1] / dt[-1])  # Add last point to match array length
        
        # Find inhalation period (before inhale-exhale transition)
        inhalation_mask = times < inhale_exhale_transition
        
        if np.any(inhalation_mask):
            inhalation_times = times[inhalation_mask]
            inhalation_adotn = adotn[inhalation_mask]
            
            # Find the time point with maximum negative AdotN during inhalation
            min_adotn_idx = np.argmin(inhalation_adotn)
            max_negative_adotn_time = inhalation_times[min_adotn_idx]
            max_negative_adotn_value = inhalation_adotn[min_adotn_idx]
            
            print(f"Found maximum negative AdotN during inhalation at t={max_negative_adotn_time:.3f}s with value {max_negative_adotn_value:.6f} m/s²")
            
            # Extract the time point number from the filename format
            time_point = int(max_negative_adotn_time * 1000)  # Convert to milliseconds
            key_time_points['inhalation_max_negative_adotn'] = time_point
    
    # Find time point during exhalation with maximum total pressure
    # Load data for all points to find maximum pressure during exhalation
    max_pressure = float('-inf')
    max_pressure_time = None
    
    for location in tracking_locations:
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
        
        if data_file.exists():
            df = pd.read_csv(data_file)
            
            # Find exhalation period (after inhale-exhale transition)
            if inhale_exhale_transition is not None:
                exhalation_mask = df['Time (s)'] > inhale_exhale_transition
                
                if np.any(exhalation_mask):
                    exhalation_times = df['Time (s)'][exhalation_mask].values
                    exhalation_pressure = df['Total Pressure (Pa)'][exhalation_mask].values
                    
                    # Find the time point with maximum pressure during exhalation
                    max_pressure_idx = np.argmax(exhalation_pressure)
                    current_max_pressure = exhalation_pressure[max_pressure_idx]
                    current_max_time = exhalation_times[max_pressure_idx]
                    
                    if current_max_pressure > max_pressure:
                        max_pressure = current_max_pressure
                        max_pressure_time = current_max_time
    
    if max_pressure_time is not None:
        print(f"Found maximum total pressure during exhalation at t={max_pressure_time:.3f}s with value {max_pressure:.2f} Pa")
        
        # Extract the time point number from the filename format
        time_point = int(max_pressure_time * 1000)  # Convert to milliseconds
        key_time_points['exhalation_max_pressure'] = time_point
    
    # Save the key time points to a JSON file
    time_points_file = results_dir / f'{subject_name}_key_time_points.json'
    with open(time_points_file, 'w') as f:
        json.dump(key_time_points, f, indent=4)
    
    print(f"Saved key time points to {time_points_file}")
    
    # Create plots for each tracking point
    print("\nGenerating plots...")
    
    # Create separate PDF for airway surface velocity plots
    velocity_pdf_name = pdfs_dir / f'{subject_name}_airway_surface_velocity.pdf'
    print(f"\nGenerating {velocity_pdf_name}...")
    
    with PdfPages(velocity_pdf_name) as velocity_pdf:
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            # Plot single point data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if data_file.exists():
                print(f"\nPlotting velocity data for {description} (single point)")
                df = pd.read_csv(data_file)
                create_airway_surface_velocity_plot(df, subject_name, description, patch_number, face_index, velocity_pdf)
            else:
                print(f"Warning: No single point data file found for {description}")
            
            # Plot patch data if enabled
            if enable_patch_analysis:
                if patch_radii is None:
                    patch_radii = [0.001, 0.002, 0.005]
                
                for radius in patch_radii:
                    radius_mm = radius * 1000
                    patch_description = f"{description} (Patch {radius_mm:.1f}mm)"
                    # Use the actual filename format that was saved
                    patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                    patch_data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv"
                    
                    if patch_data_file.exists():
                        print(f"Plotting velocity data for {patch_description}")
                        df = pd.read_csv(patch_data_file)
                        create_airway_surface_velocity_plot(df, subject_name, patch_description, patch_number, face_index, velocity_pdf)
                    else:
                        print(f"Warning: No patch data file found for {patch_description}")
    
    print(f"Generated {velocity_pdf_name}")
    
    # Create separate PDFs for 50ms and 100ms window analyses
    correlation_pdf_50ms = pdfs_dir / f'{subject_name}_pressure_motion_correlation_50ms.pdf'
    correlation_pdf_100ms = pdfs_dir / f'{subject_name}_pressure_motion_correlation_100ms.pdf'
    print(f"\nGenerating correlation analysis PDFs...")
    
    with PdfPages(correlation_pdf_50ms) as pdf_50ms, PdfPages(correlation_pdf_100ms) as pdf_100ms:
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            # Plot single point data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if data_file.exists():
                print(f"\nAnalyzing correlations for {description} (single point)")
                df = pd.read_csv(data_file)
                create_correlation_analysis_plot(df, subject_name, description, patch_number, face_index, pdf_50ms, pdf_100ms)
            else:
                print(f"Warning: No single point data file found for {description}")
            
            # Plot patch data if enabled
            if enable_patch_analysis:
                if patch_radii is None:
                    patch_radii = [0.001, 0.002, 0.005]
                
                for radius in patch_radii:
                    radius_mm = radius * 1000
                    patch_description = f"{description} (Patch {radius_mm:.1f}mm)"
                    # Use the actual filename format that was saved
                    patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                    patch_data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv"
                    
                    if patch_data_file.exists():
                        print(f"Analyzing correlations for {patch_description}")
                        df = pd.read_csv(patch_data_file)
                        create_correlation_analysis_plot(df, subject_name, patch_description, patch_number, face_index, pdf_50ms, pdf_100ms)
                    else:
                        print(f"Warning: No patch data file found for {patch_description}")
    
    print(f"Generated {correlation_pdf_50ms} and {correlation_pdf_100ms}")
    
    # Create dp/dt vs da/dt analysis PDF
    dpdt_vs_dadt_pdf_name = pdfs_dir / f'{subject_name}_dpdt_vs_dadt_analysis.pdf'
    print(f"\nGenerating dp/dt vs da/dt analysis PDF...")
    
    with PdfPages(dpdt_vs_dadt_pdf_name) as dpdt_dadt_pdf:
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            # Plot single point data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if data_file.exists():
                print(f"\nAnalyzing dp/dt vs da/dt for {description} (single point)")
                df = pd.read_csv(data_file)
                create_dpdt_vs_dadt_plot(df, subject_name, description, patch_number, face_index, dpdt_dadt_pdf)
            else:
                print(f"Warning: No single point data file found for {description}")
            
            # Plot patch data if enabled
            if enable_patch_analysis:
                if patch_radii is None:
                    patch_radii = [0.001, 0.002, 0.005]
                
                for radius in patch_radii:
                    radius_mm = radius * 1000
                    patch_description = f"{description} (Patch {radius_mm:.1f}mm)"
                    # Use the actual filename format that was saved
                    patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                    patch_data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv"
                    
                    if patch_data_file.exists():
                        print(f"Analyzing dp/dt vs da/dt for {patch_description}")
                        df = pd.read_csv(patch_data_file)
                        create_dpdt_vs_dadt_plot(df, subject_name, patch_description, patch_number, face_index, dpdt_dadt_pdf)
                    else:
                        print(f"Warning: No patch data file found for {patch_description}")
    
    # Create metrics vs time analysis PDF
    metrics_vs_time_pdf_name = pdfs_dir / f'{subject_name}_metrics_vs_time_analysis.pdf'
    print(f"\nGenerating metrics vs time analysis PDF...")
    
    with PdfPages(metrics_vs_time_pdf_name) as metrics_vs_time_pdf:
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            # Plot single point data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if data_file.exists():
                print(f"\nCreating metrics vs time plot for {description} (single point)")
                df = pd.read_csv(data_file)
                create_metrics_vs_time_plot(df, subject_name, description, patch_number, face_index, metrics_vs_time_pdf)
            else:
                print(f"Warning: No single point data file found for {description}")
            
            # Plot patch data if enabled
            if enable_patch_analysis:
                if patch_radii is None:
                    patch_radii = [0.001, 0.002, 0.005]
                
                for radius in patch_radii:
                    radius_mm = radius * 1000
                    patch_description = f"{description} (Patch {radius_mm:.1f}mm)"
                    # Use the actual filename format that was saved
                    patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                    patch_data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv"
                    
                    if patch_data_file.exists():
                        print(f"Creating metrics vs time plot for {patch_description}")
                        df = pd.read_csv(patch_data_file)
                        create_metrics_vs_time_plot(df, subject_name, patch_description, patch_number, face_index, metrics_vs_time_pdf)
                    else:
                        print(f"Warning: No patch data file found for {patch_description}")
    
    # Generate comprehensive CFD analysis PDF
    pdf_name = pdfs_dir / f'{subject_name}_cfd_analysis.pdf'
    print(f"\nGenerating {pdf_name}...")
    
    with PdfPages(pdf_name) as pdf:
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            # Plot single point data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if data_file.exists():
                print(f"\nPlotting data for {description} (single point)")
                df = pd.read_csv(data_file)
                create_airway_surface_analysis_plot(df, subject_name, description, 
                                                   patch_number, face_index, pdf, output_dir, smoothing_window)
            else:
                print(f"Warning: No single point data file found for {description}")
            
            # Plot patch data if enabled
            if enable_patch_analysis:
                if patch_radii is None:
                    patch_radii = [0.001, 0.002, 0.005]
                
                for radius in patch_radii:
                    radius_mm = radius * 1000
                    patch_description = f"{description} (Patch {radius_mm:.1f}mm)"
                    # Use the actual filename format that was saved
                    patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                    patch_data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv"
                    
                    if patch_data_file.exists():
                        print(f"Plotting data for {patch_description}")
                        df = pd.read_csv(patch_data_file)
                        create_airway_surface_analysis_plot(df, subject_name, patch_description, 
                                                           patch_number, face_index, pdf, output_dir, smoothing_window)
                    else:
                        print(f"Warning: No patch data file found for {patch_description}")
        
        # Load data for all points to create the symmetric comparison panel
        print("\nGenerating symmetric comparison panel...")
        dfs = {}
        location_map = {
            'Posterior border of soft palate': 'posterior_border_of_soft_palate',
            'Superior border of epiglottis': 'superior_border_of_epiglottis',
            'Back of tongue': 'back_of_tongue',
            'Anterior vocal fold': 'anterior_vocal_fold'
        }
        
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            # Skip anterior vocal fold for this plot
            if description == 'Anterior vocal fold':
                continue
                
            # Load trajectory data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
            
            if not data_file.exists():
                print(f"Warning: No data file found for {description}")
                continue
                
            df = pd.read_csv(data_file)
            dfs[location_map[description]] = df
        
        # Create symmetric comparison panel if we have all the required data
        if len(dfs) == 3:
            print(f"Creating symmetric comparison panels for {subject_name}...")
            # 1. Clean version (no markers)
            create_symmetric_comparison_panel_clean(dfs, subject_name, pdf, pdfs_dir)
            
            # 2. Detailed version (with zero-crossing markers)
            create_symmetric_comparison_panel(dfs, subject_name, pdf, pdfs_dir)
            
            # 3. Smoothed version (with moving average)
            create_symmetric_comparison_panel_smooth(dfs, subject_name, pdf, pdfs_dir)
            
            # 4. Smoothed version with zero-crossing markers
            create_symmetric_comparison_panel_smooth_with_markers(dfs, subject_name, pdf, pdfs_dir)
            
            print(f"Symmetric comparison panels completed.")
            
            # 5. CFD Analysis 3x3 panels (new)
            if CFD_ANALYSIS_AVAILABLE:
                print(f"Creating CFD Analysis 3x3 panels for {subject_name}...")
                
                # Load both single point and patch data for CFD analysis
                single_point_dfs, patch_dfs = load_cfd_data_for_analysis(
                    subject_name=subject_name,
                    output_dir=output_dir,
                    enable_patch_analysis=enable_patch_analysis,
                    patch_radii=patch_radii
                )
                
                # Create CFD analysis panels if we have data
                if single_point_dfs or patch_dfs:
                    # Create CFD analysis 3x3 panel (smoothed, no markers)
                    create_cfd_analysis_3x3_panel(single_point_dfs, patch_dfs, subject_name, pdf, pdfs_dir)
                    
                    # Create CFD analysis 3x3 panel with markers (smoothed, with markers)
                    create_cfd_analysis_3x3_panel_with_markers(single_point_dfs, patch_dfs, subject_name, pdf, pdfs_dir)
                    
                    print(f"CFD Analysis 3x3 panels completed.")
                else:
                    print(f"Warning: No CFD data found for analysis")
            else:
                print(f"Warning: CFD analysis functions not available")
        else:
            print(f"Warning: Missing data for some locations, symmetric comparison panel not generated")
    
    print("\nAnalysis complete!")
    print(f"Generated files:")
    print(f"- {pdf_name}: Contains single point analysis plots")
    print(f"- Individual PNG files in {results_dir}/figures/ directory")
    print(f"- Individual CSV files in {results_dir}/tracked_points/ directory")
    
    # Generate the clean flow profile visualization
    create_clean_flow_profile_plot(subject_name, figures_dir, pdfs_dir)
    
    # Generate interactive 3D patch visualization
    if enable_patch_visualization:
        print("\n🎯 Generating interactive 3D patch visualization...")
        try:
            # Import visualization function with fallback
            visualize_func = None
            if VISUALIZATION_AVAILABLE and VISUALIZE_PATCH_REGIONS_FUNC is not None:
                # Use the already imported function
                visualize_func = VISUALIZE_PATCH_REGIONS_FUNC
            else:
                # Try to import again with different approaches
                try:
                    from visualization.patch_visualization import visualize_patch_regions as visualize_func
                except ImportError:
                    try:
                        # Add src to path and try again
                        import sys
                        import os
                        src_dir = os.path.dirname(__file__)
                        if str(src_dir) not in sys.path:
                            sys.path.insert(0, str(src_dir))
                        from visualization.patch_visualization import visualize_patch_regions as visualize_func
                    except ImportError as e:
                        print(f"⚠️  Warning: Could not import patch visualization: {e}")
                        print("📊 Patch visualization disabled due to import error")
                        visualize_func = None
            
            if visualize_func is not None:
                # Use consistent timestep detection logic
                visualization_timestep, display_timestep = auto_detect_visualization_timestep(subject_name)
                
                # Set patch radii if not provided
                if patch_radii is None:
                    patch_radii = [0.001, 0.002, 0.005]
                
                fig = visualize_func(
                    subject_name=subject_name,
                    time_step=visualization_timestep,
                    patch_radii=patch_radii,
                    use_pipeline_data=True,  # Use pre-filtered data from pipeline
                    normal_angle_threshold=normal_angle_threshold,
                    output_dir=str(interactive_dir),
                    hdf5_file_path=hdf5_file_path  # Use HDF5 data instead of CSV
                )
                
                if fig:
                    print(f"✅ Interactive patch visualization saved to: {interactive_dir}")
                else:
                    print("⚠️  Warning: Failed to generate patch visualization")
            else:
                print("⚠️  Warning: Patch visualization function not available")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not generate patch visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n📊 Patch visualization disabled (use --enablevisualization to enable)")
    
    print("\nAll outputs saved in organized directory structure:")
    print(f"📁 {results_dir}/")
    print(f"   ├── 📁 tracked_points/     (CSV data files)")
    print(f"   ├── 📁 figures/           (PNG images)")
    print(f"   ├── 📁 reports/           (PDF reports)")
    print(f"   ├── 📁 interactive/       (HTML files)")
    print(f"   └── 📄 {subject_name}_key_time_points.json")
    print("\n✅ Analysis pipeline completed successfully!")

if __name__ == '__main__':
    # Create custom help formatter for better formatting
    class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _format_action_invocation(self, action):
            if not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                metavar, = self._metavar_formatter(action, default)(1)
                return metavar
            else:
                parts = []
                if action.nargs == 0:
                    parts.extend(action.option_strings)
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    for option_string in action.option_strings:
                        parts.append('%s %s' % (option_string, args_string))
                return ', '.join(parts)

    description = '''CFD Data Analysis Pipeline with Surface Normal Filtering

This pipeline processes CFD airway data to track anatomical points and analyze
pressure, velocity, and acceleration relationships during breathing cycles.

DEMO USAGE SCENARIOS:

1. 🚀 COMPLETE ANALYSIS FROM SCRATCH:
   python src/main.py --subject OSAMRI007
   
   → Processes all data, generates PDFs, creates interactive visualizations
   → Outputs: CSV data, PDF reports, PNG figures, HTML visualizations
   → Time: ~5-10 minutes for full analysis

2. 🔄 FORCE COMPLETE RERUN (overwrite existing data):
   python src/main.py --subject OSAMRI007 --forcererun
   
3. 📁 CUSTOM DATA DIRECTORY:
   python src/main.py --subject OSAMRI007 --datadir /path/to/cfd/data
   
   → Process data from a different directory (useful for server deployments)
   → All file paths will be relative to the specified data directory
   → Outputs will be created in the data directory
   
   → Reprocesses everything even if files exist
   → Use when you want fresh analysis or changed parameters
   → Subject ID is required (no auto-detection with --forcererun)

3. 🎨 HIGHLIGHT PATCH REGIONS:
   python src/main.py --subject OSAMRI007 --highlight-patches --patch-timestep 100
   
   → Quick visualization of patch regions around tracking points (100 = 0.1s)
   → Shows 1mm, 2mm, 5mm patches around anatomical landmarks
   → Time: ~30 seconds

4. 🌊 RAW SURFACE VISUALIZATION:
   python src/main.py --subject OSAMRI007 --raw-surface --surface-timestep 50
   
   → Shows complete raw airway surface for manual point selection
   → Use for selecting new tracking locations interactively
   → Time: ~1 minute

5. 🎯 INTERACTIVE POINT SELECTOR:
   python src/main.py --subject OSAMRI007 --interactive-selector --selector-timestep 100
   
   → Launch 3D point picker for selecting new tracking locations
   → Click points to add to tracking locations JSON
   → Requires PyVista

6. 📊 CUSTOM PATCH ANALYSIS:
   python src/main.py --subject OSAMRI007 --patchradii 1.0 3.0 7.0 --normalangle 45.0
   
   → Custom patch sizes (1mm, 3mm, 7mm) and surface filtering (45°)
   → Analyzes different region sizes around tracking points

7. 🔍 LIST AVAILABLE SUBJECTS:
   python src/main.py --listsubjects
   
   → Shows all detected subjects and their status
   → Checks for required flow profile files

8. ⚡ ANALYSIS WITHOUT VISUALIZATION:
   python src/main.py --subject OSAMRI007 --disablevisualization
   
   → Faster processing, generates all PDFs but no interactive HTML
   → Good for batch processing or when visualization not needed

9. 📈 SINGLE POINT ANALYSIS ONLY:
   python src/main.py --subject OSAMRI007 --disablepatchanalysis
   
   → Analyzes only individual tracking points (no patch regions)
   → Faster processing, smaller output files

REQUIRED FILES:
- {SUBJECT}FlowProfile.csv (breathing flow data)
- {SUBJECT}FlowProfile_smoothed.csv (smoothed flow data)  
- {SUBJECT}_xyz_tables/ or {SUBJECT}_xyz_tables_with_patches/ (CFD geometry data)
- {SUBJECT}_tracking_locations.json (anatomical landmark definitions)

OUTPUTS STRUCTURE:
{SUBJECT}_results/
├── tracked_points/    (CSV trajectory data)
├── figures/          (PNG images)
├── reports/          (PDF analysis reports)
├── interactive/      (HTML visualizations)
└── {SUBJECT}_key_time_points.json (breathing cycle analysis)

ENVIRONMENT:
Recommended: conda activate tf210_clone
'''

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=CustomHelpFormatter,
        epilog='''
Examples:
  python src/main.py --subject OSAMRI007                    # Full analysis
  python src/main.py --subject OSAMRI007 --rawdir qDNS_xyz_tables # Use custom raw data directory
  python src/main.py --highlight-patches --patch-timestep 100  # Highlight patch regions
  python src/main.py --raw-surface --surface-timestep 50   # Raw surface for point selection
  python src/main.py --subject OSAMRI007 --forcererun      # Force complete rerun
  python src/main.py --listsubjects                         # Show available data
        '''
    )
    
    parser.add_argument('--subject', type=str,
                      help='Subject name to process (default: auto-detect from existing folders)')
    parser.add_argument('--rawdir', type=str,
                      help='Custom raw data directory name (default: {subject}_xyz_tables). Use this to specify a different raw data directory like "qDNS_xyz_tables".')
    parser.add_argument('--forcererun', action='store_true', 
                      help='Force reprocessing of all tracked points, overwriting existing data')
    parser.add_argument('--disablepatchanalysis', action='store_true',
                      help='Disable patch-based region analysis (enabled by default)')
    parser.add_argument('--patchradii', nargs='+', type=float,
                      help='Patch radii in millimeters (default: 2.0 only for faster processing)')
    parser.add_argument('--normalangle', type=float, default=60.0,
                      help='Normal angle threshold in degrees for surface filtering (default: 60.0)')
    parser.add_argument('--disablevisualization', action='store_true',
                      help='Disable interactive 3D patch visualization (enabled by default)')
    parser.add_argument('--listsubjects', action='store_true',
                      help='List all available subjects and exit')
    parser.add_argument('--highlight-patches', action='store_true',
                      help='Highlight patch regions around tracking points (skip data processing)')
    parser.add_argument('--patch-timestep', type=int, default=100,
                      help='Time step for patch highlighting (default: 100 = 0.1s)')
    parser.add_argument('--raw-surface', action='store_true',
                      help='Show raw airway surface for manual point selection (skip data processing)')
    parser.add_argument('--surface-timestep', type=int, default=1,
                      help='Time step for surface visualization (default: 1 = 0.001s)')
    parser.add_argument('--interactive-selector', action='store_true',
                      help='Launch interactive 3D point selector for manual landmark selection')
    parser.add_argument('--selector-timestep', type=int, default=100,
                      help='Time step for interactive selector (default: 100 = 0.1s)')
    parser.add_argument('--smoothing-window', type=int, default=20,
                      help='Smoothing window size for plot data (default: 20 time steps)')
    args = parser.parse_args()
    
    # Handle list subjects command
    if args.listsubjects:
        subjects = detect_available_subjects()
        if subjects:
            print("📋 Available subjects:")
            for subject in subjects:
                print(f"  • {subject}")
                # Check if required files exist
                flow_file = Path(f"{subject}FlowProfile.csv")
                if flow_file.exists():
                    print(f"    ✅ Flow profile found")
                else:
                    print(f"    ❌ Flow profile missing")
        else:
            print("❌ No subjects found (no folders matching *_xyz_tables pattern)")
        sys.exit(0)
    
    # Convert patch radii from mm to meters if provided
    patch_radii = None
    if args.patchradii:
        patch_radii = [r / 1000.0 for r in args.patchradii]  # Convert mm to m
    
    main(overwrite_existing=args.forcererun, 
         enable_patch_analysis=not args.disablepatchanalysis,
         patch_radii=patch_radii,
         normal_angle_threshold=args.normalangle,
         enable_patch_visualization=not args.disablevisualization,
         subject_name=args.subject,
         raw_dir=args.rawdir,
         highlight_patches=getattr(args, 'highlight_patches', False),
         patch_timestep=getattr(args, 'patch_timestep', 100),
         raw_surface=getattr(args, 'raw_surface', False),
         surface_timestep=args.surface_timestep,
         interactive_selector=getattr(args, 'interactive_selector', False),
         selector_timestep=args.selector_timestep,
         smoothing_window=getattr(args, 'smoothing_window', 20)) 