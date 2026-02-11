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
    load_and_merge_configs,
    save_post_remesh_mappings,
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
        create_cfd_analysis_3x3_panel_original_scale,
        create_cfd_analysis_3x3_panel_with_markers_original_scale,
        create_cfd_analysis_3x3_panel_with_markers_both_time_versions,
        load_cfd_data_for_analysis
    )
    CFD_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import CFD analysis functions: {e}")
    CFD_ANALYSIS_AVAILABLE = False

# Import shared utilities
from utils.signal_processing import find_zero_crossings, smart_label_position, format_time_label


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

    try:
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
        distances, indices = nbrs.kneighbors([points[center_idx]])

        # Get local neighborhood points
        local_points = points[indices[0]]

        # Center the points
        centered = local_points - local_points.mean(axis=0)

        # Compute PCA to find the normal (smallest eigenvector)
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Last row is the normal direction
        return normal / np.linalg.norm(normal)
    except Exception:
        # Handle sklearn/threadpoolctl errors or SVD failures
        return np.array([0, 0, 1])  # Default normal

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

def save_trajectory_data(trajectory_data, subject_name, patch_number, face_index, description, is_region=False, time_offset_ms: float = 0.0):
    """Save trajectory data to CSV file in a dedicated folder.

    Args:
        trajectory_data: List of trajectory data points
        subject_name: Name of the subject
        patch_number: Patch number
        face_index: Face index
        description: Description of the tracking location
        is_region: Whether this is region-based tracking
        time_offset_ms: Time offset in milliseconds for normalization (breathing cycle start)
    """
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
    
    # Calculate normalized time (shifted to start from 0)
    time_offset_s = time_offset_ms / 1000.0  # Convert ms to seconds
    time_values = [p['time'] for p in unique_trajectory_data]
    time_normalized = [t - time_offset_s for t in time_values]

    df = pd.DataFrame({
        'Time Point': [p['time_point'] for p in unique_trajectory_data],
        'Time (s)': time_values,
        'Time_normalized (s)': time_normalized,
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
    # Add Time_normalized if available in source data
    if 'Time_normalized (s)' in dataframes[0].columns:
        mean_data['Time_normalized (s)'] = dataframes[0]['Time_normalized (s)'].values
    
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
    
    ax1.set_ylabel('Velocity (mm/s)', fontsize=14.4, fontweight='bold')
    ax1.set_title(f'Airway Surface Velocity at {description}\nPatch: {patch_number}, Face: {face_index}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    # Format tick labels
    ax1.tick_params(axis='both', which='major', labelsize=13.44)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Plot 2: Individual velocity components
    ax2.plot(df['Time (s)'], df['Velocity[i] (m/s)'] * 1000, 'r-', linewidth=1.5, label='v_x')
    ax2.plot(df['Time (s)'], df['Velocity[j] (m/s)'] * 1000, 'g-', linewidth=1.5, label='v_y')
    ax2.plot(df['Time (s)'], df['Velocity[k] (m/s)'] * 1000, 'b-', linewidth=1.5, label='v_z')
    
    # Add breathing cycle markers if available
    for marker in cycle_markers:
        ax2.axvline(x=marker['time'], color='k', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Time (s)', fontsize=14.4, fontweight='bold')
    ax2.set_ylabel('|v⃗| (mm/s)', fontsize=14.4, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    # Format tick labels
    ax2.tick_params(axis='both', which='major', labelsize=13.44)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight('bold')
    
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
        
        ax_raw.set_xlabel('Time (s)', fontsize=14.4, fontweight='bold', labelpad=10)
        ax_raw.set_ylabel('Normalized Values', fontsize=14.4, fontweight='bold', labelpad=10)
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
              
            ax.set_xlabel('Time (s)', fontsize=14.4, fontweight='bold', labelpad=10)
            ax.set_ylabel('Correlation', fontsize=14.4, fontweight='bold', labelpad=10)
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
    plt.xlabel('Rate of change of acceleration (dA/dt) [mm/s³]', fontsize=14.4, fontweight='bold')
    plt.ylabel('Rate of change of pressure (dP/dt) [Pa/s]', fontsize=14.4, fontweight='bold')
    plt.title(f'Rate of Change Analysis: dP/dt vs dA/dt\n{description} - Patch {patch_number}, Face {face_index}', 
              fontsize=14, fontweight='bold')
    
    # Add zero reference lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Format tick labels
    plt.gca().tick_params(axis='both', which='major', labelsize=13.44)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')
    
    # Tight layout and save
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_metrics_vs_time_plot(df, subject_name, description, patch_number, face_index, pdf, use_normalized_time: bool = False, flow_profile_path: str = None):
    """Create a plot showing various metrics over time.

    Args:
        df: DataFrame with tracking data
        subject_name: Name of the subject
        description: Description of the tracking location
        patch_number: Patch number
        face_index: Face index
        pdf: PDF object to save the plot to
        use_normalized_time: If True, use Time_normalized (s) column (starts from 0)
        flow_profile_path: Optional explicit path to flow profile CSV
    """
    # Common settings
    LABEL_SIZE = 13.44  # Reduced by 20% from 16.8
    TITLE_SIZE = 17  # Increased by 20% from 14

    # Get data - use normalized time if requested and available
    if use_normalized_time and 'Time_normalized (s)' in df.columns:
        times = df['Time_normalized (s)'].values
        time_label = 'Time (s) - Normalized'
        time_suffix = '_normalized'
    else:
        times = df['Time (s)'].values
        time_label = 'Time (s)'
        time_suffix = ''
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
    if flow_profile_path is not None:
        flow_profile = pd.read_csv(flow_profile_path)
    else:
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
    if inhale_exhale_transition is not None:
        ax5.axvline(x=inhale_exhale_transition, color='k', linestyle=':', alpha=0.7, label=inhale_exhale_label)
    ax5.set_xlabel(time_label, fontsize=LABEL_SIZE, fontweight='bold')
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
    png_filename = figures_dir / f"{subject_name}_metrics_vs_time_{description.lower().replace(' ', '_')}_patch{patch_number}_face{face_index}{time_suffix}.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    
    plt.close()

def create_symmetric_comparison_panel_clean(dfs, subject_name, pdf, pdfs_dir=None, locations=None):
    """
    Create a symmetric NxM panel comparison of pressure, velocity, and acceleration across anatomical points.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.
    This version does not include zero-crossing markers.

    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to (can be None if standalone_output=True)
        pdfs_dir: Directory for PDF output
        locations: List of location dictionaries with 'key' and 'description' fields (from JSON)
    """
    if locations is None or len(locations) == 0:
        print("Warning: No locations provided for comparison panel")
        return

    print(f"\nGenerating clean symmetric comparison panel (without markers) for {len(locations)} locations...")

    # Calculate grid dimensions based on number of locations
    n_locations = len(locations)
    # Create figure with dynamic layout (n_locations rows x 3 columns for P, V, A)
    fig = plt.figure(figsize=(20, 6 * n_locations + 2))
    gs = fig.add_gridspec(n_locations, 3, hspace=0.3, wspace=0.3)
    
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
    LABEL_SIZE = 11.2  # Reduced by 20% from 14
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
        # Format tick labels
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        
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
        # Format tick labels
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        
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
        # Format tick labels
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
    
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

def create_symmetric_comparison_panel(dfs, subject_name, pdf, pdfs_dir=None, locations=None):
    """
    Create a symmetric NxM panel comparison of pressure, velocity, and acceleration across anatomical points.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.

    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to (can be None if standalone_output=True)
        pdfs_dir: Directory for PDF output
        locations: List of location dictionaries with 'key' and 'description' fields (from JSON)
    """
    if locations is None or len(locations) == 0:
        print("Warning: No locations provided for comparison panel")
        return

    print(f"\nGenerating symmetric comparison panel for {len(locations)} locations...")

    # Calculate grid dimensions based on number of locations
    n_locations = len(locations)
    # Create figure with dynamic layout (n_locations rows x 3 columns for P, V, A)
    fig = plt.figure(figsize=(20, 6 * n_locations + 2))
    gs = fig.add_gridspec(n_locations, 3, hspace=0.3, wspace=0.3)
    
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
    LABEL_SIZE = 11.2  # Reduced by 20% from 14
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
        # Format tick labels
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        
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
        # Format tick labels
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        
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
        # Format tick labels
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
        
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

def create_symmetric_comparison_panel_smooth(dfs, subject_name, pdf, pdfs_dir=None, locations=None):
    """
    Create a symmetric NxM panel comparison of pressure, velocity, and acceleration across anatomical points.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.
    This version applies smoothing to the data.

    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to (can be None if standalone_output=True)
        pdfs_dir: Directory for PDF output
        locations: List of location dictionaries with 'key' and 'description' fields (from JSON)
    """
    if locations is None or len(locations) == 0:
        print("Warning: No locations provided for comparison panel")
        return

    print(f"\nGenerating smoothed symmetric comparison panel for {len(locations)} locations...")

    # Calculate grid dimensions based on number of locations
    n_locations = len(locations)
    # Create figure with dynamic layout (n_locations rows x 3 columns for P, V, A)
    fig = plt.figure(figsize=(20, 6 * n_locations + 2))
    gs = fig.add_gridspec(n_locations, 3, hspace=0.3, wspace=0.3)

    # Define moving average function for smoothing
    def moving_average(data, window_size=20):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
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
    LABEL_SIZE = 11.2  # Reduced by 20% from 14
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
        # Format tick labels
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        
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
        # Format tick labels
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        
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
        # Format tick labels
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
        
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

def create_symmetric_comparison_panel_smooth_with_markers(dfs, subject_name, pdf, pdfs_dir=None, locations=None, use_original_time=False):
    """
    Create a symmetric NxM panel comparison of pressure, velocity, and acceleration across anatomical points.
    This version applies a moving average smoothing filter with window size 20 to all data,
    and also adds markers and text labels for zero crossings.
    All plots are made symmetric about the origin (0,0) and share the same axis range for easy comparison.

    Arguments:
        dfs: Dictionary of DataFrames, with keys corresponding to anatomical locations
        subject_name: Name of the subject
        pdf: PDF object to save the plot to
        pdfs_dir: Directory for PDF output
        locations: List of location dictionaries with 'key' and 'description' fields (from JSON)
        use_original_time: If True, use original timestamps; if False, normalize to start from 0
    """
    if locations is None or len(locations) == 0:
        print("Warning: No locations provided for comparison panel")
        return

    time_mode = "original" if use_original_time else "normalized"
    print(f"\nGenerating smoothed symmetric comparison panel with zero-crossing markers for {len(locations)} locations ({time_mode} time)...")

    # Calculate grid dimensions based on number of locations
    n_locations = len(locations)
    # Create figure with dynamic layout (n_locations rows x 3 columns for P, V, A)
    fig = plt.figure(figsize=(20, 6 * n_locations + 2))
    gs = fig.add_gridspec(n_locations, 3, hspace=0.3, wspace=0.3)

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

    # Time offset for normalization (0 if using original time)
    time_offset = 0 if use_original_time else global_time_min

    # The original inhale-exhale transition time
    original_inhale_exhale = 1.034  # This is the value from the command output

    # Calculate the display inhale-exhale transition time
    display_inhale_exhale = original_inhale_exhale if use_original_time else (original_inhale_exhale - global_time_min)

    if use_original_time:
        print(f"Using original timestamps: Range starts at {global_time_min:.3f}s")
        print(f"Inhale-exhale transition at {original_inhale_exhale:.3f}s")
    else:
        print(f"Normalizing time: Original range started at {global_time_min:.3f}s")
        print(f"Inhale-exhale transition: Original at {original_inhale_exhale:.3f}s, Normalized at {display_inhale_exhale:.3f}s")
    
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

        # Apply time offset (0 for original time, global_time_min for normalized)
        display_times = times - time_offset

        # Calculate acceleration
        dt = times[1] - times[0]  # Use original time for dt calculation
        dvdotn = np.diff(vdotn)
        adotn = np.append(dvdotn / dt, dvdotn[-1] / dt)  # Already in mm/s² since vdotn is in mm/s

        # Find sign changes in original data before smoothing (for accurate zero crossings)
        v_crossings = find_zero_crossings(display_times, vdotn)
        a_crossings = find_zero_crossings(display_times, adotn)
        p_crossings = find_zero_crossings(display_times, pressure)

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
            'display_times': display_times,
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
    LABEL_SIZE = 11.2  # Reduced by 20% from 14
    TITLE_SIZE = 17  # Increased by 20% from 14

    # Create a custom colormap that transitions at the inhale-exhale point
    display_time_min = min([data['display_times'].min() for data in processed_data.values()])
    display_time_max = max([data['display_times'].max() for data in processed_data.values()])
    norm = plt.Normalize(display_time_min, display_time_max)
    transition_norm = (display_inhale_exhale - display_time_min) / (display_time_max - display_time_min)
    
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
        display_times = data['display_times']
        vdotn = data['vdotn']
        pressure = data['pressure']
        adotn = data['adotn']
        v_crossings = data['v_crossings']
        a_crossings = data['a_crossings']
        p_crossings = data['p_crossings']

        # Initialize label tracking for collision detection
        ax1_labels = []
        ax2_labels = []
        ax3_labels = []

        # 1. Plot p vs v
        ax1 = fig.add_subplot(gs[i, 0])
        scatter1 = ax1.scatter(vdotn, pressure, c=display_times, cmap=custom_cmap, norm=norm)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Mark the zero crossings in the p vs v plot
        for v_cross in v_crossings:
            t_idx = np.abs(display_times - v_cross).argmin()
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(display_times - v_cross)).argmin()
            p_at_cross = pressure[idx_at_cross]
            
            # Use smart positioning
            data_points = list(zip(vdotn, pressure))
            time_label = format_time_label(v_cross)
            xytext, ha, va = smart_label_position(ax1, (0, p_at_cross), time_label, 
                                                ax1_labels, data_points)
            ax1_labels.append(xytext)
            
            ax1.annotate(time_label, 
                       xy=(0, p_at_cross), 
                       xytext=xytext,
                       arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                       color='black', fontsize=11, fontweight='bold',
                       ha=ha, va=va)
        
        for p_cross in p_crossings:
            t_idx = np.abs(display_times - p_cross).argmin()
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(display_times - p_cross)).argmin()
            v_at_cross = vdotn[idx_at_cross]

            # Use smart positioning
            data_points = list(zip(vdotn, pressure))
            time_label = format_time_label(p_cross)
            xytext, ha, va = smart_label_position(ax1, (v_at_cross, 0), time_label,
                                                ax1_labels, data_points)
            ax1_labels.append(xytext)

            ax1.annotate(time_label,
                       xy=(v_at_cross, 0),
                       xytext=xytext,
                       arrowprops=dict(arrowstyle="->", color='blue', lw=1.5),
                       color='black', fontsize=11, fontweight='bold',
                       ha=ha, va=va)

        ax1.set_xlim(-v_max, v_max)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # Format tick labels
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')

        # 2. Plot p vs a
        ax2 = fig.add_subplot(gs[i, 1])
        scatter2 = ax2.scatter(adotn, pressure, c=display_times, cmap=custom_cmap, norm=norm)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Mark the zero crossings in the p vs a plot
        for a_cross in a_crossings:
            t_idx = np.abs(display_times - a_cross).argmin()
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(display_times - a_cross)).argmin()
            p_at_cross = pressure[idx_at_cross]
            
            # Use smart positioning
            data_points = list(zip(adotn, pressure))
            time_label = format_time_label(a_cross)
            xytext, ha, va = smart_label_position(ax2, (0, p_at_cross), time_label, 
                                                ax2_labels, data_points)
            ax2_labels.append(xytext)
            
            ax2.annotate(time_label, 
                       xy=(0, p_at_cross), 
                       xytext=xytext,
                       arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                       color='black', fontsize=11, fontweight='bold',
                       ha=ha, va=va)
        
        for p_cross in p_crossings:
            t_idx = np.abs(display_times - p_cross).argmin()
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(display_times - p_cross)).argmin()
            a_at_cross = adotn[idx_at_cross]

            # Use smart positioning
            data_points = list(zip(adotn, pressure))
            time_label = format_time_label(p_cross)
            xytext, ha, va = smart_label_position(ax2, (a_at_cross, 0), time_label,
                                                ax2_labels, data_points)
            ax2_labels.append(xytext)

            ax2.annotate(time_label,
                       xy=(a_at_cross, 0),
                       xytext=xytext,
                       arrowprops=dict(arrowstyle="->", color='blue', lw=1.5),
                       color='black', fontsize=11, fontweight='bold',
                       ha=ha, va=va)

        ax2.set_xlim(-a_max, a_max)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        # Format tick labels
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')

        # 3. Plot v vs a
        ax3 = fig.add_subplot(gs[i, 2])
        scatter3 = ax3.scatter(adotn, vdotn, c=display_times, cmap=custom_cmap, norm=norm)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Mark the zero crossings in the v vs a plot
        for v_cross in v_crossings:
            t_idx = np.abs(display_times - v_cross).argmin()
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(display_times - v_cross)).argmin()
            a_at_cross = adotn[idx_at_cross]

            # Use smart positioning
            data_points = list(zip(adotn, vdotn))
            time_label = format_time_label(v_cross)
            xytext, ha, va = smart_label_position(ax3, (a_at_cross, 0), time_label,
                                                ax3_labels, data_points)
            ax3_labels.append(xytext)

            ax3.annotate(time_label,
                       xy=(a_at_cross, 0),
                       xytext=xytext,
                       arrowprops=dict(arrowstyle="->", color='green', lw=1.5),
                       color='black', fontsize=11, fontweight='bold',
                       ha=ha, va=va)

        for a_cross in a_crossings:
            t_idx = np.abs(display_times - a_cross).argmin()
            # Add a timestamp label to the marker
            idx_at_cross = (np.abs(display_times - a_cross)).argmin()
            v_at_cross = vdotn[idx_at_cross]
            
            # Use smart positioning
            data_points = list(zip(adotn, vdotn))
            time_label = format_time_label(a_cross)
            xytext, ha, va = smart_label_position(ax3, (0, v_at_cross), time_label, 
                                                ax3_labels, data_points)
            ax3_labels.append(xytext)
            
            ax3.annotate(time_label, 
                       xy=(0, v_at_cross), 
                       xytext=xytext,
                       arrowprops=dict(arrowstyle="->", color='purple', lw=1.5),
                       color='black', fontsize=11, fontweight='bold',
                       ha=ha, va=va)
        
        ax3.set_xlim(-a_max, a_max)
        ax3.set_ylim(-v_max, v_max)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        # Format tick labels
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
    
    # Add a colorbar for time
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
    
    # Add a note about the time mode and smoothing
    if use_original_time:
        note_text = f'Note: Using original timestamps. Inhale-exhale transition at {display_inhale_exhale:.3f}s.\nData smoothed with moving average (window size: 20).'
    else:
        note_text = f'Note: Time has been normalized to start at 0s. Original data started at {global_time_min:.3f}s.\nInhale-exhale transition at {display_inhale_exhale:.3f}s. Data smoothed with moving average (window size: 20).'
    fig.text(0.5, 0.01, note_text, fontsize=10, ha='center', va='bottom')
    
    # Add overall title
    fig.suptitle(f'Comparative Analysis of Pressure, Velocity and Acceleration\nSymmetric Plots Centered at Origin (Smoothed Version with Zero-Crossing Markers)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # [left, bottom, right, top]
    
    # Save to main PDF if provided
    if pdf is not None:
        pdf.savefig(fig)
    
    # Save a standalone PDF version to pdfs directory
    time_suffix = "_original_time" if use_original_time else "_normalized_time"
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_3x3_panel_smoothed_with_markers{time_suffix}.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_3x3_panel_smoothed_with_markers{time_suffix}.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    
    plt.close()
    print("Smoothed symmetric comparison panel with markers completed.")


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
    LABEL_SIZE = 13.44  # Reduced by 20% from 16.8  
    TITLE_SIZE = 17  # Increased by 20% from 14
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                ax.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
                # Make tick labels bold
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
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
    ax6_flow.tick_params(axis='y', labelcolor='k', labelsize=13.44)
    ax6_flow.tick_params(axis='x', labelsize=13.44)
    # Make tick labels bold
    for label in ax6_flow.get_xticklabels() + ax6_flow.get_yticklabels():
        label.set_fontweight('bold')
            
            # Calculate and plot signed velocity
    signed_velocity = velocities * np.sign(vdotn)
    vel_line = ax6_vel.plot(times, signed_velocity, 'b--', 
                                  label='Signed Velocity', linewidth=1.5)
    vdot_line = ax6_vel.plot(times, vdotn, 'r:', 
                                   label='v⃗·n⃗', linewidth=1.5)
            
    ax6_vel.set_ylabel('Velocity (m/s)', color='b')
    ax6_vel.tick_params(axis='y', labelcolor='b', labelsize=13.44)
    # Make tick labels bold  
    for label in ax6_vel.get_yticklabels():
        label.set_fontweight('bold')
            
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

def create_clean_flow_profile_plot(subject_name, hdf5_file_path, results_dir, output_dir=None, pdfs_dir=None):
    """
    Create a clean, smoothed flow profile plot showing just one breathing cycle.
    Reads data from HDF5 and uses pre-detected breathing cycle from metadata.json.
    Converts flow rate from kg/s to L/min.

    Arguments:
        subject_name: Name of the subject
        hdf5_file_path: Path to the HDF5 file containing flow profile data
        results_dir: Path to results directory containing metadata.json
        output_dir: Directory to save PNG files (optional)
        pdfs_dir: Directory to save PDF files (optional)
    """
    print(f"\nGenerating clean flow profile visualization for {subject_name}...")

    # Load flow profile data from HDF5
    from data_processing.trajectory import get_flow_profile, has_flow_profile

    if not has_flow_profile(hdf5_file_path):
        print(f"❌ No flow profile found in HDF5 for {subject_name}")
        return

    flow_df = get_flow_profile(hdf5_file_path)
    if flow_df is None:
        print(f"❌ Could not read flow profile from HDF5 for {subject_name}")
        return

    print(f"📊 Using flow profile from HDF5: {hdf5_file_path}")
    flow_times = flow_df['time (s)'].values
    flow_rates = flow_df['Massflowrate (kg/s)'].values

    # Read pre-detected zero crossings from metadata.json
    metadata_path = Path(results_dir) / f"{subject_name}_metadata.json"
    if not metadata_path.exists():
        print(f"❌ No metadata.json found for {subject_name}")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    breathing_cycle = metadata.get('breathing_cycle', {})
    zero_crossings_ms = breathing_cycle.get('zero_crossings_ms', [])

    if len(zero_crossings_ms) < 3:
        print(f"❌ Not enough zero crossings in metadata.json (need 3, got {len(zero_crossings_ms)})")
        return

    # Convert from ms to seconds
    zero_times = [t / 1000.0 for t in zero_crossings_ms[:3]]  # [start, transition, end]

    print(f"📊 Using pre-detected breathing cycle: {zero_times[0]:.3f}s - {zero_times[2]:.3f}s")

    # Extract the clean cycle using pre-detected boundaries
    # Find indices closest to zero crossings
    start_idx = np.searchsorted(flow_times, zero_times[0])
    inhale_exhale_idx = np.searchsorted(flow_times, zero_times[1])
    end_idx = np.searchsorted(flow_times, zero_times[2])

    # Adjust to start AFTER first zero crossing (where flow becomes negative for inhale)
    # and end BEFORE last zero crossing (where flow is still positive for exhale)
    # This ensures we only show data WITHIN the breathing cycle
    if start_idx < len(flow_rates) and flow_rates[start_idx] > 0:
        start_idx += 1  # Skip pre-zero-crossing point
    if end_idx > 0 and flow_rates[end_idx] < 0:
        end_idx -= 1  # Skip post-zero-crossing point

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

    # Plot ONLY the smoothed flow profile (no reference line - clean plot shows only smoothed data)
    ax.plot(normalized_times, smoothed_rates_lpm, 'b-', linewidth=2.5, label='Smoothed Flow Rate')
    
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
    ax.set_xlabel('Time (s)', fontsize=16.8, fontweight='bold')
    ax.set_ylabel('Flow Rate (L/min)', fontsize=16.8, fontweight='bold')
    ax.set_title(f'Clean Breathing Cycle Flow Profile - {subject_name}', fontsize=16, fontweight='bold')
    
    # Add a note about the normalized time
    original_start_time = clean_times[0]
    fig.text(0.5, 0.01, f'Note: Time has been normalized to start at 0s. Original data started at {original_start_time:.3f}s.', 
            fontsize=10, ha='center', va='bottom')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=13.44)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
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


def create_original_flow_profile_plot(subject_name, hdf5_file_path, results_dir, output_dir=None, pdfs_dir=None):
    """
    Create an original (raw) flow profile plot showing FULL data without cycle extraction.
    Reads data from HDF5. NO smoothing applied.
    Converts flow rate from kg/s to L/min.
    Marks detected zero crossings for reference.

    Arguments:
        subject_name: Name of the subject
        hdf5_file_path: Path to the HDF5 file containing flow profile data
        results_dir: Path to results directory containing metadata.json
        output_dir: Directory to save PNG files (optional)
        pdfs_dir: Directory to save PDF files (optional)
    """
    print(f"\nGenerating original flow profile visualization for {subject_name}...")

    # Load flow profile data from HDF5
    from data_processing.trajectory import get_flow_profile, has_flow_profile

    if not has_flow_profile(hdf5_file_path):
        print(f"❌ No flow profile found in HDF5 for {subject_name}")
        return

    flow_df = get_flow_profile(hdf5_file_path)
    if flow_df is None:
        print(f"❌ Could not read flow profile from HDF5 for {subject_name}")
        return

    print(f"📊 Using flow profile from HDF5: {hdf5_file_path}")
    flow_times = flow_df['time (s)'].values
    flow_rates = flow_df['Massflowrate (kg/s)'].values

    # Convert from kg/s to L/min
    air_density = 1.225  # kg/m³
    conversion_factor = (1 / air_density) * 1000 * 60  # kg/s to L/min
    flow_rates_lpm = flow_rates * conversion_factor

    # Read pre-detected zero crossings from metadata.json for reference lines
    metadata_path = Path(results_dir) / f"{subject_name}_metadata.json"
    zero_crossings_s = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        breathing_cycle = metadata.get('breathing_cycle', {})
        zero_crossings_ms = breathing_cycle.get('zero_crossings_ms', [])
        zero_crossings_s = [t / 1000.0 for t in zero_crossings_ms]

    # Create the plot
    fig, ax = plt.figure(figsize=(14, 8)), plt.gca()

    # Plot the raw flow profile (NO smoothing)
    ax.plot(flow_times, flow_rates_lpm, 'b-', linewidth=1.5, label='Raw Flow Rate')

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    # Add vertical lines at detected zero crossings
    colors = ['green', 'red', 'green']  # start, transition, end
    labels = ['Cycle Start', 'Inhale→Exhale', 'Cycle End']
    for i, t_cross in enumerate(zero_crossings_s[:3]):
        color = colors[i] if i < len(colors) else 'gray'
        label = labels[i] if i < len(labels) else f'Zero Crossing {i+1}'
        ax.axvline(x=t_cross, color=color, linestyle='-', linewidth=2, alpha=0.7, label=f'{label} ({t_cross:.3f}s)')

    # Add labels and title
    ax.set_xlabel('Time (s)', fontsize=16.8, fontweight='bold')
    ax.set_ylabel('Flow Rate (L/min)', fontsize=16.8, fontweight='bold')
    ax.set_title(f'Original Flow Profile (Full Data) - {subject_name}', fontsize=16, fontweight='bold')

    # Add info about data
    fig.text(0.5, 0.01, f'Total data points: {len(flow_times)} | Time range: {flow_times[0]:.3f}s - {flow_times[-1]:.3f}s',
            fontsize=10, ha='center', va='bottom')

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Format tick labels
    ax.tick_params(axis='both', which='major', labelsize=13.44)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save as a standalone PDF to pdfs directory
    if pdfs_dir:
        standalone_filename = pdfs_dir / f"{subject_name}_original_flow_profile.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")
    else:
        standalone_filename = f"{subject_name}_original_flow_profile.pdf"
        plt.savefig(standalone_filename, bbox_inches='tight')
        print(f"Saved standalone PDF: {standalone_filename}")

    # If we have an output directory, also save there
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_filename = output_dir / f"{subject_name}_original_flow_profile.png"
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

def auto_detect_visualization_timestep(subject_name: str, flow_profile_path: str = None) -> Tuple[float, int]:
    """
    Auto-detect the appropriate timestep for visualization.

    Priority:
    1. HDF5 metadata (if exists) - most reliable for plotting mode
    2. CSV files + flow profile - for prepare mode

    Args:
        subject_name: Name of the subject
        flow_profile_path: Optional explicit path to flow profile CSV file

    Returns:
        Tuple of (timestep_value, display_timestep_ms) where:
        - timestep_value: Raw timestep value to use for visualization
        - display_timestep_ms: Timestep converted to milliseconds for display
    """
    # Priority 1: Check HDF5 file for breathing cycle metadata (most reliable)
    # Check results folder first, then root for backwards compatibility
    hdf5_file_path = Path(f'{subject_name}_results/{subject_name}_cfd_data.h5')
    if not hdf5_file_path.exists():
        hdf5_file_path = Path(f'{subject_name}_cfd_data.h5')  # Backwards compatibility
    if hdf5_file_path.exists():
        try:
            import h5py
            with h5py.File(hdf5_file_path, 'r') as f:
                if 'time_points' in f:
                    time_points = f['time_points'][:]
                    if len(time_points) > 0:
                        # First time point in HDF5 is already filtered to breathing cycle
                        first_time_s = time_points[0]
                        display_timestep = round(first_time_s * 1000)  # Convert to ms
                        print(f"🎯 Auto-detected timestep {display_timestep}ms from HDF5 (first in breathing cycle)")
                        return first_time_s * 1000, display_timestep  # Return in ms for consistency
        except Exception as e:
            print(f"⚠️  Warning: Could not read HDF5 metadata: {e}")

    # Priority 2: Fall back to CSV files + flow profile detection
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
    start_time, end_time = find_breathing_cycle_bounds(subject_name, flow_profile_path)
    if start_time is not None and end_time is not None:
        filtered_files = filter_xyz_files_by_time(sorted_files, start_time, end_time)
        if filtered_files:
            # Use the first file from the breathing cycle (same as tracking analysis)
            first_file = filtered_files[0]
            visualization_timestep = extract_timestep_from_filename(first_file)

            # Convert to milliseconds for display message only
            if 'e+' in first_file.stem or 'e-' in first_file.stem:
                # Scientific notation - likely in seconds
                display_timestep = round(visualization_timestep * 1000)
            else:
                # Likely already in milliseconds
                display_timestep = round(visualization_timestep)

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
        display_timestep = round(first_timestep * 1000)
    else:
        display_timestep = round(first_timestep)
    
    print(f"🎯 Auto-detected timestep {display_timestep}ms for visualization (first available file)")
    return visualization_timestep, display_timestep

def main(overwrite_existing: bool = False,
         enable_patch_analysis: bool = True,
         patch_radii: List[float] = None,
         normal_angle_threshold: float = 60.0,
         enable_patch_visualization: bool = True,
         subject_name: str = None,
         raw_dir: str = None,
         xyz_path: str = None,
         highlight_patches: bool = False,
         patch_timestep: int = 100,
         raw_surface: bool = False,
         surface_timestep: int = 1,
         interactive_selector: bool = False,
         selector_timestep: int = 100,
         smoothing_window: int = 20,
         patch_selection_mode: bool = False,
         plotting_mode: bool = False,
         all_in_one_mode: bool = False,
         has_remesh: bool = False,
         remesh_before: str = None,
         remesh_after: str = None,
         flow_profile_path: str = None,
         inhale_start: float = None,
         transition: float = None,
         exhale_end: float = None):
    """
    Main function to process CFD data and generate analysis plots.

    Two-phase workflow:
    - Phase 1 (--prepare): Create HDF5, detect breathing cycle/remesh, create templates
    - Phase 2 (--point-picker): Select anatomical landmarks on the airway surface
    - Phase 3 (--plotting): Generate analysis and plots using picked points
    - All-in-one (--all): Run prepare + plotting in one pass (requires pre-configured tracking)

    Remesh handling:
    - If --has-remesh is set, use coordinate-based mapping across mesh changes
    """
    # Handle --all mode: run both prepare and plotting
    if all_in_one_mode:
        patch_selection_mode = True
        plotting_mode = True

    # Determine subject name
    if subject_name is None:
        subject_name = auto_detect_subject()
    else:
        # Validate provided subject name
        if xyz_path:
            # When xyz_path is provided, check if subject string is in the path
            if subject_name not in xyz_path:
                print(f"⚠️  Warning: Subject '{subject_name}' not found in path '{xyz_path}'")
                print(f"   Proceeding anyway - ensure this is intentional")
            else:
                print(f"✅ Using subject '{subject_name}' with XYZ path: {xyz_path}")
        else:
            # For plotting mode, only need results folder with HDF5 - don't require XYZ tables
            if plotting_mode:
                results_dir = Path(f"{subject_name}_results")
                hdf5_file = results_dir / f"{subject_name}_cfd_data.h5"
                if results_dir.exists() and hdf5_file.exists():
                    print(f"✅ Using subject '{subject_name}' (plotting mode - HDF5 found)")
                else:
                    print(f"❌ Subject '{subject_name}' not found for plotting mode!")
                    if not results_dir.exists():
                        print(f"   Missing results folder: {results_dir}")
                    elif not hdf5_file.exists():
                        print(f"   Missing HDF5 file: {hdf5_file}")
                    print(f"💡 Tip: Run Phase 1 first to create the HDF5 file")
                    raise ValueError(f"Subject '{subject_name}' not found")
            else:
                # Check local folders for subject (requires XYZ tables)
                available_subjects = detect_available_subjects()
                if subject_name not in available_subjects:
                    print(f"❌ Subject '{subject_name}' not found!")
                    if available_subjects:
                        print(f"Available subjects: {available_subjects}")
                    else:
                        print("No subjects detected in current directory.")
                    print(f"💡 Tip: Use --xyz-path to specify XYZ tables location if data is elsewhere")
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
            patch_timestep, display_timestep = auto_detect_visualization_timestep(subject_name, flow_profile_path)
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
        
        # Find HDF5 file (check results folder first, then root for backwards compatibility)
        hdf5_path = f"{subject_name}_results/{subject_name}_cfd_data.h5"
        if not Path(hdf5_path).exists():
            hdf5_path = f"{subject_name}_cfd_data.h5"

        # Run visualization
        try:
            fig = visualize_patch_regions(
                subject_name=subject_name,
                time_step=patch_timestep,
                patch_radii=patch_radii,
                use_pipeline_data=True,
                normal_angle_threshold=normal_angle_threshold,
                hdf5_file_path=hdf5_path
            )
            
            if fig:
                print(f"✅ Patch highlighting completed successfully!")
                print(f"📁 Interactive visualization saved as: {subject_name}_patch_regions_t{int(round(patch_timestep))}ms.html")
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

    # Handle prepare mode (Phase 1: Create HDF5, HTML, and template JSON)
    if patch_selection_mode:
        print(f"\n📋 Prepare Mode: Phase 1 of two-phase workflow")
        print("   This mode will:")
        print("   1. Convert raw CSV files to HDF5 format")
        print("   2. Create interactive HTML for patch/face selection")
        print("   3. Create template tracking locations JSON")
        print("   4. Skip all analysis and plotting")

        # Import required functions
        from utils.file_processing import (
            find_flow_profile_file,
            validate_subject_files,
            ask_remesh_questions_interactive,
            detect_timestep_from_csv,
            detect_remesh_from_file_sizes,
            detect_breathing_cycle_enhanced
        )
        from data_processing.trajectory import auto_select_csv_to_hdf5_method, store_remesh_metadata

        # Create results directory structure
        results_dir = Path(f'{subject_name}_results')
        interactive_dir = results_dir / 'interactive'
        for dir_path in [results_dir, interactive_dir, results_dir / 'tracked_points',
                         results_dir / 'figures', results_dir / 'reports']:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Created results directory structure: {results_dir}")

        # Find raw CSV files
        # Priority: xyz_path > raw_dir > default ({subject}_xyz_tables)
        if xyz_path:
            raw_xyz_dir = Path(xyz_path)
            if not raw_xyz_dir.exists():
                print(f"❌ XYZ path does not exist: {xyz_path}")
                return
            print(f"📁 Using custom XYZ path: {xyz_path}")
        elif raw_dir:
            raw_xyz_dir = Path(raw_dir)
        else:
            raw_xyz_dir = Path(f'{subject_name}_xyz_tables')

        if not raw_xyz_dir.exists():
            print(f"❌ No XYZ directory found for {subject_name}")
            print(f"   Expected: {subject_name}_xyz_tables/")
            if xyz_path:
                print(f"   Or custom path: {xyz_path}")
            return

        xyz_files = list(raw_xyz_dir.glob('*XYZ_Internal_Table_table_*.csv'))
        if not xyz_files:
            print(f"❌ No CSV files found in {raw_xyz_dir}")
            return

        print(f"📊 Found {len(xyz_files)} CSV files in {raw_xyz_dir}")

        # Sort files chronologically
        timestep_file_pairs = []
        for file_path in xyz_files:
            try:
                timestep = extract_timestep_from_filename(file_path)
                timestep_file_pairs.append((timestep, file_path))
            except ValueError:
                continue
        timestep_file_pairs.sort(key=lambda x: x[0])
        xyz_files = [f for _, f in timestep_file_pairs]

        # Auto-detect timestep from CSV column
        print(f"\n🔍 Auto-detecting timestep from CSV data...")
        try:
            timestep_info = detect_timestep_from_csv(xyz_files[0])
            print(f"✓ Detected timestep: {timestep_info['time_ms']:.3f} ms (from column '{timestep_info['time_column_name']}')")
        except ValueError as e:
            print(f"⚠ Could not detect timestep from CSV: {e}")
            print("  Using filename-based detection as fallback")
            timestep_info = None

        # Auto-detect remesh from file size changes (2% threshold)
        print(f"\n🔍 Auto-detecting remesh from file sizes...")
        remesh_detection = detect_remesh_from_file_sizes(xyz_files)

        if remesh_detection['has_remesh']:
            events = remesh_detection.get('remesh_events', [])

            # Validate events list before accessing
            if events and len(events) > 0:
                print(f"✓ Auto-detected {len(events)} remesh event(s) (threshold: {remesh_detection['threshold_percent']:.0f}% size change)")
                for i, event in enumerate(events):
                    print(f"  Event {i+1}: at timestep {event.get('timestep_boundary', 'unknown')}")
                    before_file = event.get('before_file')
                    after_file = event.get('after_file')
                    if before_file:
                        before_name = Path(before_file).name if isinstance(before_file, str) else before_file.name
                        print(f"    Before: {before_name} ({event.get('before_size', 0):,} bytes)")
                    if after_file:
                        after_name = Path(after_file).name if isinstance(after_file, str) else after_file.name
                        print(f"    After:  {after_name} ({event.get('after_size', 0):,} bytes)")
                    print(f"    Size change: {event.get('size_change_percent', 0):+.1f}%")

                # Convert to remesh_info format
                remesh_info = {
                    'has_remesh': True,
                    'manual_remesh_timesteps_ms': [],  # Empty array template, e.g. [637, 638]
                    'remesh_events': events
                }
            else:
                print(f"⚠ Remesh flagged but no event details found - continuing without remesh info")
                remesh_info = {'has_remesh': False, 'manual_remesh_timesteps_ms': [], 'remesh_events': []}
        else:
            print(f"✓ No remesh detected (max file size variation: {remesh_detection['max_size_variation_percent']:.1f}%, threshold: {remesh_detection['threshold_percent']:.0f}%)")
            remesh_info = {'has_remesh': False, 'manual_remesh_timesteps_ms': [], 'remesh_events': []}

        # Enhanced breathing cycle detection (supports 4 modes: A, B, C, M)
        print(f"\n🫁 Detecting breathing cycle...")

        # Check for manual override arguments
        manual_times = None
        if inhale_start is not None or transition is not None or exhale_end is not None:
            manual_times = {
                'start_s': inhale_start,
                'transition_s': transition,
                'end_s': exhale_end
            }
            if all(v is not None for v in [inhale_start, transition, exhale_end]):
                print(f"📝 Using manual override for breathing cycle times")
            else:
                print(f"⚠ Partial times provided - will prompt for missing values")
                manual_times = None

        breathing_cycle = detect_breathing_cycle_enhanced(
            flow_profile_path=flow_profile_path,
            xyz_files=xyz_files,
            timestep_info=timestep_info,
            is_complete_cycle=None,  # Will prompt user if flow profile exists
            manual_times=manual_times
        )

        start_time = breathing_cycle['start_time_ms']
        end_time = breathing_cycle['end_time_ms']
        inhale_exhale_transition = breathing_cycle['inhale_exhale_transition_ms']

        if start_time is not None and end_time is not None:
            # Filter files to breathing cycle range
            xyz_files_filtered = filter_xyz_files_by_time(xyz_files, start_time, end_time)
            if xyz_files_filtered:
                xyz_files = xyz_files_filtered
                print(f"✓ Filtered to breathing cycle: {start_time:.1f}ms - {end_time:.1f}ms")
                print(f"  {len(xyz_files)} files in breathing cycle")
            else:
                print(f"⚠ No files in breathing cycle range, using all files")

        # Convert to HDF5 with breathing cycle metadata (save in results folder)
        hdf5_file_path = f"{subject_name}_results/{subject_name}_cfd_data.h5"
        print(f"\n🚀 Converting CSV to HDF5: {hdf5_file_path}")
        try:
            data_info = auto_select_csv_to_hdf5_method(
                xyz_files, hdf5_file_path, overwrite_existing,
                breathing_cycle_start_ms=start_time,
                breathing_cycle_end_ms=end_time
            )
            print(f"✅ HDF5 conversion complete: {data_info['file_path']}")

            # Store remesh metadata in HDF5 (non-critical - don't fail pipeline if this fails)
            try:
                store_remesh_metadata(
                    hdf5_file_path,
                    has_remesh=remesh_info['has_remesh'],
                    remesh_before_file=remesh_info.get('remesh_before_file'),
                    remesh_after_file=remesh_info.get('remesh_after_file'),
                    remesh_timestep_ms=remesh_info.get('remesh_timestep_ms'),
                    remesh_events=remesh_info.get('remesh_events', [])
                )
            except Exception as e:
                print(f"⚠️  Failed to store remesh metadata in HDF5: {e}")
                print(f"   Continuing - remesh info will be in metadata.json only")

            # Store flow profile in HDF5 (so Phase 3 doesn't need the CSV file)
            if flow_profile_path:
                from data_processing.trajectory import store_flow_profile
                store_flow_profile(hdf5_file_path, flow_profile_path)

            # Create lightweight HDF5 for portable point picking
            light_hdf5_path = f"{subject_name}_results/{subject_name}_cfd_data_light.h5"
            print(f"\n📦 Creating lightweight HDF5 for point picking...")
            try:
                from data_processing.trajectory import extract_single_timestep_to_hdf5
                light_timestep_ms = start_time if start_time is not None else 0.0
                light_info = extract_single_timestep_to_hdf5(
                    source_hdf5=hdf5_file_path,
                    output_path=light_hdf5_path,
                    timestep_ms=light_timestep_ms
                )
                print(f"✅ Lightweight HDF5 created: {light_hdf5_path}")
                print(f"   Single timestep: {light_info['timestep_ms']:.1f}ms")
                print(f"   Size: {light_info['file_size_mb']:.1f}MB (portable for local point picking)")
            except Exception as e:
                print(f"⚠️  Lightweight HDF5 creation failed: {e}")
                print(f"   Point picking will use full HDF5 instead")
                light_timestep_ms = start_time if start_time is not None else 0.0

        except Exception as e:
            print(f"❌ HDF5 conversion failed: {e}")
            return

        # Create interactive HTML
        print(f"\n🎨 Creating interactive HTML visualization...")
        try:
            # Use the detected breathing cycle start time, or fallback to auto-detection
            if start_time is not None:
                display_timestep = round(start_time)
                print(f"   Using timestep: {display_timestep}ms (start of breathing cycle)")
            else:
                visualization_timestep, display_timestep = auto_detect_visualization_timestep(subject_name, flow_profile_path)
                print(f"   Using timestep: {display_timestep}ms (first file in breathing cycle)")

            if Path(hdf5_file_path).exists():
                plot_3d_interactive_all_patches(hdf5_file_path, [], subject_name, interactive_dir, time_point=display_timestep)
            elif xyz_files:
                plot_3d_interactive_all_patches(xyz_files[0], [], subject_name, interactive_dir)

            print(f"✅ Interactive HTML created in: {interactive_dir}")
        except Exception as e:
            print(f"❌ Failed to create interactive HTML: {e}")

        # Create JSON config files for Phase 2 (point picker) and Phase 3 (plotting)
        print(f"\n📋 Creating JSON config files...")
        try:
            from utils.file_processing import create_metadata_json, create_picked_points_template

            # Create metadata.json (system-generated, not user-editable)
            metadata_path = create_metadata_json(
                subject_name=subject_name,
                output_dir=results_dir,
                timestep_info=timestep_info,
                breathing_cycle=breathing_cycle,
                remesh_info=remesh_info,
                light_hdf5_timestep_ms=light_timestep_ms
            )

            # Create picked_points.json (empty template for user to fill)
            picked_points_path = create_picked_points_template(
                subject_name=subject_name,
                output_dir=results_dir,
                timestep_ms=light_timestep_ms,
                light_hdf5_filename=f"{subject_name}_cfd_data_light.h5"
            )

        except Exception as e:
            print(f"❌ Failed to create JSON config files: {e}")
            raise

        print(f"\n" + "="*60)
        print("✅ PREPARE MODE COMPLETE")
        print("="*60)
        print(f"\nOutput files created in {results_dir}/:")
        print(f"   📦 {subject_name}_cfd_data.h5           (Full HDF5 - all timesteps)")
        print(f"   📦 {subject_name}_cfd_data_light.h5     (Light HDF5 - single timestep, portable)")
        print(f"   📄 {subject_name}_metadata.json         (System metadata - do not edit)")
        print(f"   📄 {subject_name}_picked_points.json    (Point picker template - EDIT THIS)")

        print(f"\n🖥️  For LOCAL point picking (recommended for large datasets):")
        print(f"1. Copy these files to your local machine:")
        print(f"   - {subject_name}_cfd_data_light.h5")
        print(f"   - {subject_name}_picked_points.json")
        print(f"\n2. Run point picker locally:")
        print(f"   ama --subject {subject_name} --point-picker --light-h5")
        print(f"\n3. Copy {subject_name}_picked_points.json back to cluster")
        print(f"\n4. Run Phase 3 on cluster:")
        print(f"   ama --subject {subject_name} --plotting")

        print(f"\n🖥️  For DIRECT editing (if interactive HTML works):")
        print(f"1. Open the interactive HTML in your browser:")
        print(f"   {interactive_dir}/{subject_name}_surface_patches_interactive_first_breathing_cycle_t{display_timestep}ms.html")
        print(f"\n2. Hover over points to identify Patch Number and Face Index")
        print(f"\n3. Edit the picked_points.json file:")
        print(f"   {results_dir}/{subject_name}_picked_points.json")
        print(f"\n4. Run Phase 3:")
        print(f"   ama --subject {subject_name} --plotting")
        print("="*60)

        if not all_in_one_mode:
            return  # Exit early for prepare-only mode

    # Handle plotting mode (Phase 3: Generate analysis using existing HDF5 and picked points)
    if plotting_mode:
        print(f"\n📊 Plotting Mode: Phase 3 of three-phase workflow")
        print("   This mode will:")
        print("   1. Use existing HDF5 file (skip CSV processing)")
        print("   2. Load tracking locations from results folder")
        print("   3. Generate all analysis, tracking, and plots")

        # Check for HDF5 file (results folder first, then root for backwards compatibility)
        hdf5_file_path = f"{subject_name}_results/{subject_name}_cfd_data.h5"
        if not Path(hdf5_file_path).exists():
            hdf5_file_path = f"{subject_name}_cfd_data.h5"  # Backwards compatibility
        if not Path(hdf5_file_path).exists():
            print(f"❌ HDF5 file not found in:")
            print(f"   - {subject_name}_results/{subject_name}_cfd_data.h5")
            print(f"   - {subject_name}_cfd_data.h5")
            print(f"   Run --prepare first to create it")
            return

        print(f"✅ Found HDF5 file: {hdf5_file_path}")

        # Flow profile priority: 1) command line, 2) metadata.json, 3) HDF5 embedded
        from data_processing.trajectory import has_flow_profile, get_flow_profile

        if flow_profile_path:
            print(f"📊 Using flow profile from command line: {flow_profile_path}")
        else:
            # Check metadata.json for flow profile path
            metadata_path = results_dir / f"{subject_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata_json = json.load(f)
                flow_profile_from_meta = metadata_json.get('flow_profile_path')
                if flow_profile_from_meta and Path(flow_profile_from_meta).exists():
                    flow_profile_path = flow_profile_from_meta
                    print(f"📊 Using flow profile from metadata.json: {flow_profile_path}")

            # Fallback to HDF5 embedded flow profile
            if not flow_profile_path and has_flow_profile(hdf5_file_path):
                print(f"📊 Using flow profile embedded in HDF5")
                import tempfile
                flow_df = get_flow_profile(hdf5_file_path)
                if flow_df is not None:
                    temp_flow_file = Path(tempfile.gettempdir()) / f"{subject_name}_flow_profile_from_hdf5.csv"
                    flow_df.to_csv(temp_flow_file, index=False)
                    flow_profile_path = str(temp_flow_file)
                    print(f"   Extracted to: {temp_flow_file}")

            if not flow_profile_path:
                print(f"⚠️  No flow profile found in: command line, metadata.json, or HDF5")
                print(f"   Some analysis features may be limited")

        # Read remesh info from HDF5 (stored during Phase 1)
        from data_processing.trajectory import get_remesh_metadata
        remesh_metadata = get_remesh_metadata(hdf5_file_path)

        # If HDF5 says has_remesh but no events, load from metadata.json
        if remesh_metadata.get('has_remesh') and not remesh_metadata.get('remesh_events'):
            metadata_path = results_dir / f"{subject_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata_json = json.load(f)
                if 'remesh_info' in metadata_json and metadata_json['remesh_info'].get('remesh_events'):
                    remesh_metadata = metadata_json['remesh_info']
                    print(f"📊 Loaded remesh metadata from metadata.json (fallback)")

        # Handle remesh: either from HDF5 metadata or from command line flags
        should_process_remesh = remesh_metadata.get('has_remesh', False) or has_remesh

        # Check for manual remesh override first from metadata.json (not HDF5)
        # Priority: metadata.json manual_remesh_timesteps_ms > auto-detected remesh_events
        manual_ts = []
        metadata_path = results_dir / f"{subject_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_for_manual = json.load(f)
            if 'remesh_info' in metadata_for_manual:
                manual_ts = metadata_for_manual['remesh_info'].get('manual_remesh_timesteps_ms', [])

        if manual_ts and isinstance(manual_ts, list) and len(manual_ts) > 0:
            print(f"📊 Using manual remesh timesteps from metadata.json: {manual_ts}")
            remesh_events_from_hdf5 = [{'timestep_ms': float(ts)} for ts in manual_ts]
            should_process_remesh = True
        else:
            # Use auto-detected remesh events
            remesh_events_from_hdf5 = remesh_metadata.get('remesh_events', [])

        if should_process_remesh:
            print(f"\n🔄 Remesh handling enabled")
            from utils.file_processing import update_tracking_locations_for_remesh_hdf5

            # Load current tracking config first to check if mappings already exist
            tracking_config = load_and_merge_configs(subject_name=subject_name, results_dir=results_dir)

            # Check if post_remesh_list already exists for all locations
            locations = tracking_config.get('locations', [])
            has_existing_mappings = all(
                loc.get('post_remesh_list') and len(loc['post_remesh_list']) > 0
                for loc in locations
            ) if locations else False

            # Check if tracking locations are still placeholders (coordinates [0,0,0])
            all_placeholder = all(
                loc.get('coordinates', [0, 0, 0]) == [0, 0, 0] or loc.get('coordinates', [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
                for loc in locations
            ) if locations else True

            if has_existing_mappings:
                print(f"   ✅ Post-remesh mappings already exist - skipping remesh calculation")
            elif all_placeholder:
                print(f"   ⚠️  Tracking locations are placeholders - skipping remesh calculation")
                print(f"   💡 Update tracking_locations.json with real coordinates, then re-run")

        # Process remesh using HDF5 (no CSV files needed)
        if should_process_remesh and not has_existing_mappings and not all_placeholder:
            # Build remesh_events list (handles both single and multiple remesh cases)
            remesh_events = remesh_events_from_hdf5.copy() if remesh_events_from_hdf5 else []

            # For single remesh (legacy format), convert to list format
            if not remesh_events and remesh_metadata.get('has_remesh'):
                single_ts = remesh_metadata.get('remesh_timestep_ms', 0)
                if single_ts > 0:
                    remesh_events = [{'timestep_ms': single_ts}]
                    print(f"   Converting single remesh event to list format: {single_ts}ms")

            if remesh_events:
                print(f"   Processing {len(remesh_events)} remesh event(s) using HDF5 data")

                # Check if post_remesh_list already exists with correct count
                has_existing_mappings = all(
                    loc.get('post_remesh_list') and len(loc['post_remesh_list']) == len(remesh_events)
                    for loc in tracking_config.get('locations', [])
                )

                if has_existing_mappings:
                    print(f"   ✅ Post-remesh mappings already exist for all {len(remesh_events)} events")
                else:
                    # Use HDF5-based coordinate matching (no CSV files needed)
                    updated_config = update_tracking_locations_for_remesh_hdf5(
                        tracking_config, hdf5_file_path, remesh_events
                    )

                    # Save updated locations to picked_points.json
                    results_dir = Path(f'{subject_name}_results')
                    picked_points_path = results_dir / f"{subject_name}_picked_points.json"
                    if picked_points_path.exists():
                        with open(picked_points_path, 'r') as f:
                            picked_points_data = json.load(f)
                        picked_points_data['locations'] = updated_config.get('locations', [])
                        with open(picked_points_path, 'w') as f:
                            json.dump(picked_points_data, f, indent=2)
                        print(f"✅ Updated picked_points.json with post-remesh mappings")

                    # Also save post_remesh_mappings to metadata.json
                    remesh_mappings = []
                    for loc in updated_config.get('locations', []):
                        if loc.get('post_remesh_list'):
                            remesh_mappings.append({
                                'location_id': loc.get('id', loc.get('description', 'unknown')),
                                'original_patch': loc.get('patch_number'),
                                'original_face': loc.get('face_indices', [None])[0],
                                'mappings': loc['post_remesh_list']
                            })
                    if remesh_mappings:
                        save_post_remesh_mappings(subject_name, remesh_mappings, results_dir)
            else:
                print(f"   ⚠️  No remesh events found in metadata")
        else:
            print(f"   No remesh detected - using consistent patch/face indices")

        # Generate remesh comparison visualization if remesh was processed
        if should_process_remesh and locations:
            try:
                from visualization.surface_plots import visualize_remesh_comparison

                # Get remesh events from metadata or HDF5
                remesh_events = remesh_metadata.get('remesh_events', [])
                if not remesh_events and remesh_metadata.get('has_remesh'):
                    # Single remesh event from legacy format
                    remesh_events = [{
                        'timestep_boundary': remesh_metadata.get('remesh_timestep_ms', 0)
                    }]

                interactive_dir = results_dir / 'interactive'
                interactive_dir.mkdir(parents=True, exist_ok=True)

                for i, event in enumerate(remesh_events):
                    before_ms = event.get('timestep_boundary', 0) - 1
                    after_ms = event.get('timestep_boundary', 0)

                    if before_ms > 0 and after_ms > 0:
                        visualize_remesh_comparison(
                            hdf5_path=hdf5_file_path,
                            before_timestep_ms=before_ms,
                            after_timestep_ms=after_ms,
                            tracking_locations=locations,
                            subject_name=subject_name,
                            output_dir=interactive_dir,
                            remesh_event_num=i + 1
                        )
            except Exception as e:
                print(f"⚠️  Could not generate remesh comparison visualization: {e}")

        # Force processing to continue with plotting
        overwrite_existing = False  # Don't reprocess, just use existing HDF5
        # Continue to regular processing below...
        print(f"   Continuing with analysis and plotting...")

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
            print(f"   ama --raw-surface --subject {subject_name}")
        else:
            print(f"❌ Failed to create tracking locations file for {subject_name}")
    
    # Verify required files exist using smart resolution
    # For plotting mode, skip validation - HDF5 checked earlier, picked_points checked when loading config
    if not plotting_mode:
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
    # Use new format: picked_points.json + metadata.json (not legacy tracking_locations.json)
    tracking_config = load_and_merge_configs(subject_name=subject_name, results_dir=results_dir)
    if tracking_config.get('source') == 'none':
        print(f"❌ No picked_points.json found in {results_dir}")
        print(f"   Run --point-picker to select anatomical landmarks first")
        return
    tracking_locations = tracking_config['locations']
    all_files_exist = True

    # PHASE 2 PLOTTING: Always regenerate tracking CSVs to include patch analysis
    # This ensures patch radius analysis runs even if single-point CSVs exist
    if plotting_mode:
        print("📊 Plotting mode: Will regenerate all tracking data (including patch analysis)")
        all_files_exist = False  # Force regeneration

    # FRESH CASE FIX: If no tracking locations defined, force processing to create interactive HTML
    # This allows users to run pipeline on fresh cases to generate visualization for point selection
    if len(tracking_locations) == 0:
        print("📋 No tracking locations defined - fresh case detected")
        print("   Will create interactive HTML for patch/face selection")
        all_files_exist = False  # Force processing

    # Check if tracked point files exist (single point only) AND have correct format
    for location in tracking_locations:
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        single_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
        if not single_file.exists():
            all_files_exist = False
            break
        else:
            # Check if file has the Time_normalized (s) column (new format)
            try:
                header_df = pd.read_csv(single_file, nrows=0)
                if 'Time_normalized (s)' not in header_df.columns:
                    print(f"📋 CSV file missing Time_normalized column: {single_file.name}")
                    print("   Will regenerate with updated format")
                    all_files_exist = False
                    break
            except Exception:
                all_files_exist = False
                break
    
    # Set HDF5 file path before processing (check results folder first, then root for backwards compatibility)
    hdf5_file_path = f"{subject_name}_results/{subject_name}_cfd_data.h5"
    if not Path(hdf5_file_path).exists():
        legacy_path = f"{subject_name}_cfd_data.h5"
        if Path(legacy_path).exists():
            hdf5_file_path = legacy_path  # Use legacy location if it exists

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
            # Determine raw directory name
            # Priority: xyz_path > raw_dir > default ({subject}_xyz_tables)
            if xyz_path is not None:
                raw_xyz_dir = Path(xyz_path)
                print(f"📁 Using custom XYZ path: {xyz_path}")
            elif raw_dir is not None:
                raw_xyz_dir = Path(raw_dir)
                print(f"Using custom raw directory: {raw_dir}")
            else:
                raw_xyz_dir = Path(f'{subject_name}_xyz_tables')
                print(f"Using default raw directory: {raw_xyz_dir}")

            raw_xyz_files = list(raw_xyz_dir.glob('XYZ_Internal_Table_table_*.csv'))

            if raw_xyz_files:
                print(f"Found {len(raw_xyz_files)} raw XYZ files")
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

                print(f"✅ Using {len(xyz_files)} raw files for HDF5 conversion")
            else:
                print(f"No XYZ table files found in {raw_xyz_dir}")
                return
            
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

            # Find breathing cycle bounds using explicit flow profile
            start_time, end_time = find_breathing_cycle_bounds(subject_name, flow_profile_path)
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
            visualization_timestep, display_timestep = auto_detect_visualization_timestep(subject_name, flow_profile_path)
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

        # Get breathing cycle metadata for time normalization
        # Priority: 1) metadata.json (allows user override), 2) HDF5, 3) flow profile
        from data_processing.trajectory import get_breathing_cycle_metadata
        from utils.file_processing import load_metadata

        breathing_metadata = {}
        start_time = 0.0
        inhale_exhale_transition = None
        end_time = None
        breathing_source = None

        # Priority 1: Check metadata.json for user override
        metadata_json = load_metadata(subject_name, results_dir)
        if metadata_json and 'breathing_cycle' in metadata_json:
            bc = metadata_json['breathing_cycle']
            if bc.get('start_time_ms') is not None:
                start_time = bc['start_time_ms']
                inhale_exhale_transition = bc.get('inhale_exhale_transition_ms')
                end_time = bc.get('end_time_ms')
                breathing_source = f"metadata.json (mode: {bc.get('mode', 'unknown')})"
                print(f"📄 Using breathing cycle from {breathing_source}")

        # Priority 2: Fall back to HDF5
        if start_time == 0.0 and Path(hdf5_file_path).exists():
            breathing_metadata = get_breathing_cycle_metadata(hdf5_file_path)
            start_time = breathing_metadata.get('breathing_cycle_start_ms', 0.0)
            inhale_exhale_transition = breathing_metadata.get('inhale_exhale_transition_ms')
            end_time = breathing_metadata.get('breathing_cycle_end_ms')
            if start_time > 0:
                breathing_source = "HDF5 metadata"

        # Priority 3: Fall back to flow profile calculation
        if start_time == 0.0 and flow_profile_path:
            bc_start, bc_end = find_breathing_cycle_bounds(subject_name, flow_profile_path)
            if bc_start is not None:
                start_time = bc_start
                end_time = bc_end
                breathing_source = "flow profile"
                print(f"📊 Using breathing cycle start from flow profile: {start_time:.2f}ms")

        if start_time > 0:
            print(f"📊 Time normalization offset: {start_time:.2f}ms (source: {breathing_source})")
        else:
            print(f"⚠️  No breathing cycle metadata found - time normalization offset will be 0")

        # Get remesh metadata for split-chunk tracking
        from data_processing.trajectory import get_remesh_metadata
        remesh_metadata = get_remesh_metadata(hdf5_file_path) if Path(hdf5_file_path).exists() else {'has_remesh': False, 'remesh_events': []}

        # If HDF5 says has_remesh but no events, load from metadata.json
        if remesh_metadata.get('has_remesh') and not remesh_metadata.get('remesh_events'):
            if metadata_json and 'remesh_info' in metadata_json and metadata_json['remesh_info'].get('remesh_events'):
                remesh_metadata = metadata_json['remesh_info']

        has_remesh = remesh_metadata.get('has_remesh', False)
        remesh_events = remesh_metadata.get('remesh_events', [])
        remesh_timestep_ms = remesh_metadata.get('remesh_timestep_ms', None)  # Backward compat

        # Check for manual remesh override from metadata.json (priority over auto-detected)
        if metadata_json and 'remesh_info' in metadata_json:
            manual_ts = metadata_json['remesh_info'].get('manual_remesh_timesteps_ms', [])
            if manual_ts and isinstance(manual_ts, list) and len(manual_ts) > 0:
                print(f"📊 Using manual remesh timesteps from metadata.json: {manual_ts}")
                remesh_events = [{'timestep_ms': float(ts)} for ts in manual_ts]
                has_remesh = True

        if has_remesh and remesh_events:
            print(f"\n🔄 Remesh detected: {len(remesh_events)} event(s)")
            # Helper to get timestep (metadata.json uses 'timestep_boundary', HDF5 uses 'timestep_ms')
            def get_ts(evt):
                return evt.get('timestep_ms') or evt.get('timestep_boundary', 0)
            for i, event in enumerate(remesh_events, 1):
                print(f"   #{i}: boundary at {get_ts(event):.1f}ms")
            if len(remesh_events) == 1:
                print(f"   Chunk 0: t < {get_ts(remesh_events[0]):.1f}ms (original mesh)")
                print(f"   Chunk 1: t >= {get_ts(remesh_events[0]):.1f}ms (remeshed)")
            else:
                print(f"   Chunk 0: t < {get_ts(remesh_events[0]):.1f}ms (original mesh)")
                for i, event in enumerate(remesh_events[:-1]):
                    next_event = remesh_events[i+1]
                    print(f"   Chunk {i+1}: {get_ts(event):.1f}ms <= t < {get_ts(next_event):.1f}ms")
                print(f"   Chunk {len(remesh_events)}: t >= {get_ts(remesh_events[-1]):.1f}ms")

            # NOTE: Remesh boundary visualizations will be generated AFTER coordinate matching
            # so we can show the correct post-remesh mapped tracking points

        # Track which locations were updated (for coordinate auto-update)
        locations_updated = False

        for idx, location in enumerate(tracking_locations):
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']

            # VALIDATION: Skip placeholder locations (patch=0, face=0)
            if patch_number == 0 and face_index == 0:
                print(f"\n⏭️  Skipping '{description}' - placeholder values (patch=0, face=0)")
                print(f"   Update the tracking locations JSON with correct values first")
                continue

            # Check if description contains placeholder text - only warn, don't skip if valid patch/face
            if "UPDATE THIS" in description.upper():
                print(f"\n⚠️  Note: '{description}' has placeholder description")
                print(f"   Consider updating the description in tracking locations JSON")
                # Continue processing since patch/face values are valid

            print(f"\nProcessing {description}")
            print(f"Tracking point: Patch {patch_number}, Face {face_index}")

            # Process single point
            print("\nProcessing single point...")

            # Check if this location needs split-chunk tracking (has remesh and post_remesh mappings)
            post_remesh = location.get('post_remesh', None)
            post_remesh_list = location.get('post_remesh_list', [])

            # Use post_remesh_list if available, otherwise fall back to single post_remesh
            if not post_remesh_list and post_remesh:
                post_remesh_list = [post_remesh]

            use_split_chunk = has_remesh and remesh_events and len(post_remesh_list) > 0

            if use_split_chunk:
                # MULTI-CHUNK TRACKING: Use different patch/face for each mesh segment
                n_events = len(remesh_events)
                n_chunks = n_events + 1

                print(f"🔄 Multi-chunk tracking enabled ({n_chunks} chunks for {n_events} remesh event(s)):")

                all_chunk_data = []

                # Helper to get timestep from remesh event (handles both formats)
                def get_event_timestep(evt):
                    return evt.get('timestep_ms') or evt.get('timestep_boundary', 0)

                # Track each chunk with appropriate patch/face
                for chunk_idx in range(n_chunks):
                    if chunk_idx == 0:
                        # First chunk: original mesh, t < first_remesh
                        chunk_patch = patch_number
                        chunk_face = face_index
                        t_start = 0
                        t_end = get_event_timestep(remesh_events[0])
                        chunk_label = f"Chunk 0 (original)"
                    else:
                        # Subsequent chunks: use post_remesh mapping
                        if chunk_idx - 1 < len(post_remesh_list):
                            mapping = post_remesh_list[chunk_idx - 1]
                            chunk_patch = mapping.get('patch_number')
                            chunk_face = mapping.get('face_index')
                        else:
                            # Fallback if mapping missing
                            print(f"   ⚠️  Missing post_remesh mapping for chunk {chunk_idx}, using previous")
                            chunk_patch = chunk_patch  # Keep previous
                            chunk_face = chunk_face

                        t_start = get_event_timestep(remesh_events[chunk_idx - 1])
                        if chunk_idx < n_events:
                            t_end = get_event_timestep(remesh_events[chunk_idx])
                        else:
                            t_end = float('inf')
                        chunk_label = f"Chunk {chunk_idx} (after remesh #{chunk_idx})"

                    print(f"   {chunk_label}: Patch {chunk_patch}, Face {chunk_face}, {t_start:.1f}ms <= t < {t_end:.1f}ms")

                    # Track this chunk
                    if Path(hdf5_file_path).exists() and len(xyz_files) == 0:
                        chunk_data = auto_select_hdf5_point_tracking_method(hdf5_file_path, chunk_patch, chunk_face)
                    else:
                        chunk_data = track_point_parallel(xyz_files, chunk_patch, chunk_face)

                    # Filter to time range
                    if chunk_data:
                        if t_end == float('inf'):
                            filtered = [p for p in chunk_data if p.get('time_ms', 0) >= t_start]
                        else:
                            filtered = [p for p in chunk_data if t_start <= p.get('time_ms', 0) < t_end]
                        print(f"      → {len(filtered)} timesteps")
                        all_chunk_data.extend(filtered)
                    else:
                        print(f"      → Warning: No data found")

                # Merge and sort all chunks
                single_point_data = all_chunk_data
                single_point_data.sort(key=lambda p: p.get('time_ms', 0))

                print(f"   Total merged: {len(single_point_data)} timesteps")
            else:
                # Standard single-chunk tracking (no remesh)
                if Path(hdf5_file_path).exists() and len(xyz_files) == 0:
                    single_point_data = auto_select_hdf5_point_tracking_method(hdf5_file_path, patch_number, face_index)
                else:
                    single_point_data = track_point_parallel(xyz_files, patch_number, face_index)

            if single_point_data:
                print(f"Tracked single point through {len(single_point_data)} time steps")
                save_trajectory_data(single_point_data, subject_name, patch_number, face_index, description, is_region=False, time_offset_ms=start_time)

                # AUTO-UPDATE COORDINATES: Extract from first time point
                first_point = single_point_data[0]
                new_coords = [first_point['x'], first_point['y'], first_point['z']]
                old_coords = location.get('coordinates', [0.0, 0.0, 0.0])

                # Update if coordinates were placeholder or missing
                if old_coords == [0.0, 0.0, 0.0] or old_coords != new_coords:
                    tracking_config['locations'][idx]['coordinates'] = new_coords
                    locations_updated = True
                    print(f"   📍 Updated coordinates: ({new_coords[0]:.6f}, {new_coords[1]:.6f}, {new_coords[2]:.6f})")
            else:
                print(f"Warning: No trajectory found for single point")
            
            # Process patch regions if enabled
            if enable_patch_analysis:
                # Note: Patch region analysis is skipped for remeshed locations
                # because the spatial relationship of nearby points changes across remesh
                if use_split_chunk:
                    print(f"\n⚠️  Skipping patch region analysis for '{description}' - location has remesh")
                    print(f"   Patch analysis tracks neighboring points which change across remesh")
                    print(f"   Single point tracking above handles remesh correctly")
                    continue  # Skip to next location

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
                        save_trajectory_data(patch_data, subject_name, patch_number, face_index, patch_description, is_region=True, time_offset_ms=start_time)
                    else:
                        print(f"  Warning: No trajectory found for {radius_mm:.1f}mm patch")
        
        # Save updated tracking config if coordinates were updated
        if locations_updated:
            # Save to results folder using new format (picked_points.json)
            updated_json_path = results_dir / f"{subject_name}_picked_points.json"
            with open(updated_json_path, 'w') as f:
                json.dump(tracking_config, f, indent=2)
            print(f"\n📋 Updated picked points saved with coordinates: {updated_json_path}")

        # Regenerate remesh visualizations NOW that tracking_locations have post_remesh_list populated
        if has_remesh and remesh_events and tracking_locations:
            print(f"\n📊 Generating remesh boundary visualizations with mapped points...")
            from data_processing.trajectory import extract_single_timestep_to_hdf5

            # Helper to get timestep from remesh event
            def get_remesh_ts(evt):
                return evt.get('timestep_ms') or evt.get('timestep_boundary', 0)

            # Helper to build tracking_locations for a specific chunk
            def get_locations_for_chunk(chunk_idx, locations):
                """Return tracking locations appropriate for the given mesh chunk."""
                if chunk_idx == 0:
                    # Original mesh - use original locations
                    return locations
                else:
                    # Post-remesh chunk - use post_remesh_list[chunk_idx - 1]
                    chunk_locations = []
                    for loc in locations:
                        post_list = loc.get('post_remesh_list', [])
                        if chunk_idx - 1 < len(post_list):
                            mapping = post_list[chunk_idx - 1]
                            chunk_locations.append({
                                'description': loc['description'],
                                'patch_number': mapping.get('patch_number', 0),
                                'face_indices': [mapping.get('face_index', 0)],
                                'coordinates': mapping.get('coordinates', loc.get('coordinates', [0, 0, 0]))
                            })
                        else:
                            # No mapping for this chunk - skip or use original
                            chunk_locations.append(loc)
                    return chunk_locations

            # Determine chunk boundaries
            chunk_boundaries = [get_remesh_ts(evt) for evt in remesh_events]

            # Generate HTML for each remesh boundary (before and after)
            for event_idx, event in enumerate(remesh_events):
                boundary_ts = int(get_remesh_ts(event))
                before_ts = boundary_ts - 1

                # Before remesh: chunk = event_idx (0 for first remesh)
                # After remesh: chunk = event_idx + 1
                before_chunk = event_idx
                after_chunk = event_idx + 1

                before_locations = get_locations_for_chunk(before_chunk, tracking_locations)
                after_locations = get_locations_for_chunk(after_chunk, tracking_locations)

                # Generate BEFORE visualization
                try:
                    temp_h5_before = f"{subject_name}_results/{subject_name}_temp_t{before_ts}ms.h5"
                    if Path(hdf5_file_path).exists():
                        extract_single_timestep_to_hdf5(hdf5_file_path, temp_h5_before, timestep_ms=before_ts)
                        if Path(temp_h5_before).exists():
                            plot_3d_interactive_all_patches(
                                temp_h5_before, before_locations, subject_name,
                                interactive_dir, time_point=before_ts
                            )
                            Path(temp_h5_before).unlink()
                            print(f"   ✅ t={before_ts}ms (before remesh {event_idx+1}, chunk {before_chunk})")
                except Exception as e:
                    print(f"   ⚠️  Error for t={before_ts}ms: {e}")

                # Generate AFTER visualization
                try:
                    temp_h5_after = f"{subject_name}_results/{subject_name}_temp_t{boundary_ts}ms.h5"
                    if Path(hdf5_file_path).exists():
                        extract_single_timestep_to_hdf5(hdf5_file_path, temp_h5_after, timestep_ms=boundary_ts)
                        if Path(temp_h5_after).exists():
                            plot_3d_interactive_all_patches(
                                temp_h5_after, after_locations, subject_name,
                                interactive_dir, time_point=boundary_ts
                            )
                            Path(temp_h5_after).unlink()
                            print(f"   ✅ t={boundary_ts}ms (after remesh {event_idx+1}, chunk {after_chunk})")
                except Exception as e:
                    print(f"   ⚠️  Error for t={boundary_ts}ms: {e}")

            # Regenerate remesh comparison HTMLs with updated locations
            try:
                from visualization.surface_plots import visualize_remesh_comparison
                for i, event in enumerate(remesh_events):
                    before_ms = int(get_remesh_ts(event)) - 1
                    after_ms = int(get_remesh_ts(event))
                    if before_ms > 0 and after_ms > 0:
                        visualize_remesh_comparison(
                            hdf5_path=hdf5_file_path,
                            before_timestep_ms=before_ms,
                            after_timestep_ms=after_ms,
                            tracking_locations=tracking_locations,  # Now has post_remesh_list populated
                            subject_name=subject_name,
                            output_dir=interactive_dir,
                            remesh_event_num=i + 1
                        )
                print(f"   ✅ Remesh comparison HTMLs regenerated with mapped points")
            except Exception as e:
                print(f"   ⚠️  Error regenerating comparison HTMLs: {e}")

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
    
    # Find reference location data for key time point analysis
    # First try "Posterior border of soft palate", otherwise use the first available location
    reference_data = None
    reference_name = None
    for location in tracking_locations:
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        key = description.lower().replace(' ', '_')
        data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{key}.csv"

        if data_file.exists():
            # Prefer "Posterior border of soft palate" if it exists
            if 'soft palate' in description.lower() or 'palate' in description.lower():
                reference_data = pd.read_csv(data_file)
                reference_name = description
                print(f"Using '{description}' as reference for key time points")
                break
            # Otherwise keep the first valid location as fallback
            elif reference_data is None:
                reference_data = pd.read_csv(data_file)
                reference_name = description

    if reference_data is not None and reference_name is not None:
        print(f"Loaded reference data from '{reference_name}' for key time point analysis")
    else:
        print(f"Warning: No reference data found for key time point analysis")
    
    # Initialize variables to store key time points
    key_time_points = {
        'inhalation_max_negative_adotn': None,
        'exhalation_max_pressure': None
    }
    
    # Find time point during inhalation with maximum negative AdotN at the reference location
    if reference_data is not None and inhale_exhale_transition is not None:
        # Calculate AdotN from VdotN
        times = reference_data['Time (s)'].values
        vdotn = reference_data['VdotN'].values * 1000  # Convert from m/s to mm/s
        
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
                # Generate both original and normalized time plots
                create_metrics_vs_time_plot(df, subject_name, description, patch_number, face_index, metrics_vs_time_pdf, use_normalized_time=False, flow_profile_path=flow_profile_path)
                if 'Time_normalized (s)' in df.columns:
                    create_metrics_vs_time_plot(df, subject_name, description, patch_number, face_index, metrics_vs_time_pdf, use_normalized_time=True, flow_profile_path=flow_profile_path)
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
                        # Generate both original and normalized time plots
                        create_metrics_vs_time_plot(df, subject_name, patch_description, patch_number, face_index, metrics_vs_time_pdf, use_normalized_time=False, flow_profile_path=flow_profile_path)
                        if 'Time_normalized (s)' in df.columns:
                            create_metrics_vs_time_plot(df, subject_name, patch_description, patch_number, face_index, metrics_vs_time_pdf, use_normalized_time=True, flow_profile_path=flow_profile_path)
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
        locations_for_panel = []  # Dynamic list of locations from JSON

        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']

            # Create a key from the description (lowercase with underscores)
            key = description.lower().replace(' ', '_')

            # Load trajectory data
            data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{key}.csv"

            if not data_file.exists():
                print(f"Warning: No data file found for {description}")
                continue

            df = pd.read_csv(data_file)
            dfs[key] = df
            locations_for_panel.append({'key': key, 'description': description})

        # Create symmetric comparison panel if we have data
        if len(dfs) >= 1:
            print(f"Creating symmetric comparison panels for {subject_name} with {len(locations_for_panel)} locations...")
            # 1. Clean version (no markers)
            create_symmetric_comparison_panel_clean(dfs, subject_name, pdf, pdfs_dir, locations_for_panel)

            # 2. Detailed version (with zero-crossing markers)
            create_symmetric_comparison_panel(dfs, subject_name, pdf, pdfs_dir, locations_for_panel)

            # 3. Smoothed version (with moving average)
            create_symmetric_comparison_panel_smooth(dfs, subject_name, pdf, pdfs_dir, locations_for_panel)

            # 4. Smoothed version with zero-crossing markers (normalized time - starts at 0)
            create_symmetric_comparison_panel_smooth_with_markers(dfs, subject_name, pdf, pdfs_dir, locations_for_panel, use_original_time=False)

            # 5. Smoothed version with zero-crossing markers (original time - for traceability)
            create_symmetric_comparison_panel_smooth_with_markers(dfs, subject_name, None, pdfs_dir, locations_for_panel, use_original_time=True)

            print(f"Symmetric comparison panels completed.")
            
            # 6. CFD Analysis 3x3 panels (new)
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
                    # Create CFD analysis 3x3 panel (smoothed, no markers) - shared scale
                    create_cfd_analysis_3x3_panel(single_point_dfs, patch_dfs, subject_name, pdf, pdfs_dir)

                    # Create CFD analysis 3x3 panel with markers (smoothed, with markers) - shared scale
                    create_cfd_analysis_3x3_panel_with_markers(single_point_dfs, patch_dfs, subject_name, pdf, pdfs_dir)

                    # Create CFD analysis 3x3 panels with original data scale (each subplot auto-scaled)
                    create_cfd_analysis_3x3_panel_original_scale(single_point_dfs, patch_dfs, subject_name, pdf, pdfs_dir)
                    create_cfd_analysis_3x3_panel_with_markers_original_scale(single_point_dfs, patch_dfs, subject_name, pdf, pdfs_dir)

                    # Create CFD analysis 3x3 panels with both normalized and original TIME versions
                    create_cfd_analysis_3x3_panel_with_markers_both_time_versions(single_point_dfs, patch_dfs, subject_name, pdfs_dir)

                    print(f"CFD Analysis 3x3 panels completed (shared scale + original data scale + time versions).")
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
    
    # Generate flow profile visualizations (both clean and original)
    create_clean_flow_profile_plot(subject_name, hdf5_file_path, results_dir, figures_dir, pdfs_dir)
    create_original_flow_profile_plot(subject_name, hdf5_file_path, results_dir, figures_dir, pdfs_dir)
    
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
                # Use consistent timestep detection logic (pass flow_profile_path for consistency)
                visualization_timestep, display_timestep = auto_detect_visualization_timestep(subject_name, flow_profile_path)
                
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

def cli_main():
    """Entry point for the CLI tool (`ama` command)."""
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

═══════════════════════════════════════════════════════════════════════════════
THREE-PHASE WORKFLOW (RECOMMENDED FOR NEW SUBJECTS)
═══════════════════════════════════════════════════════════════════════════════

Phase 1: PREPARE - Convert CSV to HDF5, detect breathing cycle/remesh, create templates
   ama --subject OSAMRI007 --prepare --flow-profile OSAMRI007FlowProfile_smoothed.csv

   → Converts raw CSV files to HDF5 format (85% size reduction)
   → Detects breathing cycle from flow profile
   → Auto-detects remesh events
   → Creates interactive HTML and template JSON in {SUBJECT}_results/

Phase 2: PICK POINTS - Select anatomical landmarks on the airway surface
   ama --subject OSAMRI007 --point-picker

   → Opens PyVista GUI for interactive 3D point selection
   → Click on airway surface to pick tracking locations
   → Saves selected points to {SUBJECT}_results/{SUBJECT}_picked_points.json
   → Alternative: edit picked_points.json manually using the interactive HTML

Phase 3: PLOTTING - Generate all analysis and plots
   ama --subject OSAMRI007 --plotting

   → Uses existing HDF5 file (skips CSV processing)
   → Loads tracking locations from picked_points.json
   → Generates all analysis, tracking, and visualization outputs
   → Creates both normalized and original time PDFs for traceability

═══════════════════════════════════════════════════════════════════════════════

DEMO USAGE SCENARIOS:

1. 🚀 COMPLETE ANALYSIS IN ONE PASS (requires pre-configured tracking):
   ama --subject OSAMRI007 --all --flow-profile OSAMRI007FlowProfile_smoothed.csv

   → Runs prepare + tracking + plotting in one pass
   → Outputs: HDF5, CSV data, PDF reports, PNG figures, HTML visualizations
   → Time: ~5-10 minutes for full analysis

2. 🔄 FORCE COMPLETE RERUN (overwrite existing data):
   ama --subject OSAMRI007 --forcererun

   → Reprocesses everything even if files exist
   → Use when you want fresh analysis or changed parameters

3. 🎨 HIGHLIGHT PATCH REGIONS:
   ama --subject OSAMRI007 --highlight-patches --patch-timestep 100
   
   → Quick visualization of patch regions around tracking points (100 = 0.1s)
   → Shows 1mm, 2mm, 5mm patches around anatomical landmarks
   → Time: ~30 seconds

4. 🌊 RAW SURFACE VISUALIZATION:
   ama --subject OSAMRI007 --raw-surface --surface-timestep 50
   
   → Shows complete raw airway surface for manual point selection
   → Use for selecting new tracking locations interactively
   → Time: ~1 minute

5. 🎯 INTERACTIVE POINT SELECTOR:
   ama --subject OSAMRI007 --interactive-selector --selector-timestep 100
   
   → Launch 3D point picker for selecting new tracking locations
   → Click points to add to tracking locations JSON
   → Requires PyVista

6. 📊 CUSTOM PATCH ANALYSIS:
   ama --subject OSAMRI007 --patchradii 1.0 3.0 7.0 --normalangle 45.0
   
   → Custom patch sizes (1mm, 3mm, 7mm) and surface filtering (45°)
   → Analyzes different region sizes around tracking points

7. 🔍 LIST AVAILABLE SUBJECTS:
   ama --listsubjects
   
   → Shows all detected subjects and their status
   → Checks for required flow profile files

8. ⚡ ANALYSIS WITHOUT VISUALIZATION:
   ama --subject OSAMRI007 --disablevisualization
   
   → Faster processing, generates all PDFs but no interactive HTML
   → Good for batch processing or when visualization not needed

9. 📈 SINGLE POINT ANALYSIS ONLY:
   ama --subject OSAMRI007 --disablepatchanalysis
   
   → Analyzes only individual tracking points (no patch regions)
   → Faster processing, smaller output files

REQUIRED INPUT:
- {SUBJECT}_xyz_tables/              (CFD geometry CSV files)
- {SUBJECT}FlowProfile_smoothed.csv  (breathing flow data, optional for Phase 1)

OUTPUTS STRUCTURE:
{SUBJECT}_results/                              (self-contained results folder)
├── {SUBJECT}_cfd_data.h5                       (HDF5 cache - all timesteps)
├── {SUBJECT}_cfd_data_light.h5                 (Light HDF5 - single timestep, portable)
├── {SUBJECT}_picked_points.json                (tracking locations - EDIT THIS)
├── {SUBJECT}_metadata.json                     (system metadata)
├── {SUBJECT}_key_time_points.json              (breathing cycle analysis)
├── tracked_points/                             (CSV trajectory data)
├── figures/                                    (PNG images)
├── reports/                                    (PDF analysis reports)
└── interactive/                                (HTML visualizations)

ENVIRONMENT:
Setup: bash setup.sh   (or: conda env create -f environment.yml && conda activate ama && pip install .)
'''

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=CustomHelpFormatter,
        epilog='''
Examples (Three-Phase Workflow - Recommended):
  # Phase 1: Create HDF5 and interactive HTML
  ama --subject OSAMRI007 --prepare --flow-profile OSAMRI007FlowProfile_smoothed.csv

  # Phase 1 with manual breathing cycle times (skips auto-detection)
  ama --subject OSAMRI007 --prepare --flow-profile OSAMRI007FlowProfile_smoothed.csv --inhale-start 0.05 --transition 1.0 --exhale-end 2.2

  # Phase 2: Pick anatomical landmarks (opens PyVista GUI)
  ama --subject OSAMRI007 --point-picker

  # Phase 2 with custom H5 file
  ama --subject OSAMRI007 --point-picker --h5-file OSAMRI007_results/OSAMRI007_cfd_data_light.h5

  # Phase 3: Generate all analysis and plots
  ama --subject OSAMRI007 --plotting

Examples (Custom XYZ Path - Data stored elsewhere):
  # Phase 1 with custom XYZ path (results still saved to ./OSAMRI007_results/)
  ama --subject OSAMRI007 --prepare \\
      --xyz-path /data/cfd_simulations/OSAMRI007/xyz_tables \\
      --flow-profile /data/cfd_simulations/OSAMRI007/FlowProfile_smoothed.csv

  # Phase 3 with custom XYZ path
  ama --subject OSAMRI007 --plotting \\
      --xyz-path /data/cfd_simulations/OSAMRI007/xyz_tables

Examples (Other Commands):
  ama --subject OSAMRI007                    # Full analysis (legacy)
  ama --subject OSAMRI007 --forcererun      # Force complete rerun
  ama --highlight-patches --patch-timestep 100  # Highlight patch regions
  ama --listsubjects                         # Show available data
        '''
    )
    
    parser.add_argument('--subject', type=str,
                      help='Subject name to process (default: auto-detect from existing folders)')
    parser.add_argument('--rawdir', type=str,
                      help='Custom raw data directory name (default: {subject}_xyz_tables). Use this to specify a different raw data directory like "qDNS_xyz_tables".')
    parser.add_argument('--xyz-path', type=str, dest='xyz_path',
                      help='Full path to XYZ tables directory (absolute or relative). Takes priority over --rawdir. Results still saved to {subject}_results/ in current directory.')
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

    # Two-phase pipeline arguments
    parser.add_argument('--prepare', action='store_true',
                      help='Phase 1: Convert CSV to HDF5, detect breathing cycle/remesh, create templates')
    parser.add_argument('--plotting', action='store_true',
                      help='Phase 3: Generate analysis and plots using existing HDF5 and picked points')
    parser.add_argument('--all', action='store_true',
                      help='Run complete pipeline: prepare + tracking + plotting in one pass (requires pre-configured tracking locations)')

    # Input files for production workflow
    parser.add_argument('--flow-profile', type=str,
                      help='Path to breathing flow profile CSV file (optional for Phase 1 - will use CSV timestep if not provided)')

    # Breathing cycle manual override (all times in SECONDS)
    parser.add_argument('--inhale-start', type=float, metavar='SEC',
                      help='Manual override: Start of inhale in seconds (skips auto-detection)')
    parser.add_argument('--transition', type=float, metavar='SEC',
                      help='Manual override: Inhale-to-exhale transition in seconds')
    parser.add_argument('--exhale-end', type=float, metavar='SEC',
                      help='Manual override: End of exhale in seconds')

    # Remesh handling arguments (now auto-detected, these flags are for manual override)
    parser.add_argument('--has-remesh', action='store_true',
                      help='Override auto-detection: indicate simulation has mesh remeshing')
    parser.add_argument('--remesh-before', type=str,
                      help='Override auto-detection: CSV filename of last timestep BEFORE remesh')
    parser.add_argument('--remesh-after', type=str,
                      help='Override auto-detection: CSV filename of first timestep AFTER remesh')
    parser.add_argument('--point-picker', action='store_true',
                      help='Phase 2: Launch PyVista GUI to select anatomical landmarks on airway surface')
    parser.add_argument('--h5-file', type=str,
                      help='Specify HDF5 file path for point picker (e.g., OSAMRI007_results/OSAMRI007_cfd_data_light.h5)')
    parser.add_argument('--picker-timestep', type=int,
                      help='Specific timestep for point picker (optional, will prompt if not provided)')

    args = parser.parse_args()

    # Validate subject's xyz_tables directory exists (early check before any processing)
    if args.subject and not args.listsubjects:
        xyz_path_arg = getattr(args, 'xyz_path', None)
        rawdir_arg = getattr(args, 'rawdir', None)
        if not xyz_path_arg and not rawdir_arg:
            xyz_dir = Path(f'{args.subject}_xyz_tables')
            if not xyz_dir.exists():
                print(f"❌ ERROR: No input data folder found for subject '{args.subject}'")
                print(f"   Expected: {xyz_dir}/")
                print(f"   Available subjects:")
                available = list(Path('.').glob('*_xyz_tables'))
                if available:
                    for d in sorted(available):
                        name = d.name.replace('_xyz_tables', '')
                        print(f"     • {name}")
                else:
                    print(f"     (none found in current directory)")
                sys.exit(1)

    # Validate --flow-profile for production modes (now optional for Phase 1)
    if getattr(args, 'prepare', False) or getattr(args, 'all', False):
        # Phase 1 / All mode: flow profile is optional (will use Mode C if not provided)
        if args.flow_profile:
            flow_profile_path = Path(args.flow_profile)
            if not flow_profile_path.exists():
                print(f"❌ ERROR: Flow profile file not found: {args.flow_profile}")
                sys.exit(1)
            print(f"✅ Using flow profile: {args.flow_profile}")
        else:
            print("ℹ️  No flow profile provided - will use CSV timestep data for breathing cycle detection")
    elif getattr(args, 'plotting', False):
        # Phase 3: flow profile can come from HDF5 or command line
        if args.flow_profile:
            flow_profile_path = Path(args.flow_profile)
            if not flow_profile_path.exists():
                print(f"❌ ERROR: Flow profile file not found: {args.flow_profile}")
                sys.exit(1)
            print(f"✅ Using flow profile: {args.flow_profile}")
        else:
            # Check if HDF5 has embedded flow profile
            from data_processing.trajectory import has_flow_profile
            hdf5_file = Path(f"{args.subject}_results/{args.subject}_cfd_data.h5")
            if hdf5_file.exists() and has_flow_profile(str(hdf5_file)):
                print(f"✅ Flow profile will be extracted from HDF5")
            else:
                print("❌ ERROR: --flow-profile is required (not found in HDF5)")
                print("   Example: ama --subject OSAMRI007 --plotting --flow-profile OSAMRI007FlowProfile.csv")
                sys.exit(1)

    # Handle list subjects command
    if args.listsubjects:
        subjects = detect_available_subjects()
        if subjects:
            print("📋 Available subjects:")
            for subject in subjects:
                print(f"  • {subject}")
                # Check for flow profile with base subject fallback (quiet check)
                base = extract_base_subject(subject)
                flow_candidates = [
                    Path(f"{subject}FlowProfile.csv"),
                    Path(f"{subject}FlowProfile_smoothed.csv"),
                    Path(f"{base}FlowProfile.csv"),
                    Path(f"{base}FlowProfile_smoothed.csv"),
                ]
                flow_file = next((f for f in flow_candidates if f.exists()), None)
                if flow_file:
                    if base != subject and not flow_file.name.startswith(subject):
                        print(f"    ✅ Flow profile found (via base subject {base}): {flow_file.name}")
                    else:
                        print(f"    ✅ Flow profile found: {flow_file.name}")
                else:
                    print(f"    ❌ Flow profile missing")
        else:
            print("❌ No subjects found (no folders matching *_xyz_tables pattern)")
        sys.exit(0)

    # Handle point picker mode
    if getattr(args, 'point_picker', False):
        if not args.subject:
            print("❌ ERROR: --subject is required for --point-picker mode")
            sys.exit(1)

        results_dir = Path(f"{args.subject}_results")

        if not results_dir.exists():
            print(f"❌ ERROR: Results directory not found: {results_dir}")
            print("   Run --prepare first to create HDF5 cache")
            sys.exit(1)

        # Check for HDF5 file (custom path or default)
        h5_file_arg = getattr(args, 'h5_file', None)
        if h5_file_arg:
            h5_file = Path(h5_file_arg)
            if not h5_file.exists():
                print(f"❌ ERROR: Specified HDF5 file not found: {h5_file}")
                sys.exit(1)
        else:
            # Prefer light H5 (single timestep, ~37MB) over full H5 (~8GB)
            h5_light = results_dir / f"{args.subject}_cfd_data_light.h5"
            h5_full = results_dir / f"{args.subject}_cfd_data.h5"
            if h5_light.exists():
                h5_file = h5_light
            elif h5_full.exists():
                h5_file = h5_full
            else:
                print(f"❌ ERROR: No HDF5 cache found in {results_dir}")
                print("   Run --prepare first to create HDF5 cache")
                sys.exit(1)

        print(f"📦 Loading HDF5: {h5_file}")

        # Try Qt GUI first, fall back to simple picker
        try:
            from visualization.point_picker_gui import run_point_picker_gui
            run_point_picker_gui(args.subject, args.picker_timestep, results_dir, h5_path=h5_file)
        except ImportError as e:
            print(f"⚠️ Qt GUI not available ({e}), using simple picker")
            from visualization.point_picker import run_point_picker
            run_point_picker(args.subject, args.picker_timestep, results_dir)
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
         xyz_path=getattr(args, 'xyz_path', None),
         highlight_patches=getattr(args, 'highlight_patches', False),
         patch_timestep=getattr(args, 'patch_timestep', 100),
         raw_surface=getattr(args, 'raw_surface', False),
         surface_timestep=args.surface_timestep,
         interactive_selector=getattr(args, 'interactive_selector', False),
         selector_timestep=args.selector_timestep,
         smoothing_window=getattr(args, 'smoothing_window', 20),
         patch_selection_mode=getattr(args, 'prepare', False),
         plotting_mode=getattr(args, 'plotting', False),
         all_in_one_mode=getattr(args, 'all', False),
         has_remesh=getattr(args, 'has_remesh', False),
         remesh_before=getattr(args, 'remesh_before', None),
         remesh_after=getattr(args, 'remesh_after', None),
         flow_profile_path=getattr(args, 'flow_profile', None),
         inhale_start=getattr(args, 'inhale_start', None),
         transition=getattr(args, 'transition', None),
         exhale_end=getattr(args, 'exhale_end', None))


if __name__ == '__main__':
    cli_main()