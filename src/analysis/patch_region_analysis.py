#!/usr/bin/env python3
"""
Patch-based surface region analysis module.
Extends single-point tracking to include averaged patch regions around points.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

def find_points_in_circle_enhanced(df: pd.DataFrame, 
                                 center_point: Tuple[float, float, float], 
                                 radius: float, 
                                 patch_number: int,
                                 min_normal_alignment: float = 0.7) -> pd.DataFrame:
    """
    Enhanced version of find_points_in_circle with adjustable parameters.
    
    Args:
        df: DataFrame containing point coordinates and data
        center_point: (x, y, z) coordinates of the center point
        radius: Radius of the circular region in meters
        patch_number: The patch number of the center point
        min_normal_alignment: Minimum dot product for normal vector alignment (0.0-1.0)
        
    Returns:
        DataFrame containing points within the circular region on the same surface side
    """
    x, y, z = center_point
    
    # Get the center point's data to determine its normal direction
    center_data = df[(df['X (m)'] == x) & (df['Y (m)'] == y) & (df['Z (m)'] == z)]
    if len(center_data) == 0:
        raise ValueError("Center point not found in data")
    
    # Expand search to include nearby patches (adjustable range)
    patch_range = max(2, int(radius * 1000))  # Adaptive patch range based on radius
    nearby_patches = list(range(patch_number - patch_range, patch_number + patch_range + 1))
    df_nearby = df[df['Patch Number'].isin(nearby_patches)]
    
    # Get the area components and normalize them to create a unit normal vector for center point
    center_area = np.array([
        center_data['Area[i] (m^2)'].iloc[0],
        center_data['Area[j] (m^2)'].iloc[0],
        center_data['Area[k] (m^2)'].iloc[0]
    ])
    center_normal = center_area / np.linalg.norm(center_area)
    
    # Create two orthogonal vectors in the tangent plane
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
    normal_alignment = np.sum(point_normals * center_normal, axis=1)
    same_side = normal_alignment > min_normal_alignment
    
    # Filter points that are both within radius and have aligned normals
    return df_nearby[(distances_in_plane <= radius) & same_side]

def calculate_patch_statistics(region_df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive statistics for a patch region.
    
    Args:
        region_df: DataFrame containing points in the region
        
    Returns:
        Dictionary with statistical measures
    """
    if len(region_df) == 0:
        return {
            'num_points': 0,
            'pressure': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'velocity': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan},
            'vdotn': {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        }
    
    return {
        'num_points': len(region_df),
        'pressure': {
            'mean': region_df['Total Pressure (Pa)'].mean(),
            'std': region_df['Total Pressure (Pa)'].std(),
            'min': region_df['Total Pressure (Pa)'].min(),
            'max': region_df['Total Pressure (Pa)'].max()
        },
        'velocity': {
            'mean': region_df['Velocity: Magnitude (m/s)'].mean(),
            'std': region_df['Velocity: Magnitude (m/s)'].std(),
            'min': region_df['Velocity: Magnitude (m/s)'].min(),
            'max': region_df['Velocity: Magnitude (m/s)'].max()
        },
        'vdotn': {
            'mean': region_df['VdotN'].mean(),
            'std': region_df['VdotN'].std(),
            'min': region_df['VdotN'].min(),
            'max': region_df['VdotN'].max()
        }
    }

def analyze_multi_radius_patches(df: pd.DataFrame, 
                               center_point: Tuple[float, float, float],
                               patch_number: int,
                               radii: List[float] = [0.001, 0.002, 0.005]) -> Dict:
    """
    Analyze patch regions at multiple radii around a center point.
    
    Args:
        df: DataFrame containing all point data
        center_point: (x, y, z) coordinates of center point
        patch_number: Patch number of center point
        radii: List of radii to analyze (in meters)
        
    Returns:
        Dictionary with results for each radius
    """
    results = {}
    
    for radius in radii:
        radius_mm = radius * 1000
        print(f"  Analyzing {radius_mm:.1f}mm radius patch...")
        
        # Find points in this radius
        region_df = find_points_in_circle_enhanced(df, center_point, radius, patch_number)
        
        # Calculate statistics
        stats = calculate_patch_statistics(region_df)
        
        results[f'{radius_mm:.1f}mm'] = {
            'radius_m': radius,
            'radius_mm': radius_mm,
            'region_data': region_df,
            'statistics': stats
        }
        
        print(f"    Found {stats['num_points']} points")
        if stats['num_points'] > 0:
            print(f"    Mean pressure: {stats['pressure']['mean']:.2f} ± {stats['pressure']['std']:.2f} Pa")
            print(f"    Mean velocity: {stats['velocity']['mean']:.6f} ± {stats['velocity']['std']:.6f} m/s")
    
    return results

def create_patch_averaged_trajectory(tracked_data: pd.DataFrame,
                                   original_csv_files: List[Path],
                                   patch_number: int,
                                   face_index: int,
                                   radii: List[float] = [0.001, 0.002, 0.005]) -> Dict:
    """
    Create patch-averaged trajectories for multiple radii by processing all time steps.
    
    Args:
        tracked_data: DataFrame with single-point trajectory data
        original_csv_files: List of paths to original CSV files for each time step
        patch_number: Patch number of center point
        face_index: Face index of center point
        radii: List of radii to analyze
        
    Returns:
        Dictionary with patch-averaged trajectories for each radius
    """
    patch_trajectories = {f'{r*1000:.1f}mm': [] for r in radii}
    
    print(f"Creating patch-averaged trajectories for Patch {patch_number}, Face {face_index}")
    
    for time_idx, csv_file in enumerate(original_csv_files):
        if time_idx >= len(tracked_data):
            break
            
        # Get center point coordinates for this time step
        center_row = tracked_data.iloc[time_idx]
        center_point = (center_row['X (m)'], center_row['Y (m)'], center_row['Z (m)'])
        time_point = center_row['Time (s)']
        
        # Load full CSV data for this time step
        df_full = pd.read_csv(csv_file, low_memory=False)
        
        # Analyze all radii for this time step
        multi_radius_results = analyze_multi_radius_patches(df_full, center_point, patch_number, radii)
        
        # Store results for each radius
        for radius_key, radius_data in multi_radius_results.items():
            stats = radius_data['statistics']
            
            trajectory_point = {
                'Time (s)': time_point,
                'Time Point': center_row['Time Point'],
                'X (m)': center_point[0],  # Keep center point coordinates
                'Y (m)': center_point[1],
                'Z (m)': center_point[2],
                'Num_Points': stats['num_points'],
                'Pressure_Mean (Pa)': stats['pressure']['mean'],
                'Pressure_Std (Pa)': stats['pressure']['std'],
                'Velocity_Mean (m/s)': stats['velocity']['mean'],
                'Velocity_Std (m/s)': stats['velocity']['std'],
                'VdotN_Mean': stats['vdotn']['mean'],
                'VdotN_Std': stats['vdotn']['std']
            }
            
            patch_trajectories[radius_key].append(trajectory_point)
    
    # Convert to DataFrames
    for radius_key in patch_trajectories:
        patch_trajectories[radius_key] = pd.DataFrame(patch_trajectories[radius_key])
        print(f"  {radius_key} patch trajectory: {len(patch_trajectories[radius_key])} time points")
    
    return patch_trajectories

def save_patch_trajectories(patch_trajectories: Dict, 
                          subject_name: str, 
                          patch_number: int, 
                          face_index: int, 
                          description: str,
                          output_dir: Path):
    """
    Save patch-averaged trajectories to CSV files.
    
    Args:
        patch_trajectories: Dictionary of DataFrames for each radius
        subject_name: Subject identifier
        patch_number: Patch number
        face_index: Face index  
        description: Point description
        output_dir: Output directory for CSV files
    """
    output_dir = Path(output_dir)
    patch_dir = output_dir / "patch_averaged"
    patch_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_').replace(',', '')}"
    
    for radius_key, trajectory_df in patch_trajectories.items():
        filename = f"{base_filename}_patch_{radius_key}.csv"
        filepath = patch_dir / filename
        
        trajectory_df.to_csv(filepath, index=False)
        print(f"Saved patch trajectory: {filepath}") 