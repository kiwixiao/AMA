"""
CFD Analysis Comparison Tool

This tool compares CFD analysis results between two different versions/processing runs
of the same subject by overlaying their data on the same plots.

Usage:
    python src/comparison_analysis.py --subject OSAMRI007 --folder1 OSAMRI007_results --folder2 less1mmesh_OSAMRI007_results
    python src/comparison_analysis.py --subject OSAMRI007 --folder1 OSAMRI007_results --folder2 OSAMRI007v2_results --output comparison_v1_v2.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys

# Import shared utilities
from utils.signal_processing import find_zero_crossings


def load_comparison_data(subject_name: str, folder_path: Path, folder_label: str) -> Dict:
    """
    Load CFD data from a results folder for comparison.
    
    Args:
        subject_name: Subject name (e.g., 'OSAMRI007')
        folder_path: Path to results folder
        folder_label: Label for this dataset (e.g., 'Original', 'Fine Mesh')
    
    Returns:
        Dictionary with processed data for each location
    """
    print(f"Loading data from {folder_path} ({folder_label})...")
    
    # Look for tracking locations file
    tracking_file = folder_path / f"{subject_name}_tracking_locations.json"
    if not tracking_file.exists():
        # Try in parent directory
        tracking_file = Path(f"{subject_name}_tracking_locations.json")
        if not tracking_file.exists():
            print(f"Warning: No tracking locations file found for {folder_label}")
            return {}
    
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
    
    # Convert to expected format
    if isinstance(tracking_data, dict) and 'locations' in tracking_data:
        tracking_locations = tracking_data['locations']
    else:
        # Handle old format
        tracking_locations = []
        for key, description in tracking_data.items():
            patch_num, face_idx = map(int, key.split(','))
            tracking_locations.append({
                'patch_number': patch_num,
                'face_indices': [face_idx],
                'description': description
            })
    
    # Map descriptions to keys
    location_map = {
        'Posterior border of soft palate': 'posterior_border_of_soft_palate',
        'Back of tongue': 'back_of_tongue',
        'Superior border of epiglottis': 'superior_border_of_epiglottis'
    }
    
    # Load single point data
    tracked_points_dir = folder_path / "tracked_points"
    comparison_data = {}
    
    for location in tracking_locations:
        description = location['description']
        
        if description not in location_map:
            continue
            
        location_key = location_map[description]
        
        # Find CSV file by description string (not patch number)
        description_filename = description.lower().replace(' ', '_')
        data_file = None
        
        # Determine the actual file prefix from folder path
        # Extract subject prefix from folder name (e.g., "23mmeshOSAMRI007" from "23mmeshOSAMRI007_results")
        folder_name = folder_path.name
        if folder_name.endswith('_results'):
            file_prefix = folder_name[:-8]  # Remove "_results"
        else:
            file_prefix = subject_name  # Fallback to base subject name
        
        # Search for file with matching description in filename (exclude debug files)
        pattern = f"{file_prefix}_patch*_face*_{description_filename}.csv"
        for csv_file in tracked_points_dir.glob(pattern):
            # Skip debug files
            if 'debug' in csv_file.name.lower():
                continue
            data_file = csv_file
            break
        
        if data_file and data_file.exists():
            df = pd.read_csv(data_file)
            
            # Process the data
            times = df['Time (s)'].values
            vdotn = df['VdotN'].values
            pressure = df['Total Pressure (Pa)'].values
            
            # Calculate acceleration
            adotn = np.gradient(vdotn, times)
            
            # Apply smoothing - handle small datasets
            data_length = len(times)
            
            if data_length <= 5:
                # Too few points, no smoothing
                vdotn_smooth = vdotn
                pressure_smooth = pressure
                adotn_smooth = adotn
            else:
                # Calculate appropriate window size
                max_window = data_length if data_length % 2 == 1 else data_length - 1
                min_window = 5  # Minimum for polyorder=3
                
                # Try to use 20% of data length, but respect constraints
                desired_window = max(min_window, int(0.2 * data_length))
                window_length = min(desired_window, max_window)
                
                # Ensure odd
                if window_length % 2 == 0:
                    window_length -= 1
                
                try:
                    vdotn_smooth = savgol_filter(vdotn, window_length, 3)
                    pressure_smooth = savgol_filter(pressure, window_length, 3)
                    adotn_smooth = savgol_filter(adotn, window_length, 3)
                except ValueError:
                    # Fallback to no smoothing if still fails
                    vdotn_smooth = vdotn
                    pressure_smooth = pressure
                    adotn_smooth = adotn
            
            # Find zero crossings
            v_crossings = find_zero_crossings(times, vdotn_smooth)
            a_crossings = find_zero_crossings(times, adotn_smooth)
            p_crossings = find_zero_crossings(times, pressure_smooth)
            
            # Normalize times (subtract minimum time)
            global_time_min = times.min()
            normalized_times = times - global_time_min
            
            comparison_data[location_key] = {
                'times': times,
                'normalized_times': normalized_times,
                'global_time_min': global_time_min,
                'vdotn': vdotn,
                'pressure': pressure,
                'adotn': adotn,
                'vdotn_smooth': vdotn_smooth,
                'pressure_smooth': pressure_smooth,
                'adotn_smooth': adotn_smooth,
                'v_crossings': v_crossings,
                'a_crossings': a_crossings,
                'p_crossings': p_crossings,
                'folder_label': folder_label,
                'description': description
            }
            
            print(f"Loaded {folder_label} data: {description}")
        else:
            print(f"Warning: No data file found for {description} in {folder_label}")
    
    return comparison_data


def load_patch_comparison_data(subject_name: str, folder_path: Path, folder_label: str, patch_radii: List[float] = None) -> Dict:
    """
    Load patch mean data from a results folder for comparison.
    
    Args:
        subject_name: Subject name (e.g., 'OSAMRI007')
        folder_path: Path to results folder
        folder_label: Label for this dataset (e.g., 'Original', 'Fine Mesh')
        patch_radii: List of patch radii in meters (default: [0.001, 0.002, 0.005])
    
    Returns:
        Dictionary with processed patch data for each location and radius
    """
    if patch_radii is None:
        patch_radii = [0.001, 0.002, 0.005]
    
    print(f"Loading patch data from {folder_path} ({folder_label})...")
    
    # Look for tracking locations file
    tracking_file = folder_path / f"{subject_name}_tracking_locations.json"
    if not tracking_file.exists():
        # Try in parent directory
        tracking_file = Path(f"{subject_name}_tracking_locations.json")
        if not tracking_file.exists():
            print(f"Warning: No tracking locations file found for {folder_label}")
            return {}
    
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
    
    # Convert to expected format
    if isinstance(tracking_data, dict) and 'locations' in tracking_data:
        tracking_locations = tracking_data['locations']
    else:
        # Handle old format
        tracking_locations = []
        for key, description in tracking_data.items():
            patch_num, face_idx = map(int, key.split(','))
            tracking_locations.append({
                'patch_number': patch_num,
                'face_indices': [face_idx],
                'description': description
            })
    
    # Map descriptions to keys
    location_map = {
        'Posterior border of soft palate': 'posterior_border_of_soft_palate',
        'Back of tongue': 'back_of_tongue',
        'Superior border of epiglottis': 'superior_border_of_epiglottis'
    }
    
    # Load patch data
    tracked_points_dir = folder_path / "tracked_points"
    patch_comparison_data = {}
    
    for location in tracking_locations:
        description = location['description']
        
        if description not in location_map:
            continue
            
        location_key = location_map[description]
        patch_comparison_data[location_key] = {}
        
        for radius in patch_radii:
            radius_mm = radius * 1000
            radius_key = f"{radius_mm:.1f}mm"
            
            # Find patch CSV file by description string
            description_filename = description.lower().replace(' ', '_')
            
            # Determine the actual file prefix from folder path
            folder_name = folder_path.name
            if folder_name.endswith('_results'):
                file_prefix = folder_name[:-8]  # Remove "_results"
            else:
                file_prefix = subject_name  # Fallback to base subject name
            
            # Try different patch filename patterns
            patterns = [
                f"{file_prefix}_patch*_face*_{description_filename}_(fixed_patch_{radius_mm:.1f}mm)_r2mm.csv",
                f"{file_prefix}_patch*_face*_{description_filename}_fixed_patch_{radius_mm:.1f}mm.csv",
                f"{file_prefix}_patch*_face*_{description_filename}_patch_{radius_mm:.1f}mm.csv"
            ]
            
            patch_file = None
            for pattern in patterns:
                matches = list(tracked_points_dir.glob(pattern))
                # Filter out debug files
                non_debug_matches = [f for f in matches if 'debug' not in f.name.lower()]
                if non_debug_matches:
                    patch_file = non_debug_matches[0]
                    break
            
            if patch_file and patch_file.exists():
                df = pd.read_csv(patch_file)
                
                # Process the patch data (same as single point processing)
                times = df['Time (s)'].values
                vdotn = df['VdotN'].values
                pressure = df['Total Pressure (Pa)'].values
                
                # Calculate acceleration
                adotn = np.gradient(vdotn, times)
                
                # Apply smoothing - handle small datasets
                data_length = len(times)
                
                if data_length <= 5:
                    # Too few points, no smoothing
                    vdotn_smooth = vdotn
                    pressure_smooth = pressure
                    adotn_smooth = adotn
                else:
                    # Calculate appropriate window size
                    max_window = data_length if data_length % 2 == 1 else data_length - 1
                    min_window = 5  # Minimum for polyorder=3
                    
                    # Try to use 20% of data length, but respect constraints
                    desired_window = max(min_window, int(0.2 * data_length))
                    window_length = min(desired_window, max_window)
                    
                    # Ensure odd
                    if window_length % 2 == 0:
                        window_length -= 1
                    
                    try:
                        vdotn_smooth = savgol_filter(vdotn, window_length, 3)
                        pressure_smooth = savgol_filter(pressure, window_length, 3)
                        adotn_smooth = savgol_filter(adotn, window_length, 3)
                    except ValueError:
                        # Fallback to no smoothing if still fails
                        vdotn_smooth = vdotn
                        pressure_smooth = pressure
                        adotn_smooth = adotn
                
                # Find zero crossings
                v_crossings = find_zero_crossings(times, vdotn_smooth)
                a_crossings = find_zero_crossings(times, adotn_smooth)
                p_crossings = find_zero_crossings(times, pressure_smooth)
                
                # Normalize times (subtract minimum time)
                global_time_min = times.min()
                normalized_times = times - global_time_min
                
                patch_comparison_data[location_key][radius_key] = {
                    'times': times,
                    'normalized_times': normalized_times,
                    'global_time_min': global_time_min,
                    'vdotn': vdotn,
                    'pressure': pressure,
                    'adotn': adotn,
                    'vdotn_smooth': vdotn_smooth,
                    'pressure_smooth': pressure_smooth,
                    'adotn_smooth': adotn_smooth,
                    'v_crossings': v_crossings,
                    'a_crossings': a_crossings,
                    'p_crossings': p_crossings,
                    'folder_label': folder_label,
                    'description': description,
                    'radius': radius_key
                }
                
                print(f"Loaded {folder_label} patch data: {description} ({radius_key})")
            else:
                print(f"Warning: No patch data file found for {description} ({radius_key}) in {folder_label}")
    
    return patch_comparison_data


def create_comparison_3x3_page(data1: Dict, data2: Dict, subject_name: str, pdf, with_markers: bool = False):
    """
    Create a single 3x3 comparison page overlaying two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        subject_name: Subject name
        pdf: PDF object to save to
        with_markers: Whether to include zero-crossing markers
    """
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define the locations
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find mutual ranges across both datasets
    all_data = list(data1.values()) + list(data2.values())
    v_max = max([np.abs(d['vdotn_smooth']).max() * 1000 for d in all_data]) * 1.05  # Convert to mm/s
    a_max = max([np.abs(d['adotn_smooth']).max() * 1000 for d in all_data]) * 1.05  # Convert to mm/s²
    p_max = max([np.abs(d['pressure_smooth']).max() for d in all_data]) * 1.05
    
    # Common settings
    LABEL_SIZE = 14
    TITLE_SIZE = 17
    
    # Create colormap for time progression
    normalized_time_max = max([d['normalized_times'].max() for d in all_data])
    global_time_min = min([d['global_time_min'] for d in all_data])
    original_inhale_exhale = 1.034
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    norm = plt.Normalize(0, normalized_time_max)
    transition_norm = normalized_inhale_exhale / normalized_time_max
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
    
    # Get folder labels for legend
    folder1_label = list(data1.values())[0]['folder_label'] if data1 else "Dataset 1"
    folder2_label = list(data2.values())[0]['folder_label'] if data2 else "Dataset 2"
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        loc_key = loc['key']
        
        # Check if data exists for this location in both datasets
        has_data1 = loc_key in data1
        has_data2 = loc_key in data2
        
        if not (has_data1 or has_data2):
            continue
        
        # 1. Plot p vs v (smoothed) - Row i, Column 0
        ax1 = fig.add_subplot(gs[i, 0])
        
        if has_data1:
            d1 = data1[loc_key]
            vdotn_smooth_mm1 = d1['vdotn_smooth'] * 1000
            pressure_smooth1 = d1['pressure_smooth']
            scatter1 = ax1.scatter(vdotn_smooth_mm1, pressure_smooth1, 
                                 c=d1['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder1_label, marker='o')
        
        if has_data2:
            d2 = data2[loc_key]
            vdotn_smooth_mm2 = d2['vdotn_smooth'] * 1000
            pressure_smooth2 = d2['pressure_smooth']
            scatter2 = ax1.scatter(vdotn_smooth_mm2, pressure_smooth2, 
                                 c=d2['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder2_label, marker='s')
        
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlim(-v_max, v_max)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if i == 0:  # Add legend only to first row
            ax1.legend(loc='upper right', fontsize=10)
        
        # 2. Plot p vs a (smoothed) - Row i, Column 1
        ax2 = fig.add_subplot(gs[i, 1])
        
        if has_data1:
            d1 = data1[loc_key]
            adotn_smooth_mm1 = d1['adotn_smooth'] * 1000
            pressure_smooth1 = d1['pressure_smooth']
            scatter1 = ax2.scatter(adotn_smooth_mm1, pressure_smooth1, 
                                 c=d1['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder1_label, marker='o')
        
        if has_data2:
            d2 = data2[loc_key]
            adotn_smooth_mm2 = d2['adotn_smooth'] * 1000
            pressure_smooth2 = d2['pressure_smooth']
            scatter2 = ax2.scatter(adotn_smooth_mm2, pressure_smooth2, 
                                 c=d2['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder2_label, marker='s')
        
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlim(-a_max, a_max)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if i == 0:  # Add legend only to first row
            ax2.legend(loc='upper right', fontsize=10)
        
        # 3. Plot v vs a (smoothed) - Row i, Column 2
        ax3 = fig.add_subplot(gs[i, 2])
        
        if has_data1:
            d1 = data1[loc_key]
            vdotn_smooth_mm1 = d1['vdotn_smooth'] * 1000
            adotn_smooth_mm1 = d1['adotn_smooth'] * 1000
            scatter1 = ax3.scatter(adotn_smooth_mm1, vdotn_smooth_mm1, 
                                 c=d1['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder1_label, marker='o')
        
        if has_data2:
            d2 = data2[loc_key]
            vdotn_smooth_mm2 = d2['vdotn_smooth'] * 1000
            adotn_smooth_mm2 = d2['adotn_smooth'] * 1000
            scatter2 = ax3.scatter(adotn_smooth_mm2, vdotn_smooth_mm2, 
                                 c=d2['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder2_label, marker='s')
        
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlim(-a_max, a_max)
        ax3.set_ylim(-v_max, v_max)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        if i == 0:  # Add legend only to first row
            ax3.legend(loc='upper right', fontsize=10)
    
    # Add overall title
    fig.suptitle(f'{subject_name} - CFD Analysis Comparison\n{folder1_label} vs {folder2_label}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Add colorbar
    # Create a dummy scatter plot for colorbar reference
    dummy_scatter = plt.scatter([], [], c=[], cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(dummy_scatter, ax=fig.get_axes(), shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Normalized Time (s)', fontsize=12, fontweight='bold')
    
    # Save the page
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)


def create_patch_comparison_3x3_page(data1: Dict, data2: Dict, subject_name: str, radius_key: str, pdf):
    """
    Create a single 3x3 comparison page for patch mean data overlaying two datasets.
    
    Args:
        data1: First dataset (patch data for specific radius)
        data2: Second dataset (patch data for specific radius)
        subject_name: Subject name
        radius_key: Radius key (e.g., '1.0mm')
        pdf: PDF object to save to
    """
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define the locations
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find mutual ranges across both datasets
    all_data = list(data1.values()) + list(data2.values())
    if not all_data:
        return
        
    v_max = max([np.abs(d['vdotn_smooth']).max() * 1000 for d in all_data]) * 1.05  # Convert to mm/s
    a_max = max([np.abs(d['adotn_smooth']).max() * 1000 for d in all_data]) * 1.05  # Convert to mm/s²
    p_max = max([np.abs(d['pressure_smooth']).max() for d in all_data]) * 1.05
    
    # Common settings
    LABEL_SIZE = 14
    TITLE_SIZE = 17
    
    # Create colormap for time progression
    normalized_time_max = max([d['normalized_times'].max() for d in all_data])
    global_time_min = min([d['global_time_min'] for d in all_data])
    original_inhale_exhale = 1.034
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    norm = plt.Normalize(0, normalized_time_max)
    transition_norm = normalized_inhale_exhale / normalized_time_max
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
    
    # Get folder labels for legend
    folder1_label = list(data1.values())[0]['folder_label'] if data1 else "Dataset 1"
    folder2_label = list(data2.values())[0]['folder_label'] if data2 else "Dataset 2"
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        loc_key = loc['key']
        
        # Check if data exists for this location in both datasets
        has_data1 = loc_key in data1
        has_data2 = loc_key in data2
        
        if not (has_data1 or has_data2):
            continue
        
        # 1. Plot p vs v (smoothed) - Row i, Column 0
        ax1 = fig.add_subplot(gs[i, 0])
        
        if has_data1:
            d1 = data1[loc_key]
            vdotn_smooth_mm1 = d1['vdotn_smooth'] * 1000
            pressure_smooth1 = d1['pressure_smooth']
            scatter1 = ax1.scatter(vdotn_smooth_mm1, pressure_smooth1, 
                                 c=d1['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder1_label, marker='o')
        
        if has_data2:
            d2 = data2[loc_key]
            vdotn_smooth_mm2 = d2['vdotn_smooth'] * 1000
            pressure_smooth2 = d2['pressure_smooth']
            scatter2 = ax1.scatter(vdotn_smooth_mm2, pressure_smooth2, 
                                 c=d2['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder2_label, marker='s')
        
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlim(-v_max, v_max)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Total Pressure vs v⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if i == 0:  # Add legend only to first row
            ax1.legend(loc='upper right', fontsize=10)
        
        # 2. Plot p vs a (smoothed) - Row i, Column 1
        ax2 = fig.add_subplot(gs[i, 1])
        
        if has_data1:
            d1 = data1[loc_key]
            adotn_smooth_mm1 = d1['adotn_smooth'] * 1000
            pressure_smooth1 = d1['pressure_smooth']
            scatter1 = ax2.scatter(adotn_smooth_mm1, pressure_smooth1, 
                                 c=d1['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder1_label, marker='o')
        
        if has_data2:
            d2 = data2[loc_key]
            adotn_smooth_mm2 = d2['adotn_smooth'] * 1000
            pressure_smooth2 = d2['pressure_smooth']
            scatter2 = ax2.scatter(adotn_smooth_mm2, pressure_smooth2, 
                                 c=d2['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder2_label, marker='s')
        
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlim(-a_max, a_max)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Total Pressure vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if i == 0:  # Add legend only to first row
            ax2.legend(loc='upper right', fontsize=10)
        
        # 3. Plot v vs a (smoothed) - Row i, Column 2
        ax3 = fig.add_subplot(gs[i, 2])
        
        if has_data1:
            d1 = data1[loc_key]
            vdotn_smooth_mm1 = d1['vdotn_smooth'] * 1000
            adotn_smooth_mm1 = d1['adotn_smooth'] * 1000
            scatter1 = ax3.scatter(adotn_smooth_mm1, vdotn_smooth_mm1, 
                                 c=d1['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder1_label, marker='o')
        
        if has_data2:
            d2 = data2[loc_key]
            vdotn_smooth_mm2 = d2['vdotn_smooth'] * 1000
            adotn_smooth_mm2 = d2['adotn_smooth'] * 1000
            scatter2 = ax3.scatter(adotn_smooth_mm2, vdotn_smooth_mm2, 
                                 c=d2['normalized_times'], cmap=custom_cmap, norm=norm, 
                                 s=20, alpha=0.7, label=folder2_label, marker='s')
        
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlim(-a_max, a_max)
        ax3.set_ylim(-v_max, v_max)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'v⃗·n⃗ vs a⃗·n⃗\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        if i == 0:  # Add legend only to first row
            ax3.legend(loc='upper right', fontsize=10)
    
    # Add overall title
    fig.suptitle(f'{subject_name} - CFD Analysis Comparison (Patch Mean {radius_key})\n{folder1_label} vs {folder2_label}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Add colorbar
    # Create a dummy scatter plot for colorbar reference
    dummy_scatter = plt.scatter([], [], c=[], cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(dummy_scatter, ax=fig.get_axes(), shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Normalized Time (s)', fontsize=12, fontweight='bold')
    
    # Save the page
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)


def create_comparison_report(subject_name: str, folder1_path: Path, folder2_path: Path, 
                           folder1_label: str, folder2_label: str, output_file: str = None,
                           include_patches: bool = True, patch_radii: List[float] = None):
    """
    Create a comparison report between two CFD analysis results.
    
    Args:
        subject_name: Subject name
        folder1_path: Path to first results folder
        folder2_path: Path to second results folder
        folder1_label: Label for first dataset
        folder2_label: Label for second dataset
        output_file: Output PDF filename (optional)
        include_patches: Whether to include patch mean comparisons
        patch_radii: List of patch radii in meters (default: [0.001, 0.002, 0.005])
    """
    if output_file is None:
        output_file = f"{subject_name}_cfd_comparison_{folder1_label}_{folder2_label}.pdf"
    
    if patch_radii is None:
        patch_radii = [0.001, 0.002, 0.005]
    
    print(f"\nCreating CFD Analysis Comparison Report...")
    print(f"Subject: {subject_name}")
    print(f"Dataset 1: {folder1_path} ({folder1_label})")
    print(f"Dataset 2: {folder2_path} ({folder2_label})")
    print(f"Output: {output_file}")
    print(f"Include patches: {include_patches}")
    
    # Load single point data from both folders
    data1 = load_comparison_data(subject_name, folder1_path, folder1_label)
    data2 = load_comparison_data(subject_name, folder2_path, folder2_label)
    
    # Load patch data if requested
    patch_data1 = {}
    patch_data2 = {}
    if include_patches:
        patch_data1 = load_patch_comparison_data(subject_name, folder1_path, folder1_label, patch_radii)
        patch_data2 = load_patch_comparison_data(subject_name, folder2_path, folder2_label, patch_radii)
    
    if not data1 and not data2 and not patch_data1 and not patch_data2:
        print("Error: No data found in either folder!")
        return
    
    if not data1:
        print(f"Warning: No single point data found in {folder1_path}")
    if not data2:
        print(f"Warning: No single point data found in {folder2_path}")
    
    # Create PDF report
    with PdfPages(output_file) as pdf:
        # Page 1: Single point comparison
        if data1 or data2:
            print("Creating single point comparison plots...")
            create_comparison_3x3_page(data1, data2, subject_name, pdf, with_markers=False)
        
        # Additional pages: Patch mean comparisons for each radius
        if include_patches and (patch_data1 or patch_data2):
            # Get all available radii
            all_radii = set()
            if patch_data1:
                for location_data in patch_data1.values():
                    all_radii.update(location_data.keys())
            if patch_data2:
                for location_data in patch_data2.values():
                    all_radii.update(location_data.keys())
            
            # Sort radii for consistent ordering
            sorted_radii = sorted(all_radii, key=lambda x: float(x.replace('mm', '')))
            
            for radius_key in sorted_radii:
                print(f"Creating patch mean comparison plots for {radius_key}...")
                
                # Extract data for this radius
                radius_data1 = {}
                radius_data2 = {}
                
                for location_key in ['posterior_border_of_soft_palate', 'back_of_tongue', 'superior_border_of_epiglottis']:
                    if patch_data1.get(location_key, {}).get(radius_key):
                        radius_data1[location_key] = patch_data1[location_key][radius_key]
                    if patch_data2.get(location_key, {}).get(radius_key):
                        radius_data2[location_key] = patch_data2[location_key][radius_key]
                
                if radius_data1 or radius_data2:
                    # Create a modified version of the comparison page for patch data
                    create_patch_comparison_3x3_page(radius_data1, radius_data2, subject_name, radius_key, pdf)
    
    print(f"✅ Comparison report saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare CFD analysis results between two versions/processing runs.')
    parser.add_argument('--subject', type=str, required=True,
                      help='Subject name (e.g., OSAMRI007)')
    parser.add_argument('--folder1', type=str, required=True,
                      help='First results folder path')
    parser.add_argument('--folder2', type=str, required=True,
                      help='Second results folder path')
    parser.add_argument('--label1', type=str, default=None,
                      help='Label for first dataset (default: folder name)')
    parser.add_argument('--label2', type=str, default=None,
                      help='Label for second dataset (default: folder name)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output PDF filename (default: auto-generated)')
    parser.add_argument('--no-patches', action='store_true',
                      help='Skip patch mean comparisons (only single point)')
    parser.add_argument('--patch-radii', nargs='+', type=float, default=[1.0, 2.0, 5.0],
                      help='Patch radii in millimeters (default: 1.0 2.0 5.0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    folder1_path = Path(args.folder1)
    folder2_path = Path(args.folder2)
    
    if not folder1_path.exists():
        print(f"Error: Folder 1 does not exist: {folder1_path}")
        sys.exit(1)
    
    if not folder2_path.exists():
        print(f"Error: Folder 2 does not exist: {folder2_path}")
        sys.exit(1)
    
    # Set default labels
    label1 = args.label1 if args.label1 else folder1_path.name
    label2 = args.label2 if args.label2 else folder2_path.name
    
    # Convert patch radii from mm to meters
    patch_radii_m = [r / 1000.0 for r in args.patch_radii]
    
    # Create comparison report
    create_comparison_report(
        subject_name=args.subject,
        folder1_path=folder1_path,
        folder2_path=folder2_path,
        folder1_label=label1,
        folder2_label=label2,
        output_file=args.output,
        include_patches=not args.no_patches,
        patch_radii=patch_radii_m
    )


if __name__ == '__main__':
    main() 