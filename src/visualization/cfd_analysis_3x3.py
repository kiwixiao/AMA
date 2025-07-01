"""
CFD Analysis 3x3 Panel Visualizations

This module creates multi-page 3x3 panel visualizations for CFD analysis results:
1. cfd_analysis_3x3.pdf - Smoothed lines, symmetric axis, no zero-crossing markers
2. cfd_analysis_3x3_with_markers.pdf - Smoothed lines with zero-crossing markers, symmetric axis

Structure:
- Page 1: Single point data (3x3 layout)
- Page 2+: Patch mean values for each radius (3x3 layout per page)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from pathlib import Path
from typing import Dict, List, Optional
import json


def find_zero_crossings(times, values):
    """Find the times when values cross zero."""
    zero_crossings = np.where(np.diff(np.signbit(values)))[0]
    crossing_times = []
    
    for idx in zero_crossings:
        if idx + 1 < len(times) and idx >= 0:
            t0, t1 = times[idx], times[idx + 1]
            v0, v1 = values[idx], values[idx + 1]
            
            if v1 != v0:
                t_cross = t0 - v0 * (t1 - t0) / (v1 - v0)
                crossing_times.append(t_cross)
    
    return crossing_times


def create_single_3x3_page(data_dict, subject_name, page_title, pdf, with_markers=False):
    """
    Create a single 3x3 page for CFD analysis.
    
    Args:
        data_dict: Dictionary with processed data for each location
        subject_name: Subject name
        page_title: Title for this page
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
    
    # Find mutual ranges across all locations
    v_max = max([data['v_max'] for data in data_dict.values()])
    a_max = max([data['a_max'] for data in data_dict.values()])
    p_max = max([data['p_max'] for data in data_dict.values()])
    
    # Common settings
    LABEL_SIZE = 14  # Increased by 20% from 12
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Create colormap
    normalized_time_max = max([data['normalized_times'].max() for data in data_dict.values()])
    global_time_min = min([data['global_time_min'] for data in data_dict.values()])
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
    
    # Create the 3x3 grid of plots
    for i, loc in enumerate(locations):
        if loc['key'] not in data_dict:
            continue
            
        data = data_dict[loc['key']]
        
        # Get the smoothed data
        normalized_times = data['normalized_times']
        vdotn_smooth_mm = data['vdotn_smooth'] * 1000  # Convert to mm/s
        pressure_smooth = data['pressure_smooth']
        adotn_smooth_mm = data['adotn_smooth'] * 1000  # Convert to mm/s²
        
        # 1. Plot p vs v (smoothed)
        ax1 = fig.add_subplot(gs[i, 0])
        scatter1 = ax1.scatter(vdotn_smooth_mm, pressure_smooth, c=normalized_times, cmap=custom_cmap, norm=norm, s=20, alpha=0.8)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add zero-crossing markers if requested
        if with_markers and 'v_crossings' in data and 'p_crossings' in data:
            # Mark v=0 crossings
            for v_cross in data['v_crossings']:
                idx_at_cross = np.abs(normalized_times - v_cross).argmin()
                p_at_cross = pressure_smooth[idx_at_cross]
                ax1.axvline(x=0, ymin=0.45, ymax=0.55, color='red', linewidth=2)
                ax1.annotate(f"{v_cross:.2f}s", 
                           xy=(0, p_at_cross), 
                           xytext=(0.1*v_max, p_at_cross + 0.1*p_max),
                           arrowprops=dict(arrowstyle="->", color='red'),
                           color='red', fontsize=10)
            
            # Mark p=0 crossings
            for p_cross in data['p_crossings']:
                idx_at_cross = np.abs(normalized_times - p_cross).argmin()
                v_at_cross = vdotn_smooth_mm[idx_at_cross]
                ax1.axhline(y=0, xmin=0.45, xmax=0.55, color='blue', linewidth=2)
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
        
        # 2. Plot p vs a (smoothed)
        ax2 = fig.add_subplot(gs[i, 1])
        scatter2 = ax2.scatter(adotn_smooth_mm, pressure_smooth, c=normalized_times, cmap=custom_cmap, norm=norm, s=20, alpha=0.8)
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add zero-crossing markers if requested
        if with_markers and 'a_crossings' in data and 'p_crossings' in data:
            # Mark a=0 crossings
            for a_cross in data['a_crossings']:
                idx_at_cross = np.abs(normalized_times - a_cross).argmin()
                p_at_cross = pressure_smooth[idx_at_cross]
                ax2.axvline(x=0, ymin=0.45, ymax=0.55, color='red', linewidth=2)
                ax2.annotate(f"{a_cross:.2f}s", 
                           xy=(0, p_at_cross), 
                           xytext=(0.1*a_max, p_at_cross + 0.1*p_max),
                           arrowprops=dict(arrowstyle="->", color='red'),
                           color='red', fontsize=10)
            
            # Mark p=0 crossings
            for p_cross in data['p_crossings']:
                idx_at_cross = np.abs(normalized_times - p_cross).argmin()
                a_at_cross = adotn_smooth_mm[idx_at_cross]
                ax2.axhline(y=0, xmin=0.45, xmax=0.55, color='blue', linewidth=2)
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
        
        # 3. Plot v vs a (smoothed)
        ax3 = fig.add_subplot(gs[i, 2])
        scatter3 = ax3.scatter(adotn_smooth_mm, vdotn_smooth_mm, c=normalized_times, cmap=custom_cmap, norm=norm, s=20, alpha=0.8)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Add zero-crossing markers if requested
        if with_markers and 'v_crossings' in data and 'a_crossings' in data:
            # Mark v=0 crossings
            for v_cross in data['v_crossings']:
                idx_at_cross = np.abs(normalized_times - v_cross).argmin()
                a_at_cross = adotn_smooth_mm[idx_at_cross]
                ax3.axhline(y=0, xmin=0.45, xmax=0.55, color='green', linewidth=2)
                ax3.annotate(f"{v_cross:.2f}s", 
                           xy=(a_at_cross, 0), 
                           xytext=(a_at_cross + 0.1*a_max, 0.1*v_max),
                           arrowprops=dict(arrowstyle="->", color='green'),
                           color='green', fontsize=10)
            
            # Mark a=0 crossings
            for a_cross in data['a_crossings']:
                idx_at_cross = np.abs(normalized_times - a_cross).argmin()
                v_at_cross = vdotn_smooth_mm[idx_at_cross]
                ax3.axvline(x=0, ymin=0.45, ymax=0.55, color='purple', linewidth=2)
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
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Time (s)', fontsize=LABEL_SIZE * 1.1, fontweight='bold')
    
    # Add overall title
    marker_text = " with Zero-Crossing Markers" if with_markers else ""
    fig.suptitle(f'CFD Analysis: {page_title}\nSmoothed, Symmetric Plots{marker_text}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add note
    marker_note = " Zero-crossing markers show v=0 (red/green), a=0 (purple), p=0 (blue)." if with_markers else ""
    fig.text(0.5, 0.01, f'Time normalized to start at 0s. Inhale-exhale transition at {normalized_inhale_exhale:.2f}s.{marker_note}', 
            fontsize=10, ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def process_data_for_page(dfs_dict, is_patch=False, radius_key=None):
    """
    Process data for a single page (either single points or patch data).
    
    Returns:
        Dictionary with processed data for each location
    """
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Find global time minimum
    global_time_min = float('inf')
    for loc in locations:
        if loc['key'] in dfs_dict:
            if is_patch and radius_key:
                if radius_key in dfs_dict[loc['key']]:
                    df = dfs_dict[loc['key']][radius_key]
                    times = df['Time (s)'].values
                    global_time_min = min(global_time_min, times.min())
            else:
                df = dfs_dict[loc['key']]
                times = df['Time (s)'].values
                global_time_min = min(global_time_min, times.min())
    
    # Process each location
    processed_data = {}
    v_max_global = 0
    a_max_global = 0
    p_max_global = 0
    
    for loc in locations:
        if loc['key'] not in dfs_dict:
            continue
            
        # Get the appropriate DataFrame
        if is_patch and radius_key:
            if radius_key not in dfs_dict[loc['key']]:
                continue
            df = dfs_dict[loc['key']][radius_key]
            # Use patch mean columns if available
            vdotn_col = 'VdotN_Mean' if 'VdotN_Mean' in df.columns else 'VdotN'
            pressure_col = 'Pressure_Mean (Pa)' if 'Pressure_Mean (Pa)' in df.columns else 'Total Pressure (Pa)'
        else:
            df = dfs_dict[loc['key']]
            vdotn_col = 'VdotN'
            pressure_col = 'Total Pressure (Pa)'
        
        # Get the data
        times = df['Time (s)'].values
        vdotn = df[vdotn_col].values
        pressure = df[pressure_col].values
        
        # Normalize time to start from 0
        normalized_times = times - global_time_min
        
        # Apply smoothing - handle small datasets
        data_length = len(times)
        
        if data_length <= 5:
            # Too few points, no smoothing
            vdotn_smooth = vdotn.copy()
            pressure_smooth = pressure.copy()
        else:
            # Calculate appropriate window size
            max_window = data_length if data_length % 2 == 1 else data_length - 1
            min_window = 5  # Minimum for polyorder=3
            
            # Try to use 20% of data length, but respect constraints
            desired_window = max(min_window, int(0.2 * data_length))
            window_size = min(desired_window, max_window)
            
            # Ensure odd
            if window_size % 2 == 0:
                window_size -= 1
            
            try:
                vdotn_smooth = savgol_filter(vdotn, window_size, 3)
                pressure_smooth = savgol_filter(pressure, window_size, 3)
            except ValueError as e:
                print(f"Savgol filter failed in CFD 3x3, falling back to moving average: {e}")
                # Fallback to simple moving average
                import pandas as pd
                def simple_smooth(data, window):
                    return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
                
                vdotn_smooth = simple_smooth(vdotn, min(5, data_length))
                pressure_smooth = simple_smooth(pressure, min(5, data_length))
        
        # Calculate acceleration from smoothed velocity
        dt = times[1] - times[0]
        dvdotn = np.diff(vdotn_smooth)
        adotn_smooth = np.append(dvdotn / dt, dvdotn[-1] / dt)
        
        # Find zero crossings for markers
        v_crossings = find_zero_crossings(normalized_times, vdotn_smooth)
        a_crossings = find_zero_crossings(normalized_times, adotn_smooth)
        p_crossings = find_zero_crossings(normalized_times, pressure_smooth)
        
        # Update global max values (convert to mm/s and mm/s²)
        v_max_global = max(v_max_global, np.max(np.abs(vdotn_smooth * 1000)))
        a_max_global = max(a_max_global, np.max(np.abs(adotn_smooth * 1000)))
        p_max_global = max(p_max_global, np.max(np.abs(pressure_smooth)))
        
        # Store processed data
        processed_data[loc['key']] = {
            'normalized_times': normalized_times,
            'vdotn_smooth': vdotn_smooth,
            'pressure_smooth': pressure_smooth,
            'adotn_smooth': adotn_smooth,
            'v_crossings': v_crossings,
            'a_crossings': a_crossings,
            'p_crossings': p_crossings,
            'global_time_min': global_time_min,
            'v_max': v_max_global,
            'a_max': a_max_global,
            'p_max': p_max_global
        }
    
    # Add margins and update all entries
    v_max_global *= 1.05
    a_max_global *= 1.05
    p_max_global *= 1.05
    
    for key in processed_data:
        processed_data[key]['v_max'] = v_max_global
        processed_data[key]['a_max'] = a_max_global
        processed_data[key]['p_max'] = p_max_global
    
    return processed_data


def create_cfd_analysis_3x3_panel(dfs, patch_dfs, subject_name, pdf, pdfs_dir=None):
    """
    Create multi-page CFD analysis 3x3 panels (smoothed, no zero-crossing markers).
    
    Page 1: Single point data
    Page 2+: Patch mean data for each radius
    """
    print(f"\nGenerating CFD Analysis 3x3 Panel (multi-page, smoothed, no markers)...")
    
    # Page 1: Single point data
    if dfs:
        print("Creating Page 1: Single Point Data")
        single_point_data = process_data_for_page(dfs, is_patch=False)
        if single_point_data:
            create_single_3x3_page(single_point_data, subject_name, "Single Point Data", pdf, with_markers=False)
    
    # Page 2+: Patch data for each radius
    if patch_dfs:
        # Get all available radii
        all_radii = set()
        for location_data in patch_dfs.values():
            all_radii.update(location_data.keys())
        
        # Sort radii for consistent ordering
        sorted_radii = sorted(all_radii, key=lambda x: float(x.replace('mm', '')))
        
        for radius_key in sorted_radii:
            print(f"Creating Page: Patch Mean Data ({radius_key})")
            patch_data = process_data_for_page(patch_dfs, is_patch=True, radius_key=radius_key)
            if patch_data:
                page_title = f"Patch Mean Data ({radius_key})"
                create_single_3x3_page(patch_data, subject_name, page_title, pdf, with_markers=False)
    
    # Save standalone PDF
    if pdfs_dir:
        print(f"Saved to: {pdfs_dir / f'{subject_name}_cfd_analysis_3x3.pdf'}")
    else:
        print(f"Saved to: {subject_name}_cfd_analysis_3x3.pdf")


def create_cfd_analysis_3x3_panel_with_markers(dfs, patch_dfs, subject_name, pdf, pdfs_dir=None):
    """
    Create multi-page CFD analysis 3x3 panels (smoothed, with zero-crossing markers).
    
    Page 1: Single point data
    Page 2+: Patch mean data for each radius
    """
    print(f"\nGenerating CFD Analysis 3x3 Panel with Markers (multi-page, smoothed, with zero-crossing markers)...")
    
    # Page 1: Single point data
    if dfs:
        print("Creating Page 1: Single Point Data (with markers)")
        single_point_data = process_data_for_page(dfs, is_patch=False)
        if single_point_data:
            create_single_3x3_page(single_point_data, subject_name, "Single Point Data", pdf, with_markers=True)
    
    # Page 2+: Patch data for each radius
    if patch_dfs:
        # Get all available radii
        all_radii = set()
        for location_data in patch_dfs.values():
            all_radii.update(location_data.keys())
        
        # Sort radii for consistent ordering
        sorted_radii = sorted(all_radii, key=lambda x: float(x.replace('mm', '')))
        
        for radius_key in sorted_radii:
            print(f"Creating Page: Patch Mean Data ({radius_key}) with markers")
            patch_data = process_data_for_page(patch_dfs, is_patch=True, radius_key=radius_key)
            if patch_data:
                page_title = f"Patch Mean Data ({radius_key})"
                create_single_3x3_page(patch_data, subject_name, page_title, pdf, with_markers=True)
    
    # Save standalone PDF
    if pdfs_dir:
        print(f"Saved to: {pdfs_dir / f'{subject_name}_cfd_analysis_3x3_with_markers.pdf'}")
    else:
        print(f"Saved to: {subject_name}_cfd_analysis_3x3_with_markers.pdf")


def load_cfd_data_for_analysis(subject_name: str, output_dir: Path, enable_patch_analysis: bool = True, patch_radii: List[float] = None) -> tuple:
    """
    Load both single point and patch data for CFD analysis.
    
    Returns:
        tuple: (single_point_dfs, patch_dfs) dictionaries
    """
    print(f"Loading CFD data for analysis...")
    
    # Load tracking locations
    tracking_file = f"{subject_name}_tracking_locations.json"
    if not Path(tracking_file).exists():
        print(f"Warning: No tracking locations file found: {tracking_file}")
        return {}, {}
    
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
    
    # Map descriptions to keys for compatibility
    location_map = {
        'Posterior border of soft palate': 'posterior_border_of_soft_palate',
        'Back of tongue': 'back_of_tongue',
        'Superior border of epiglottis': 'superior_border_of_epiglottis'
    }
    
    # Load single point data
    single_point_dfs = {}
    for location in tracking_locations:
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        
        if description not in location_map:
            continue
            
        # Load single point data
        data_file = output_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}.csv"
        
        if data_file.exists():
            df = pd.read_csv(data_file)
            single_point_dfs[location_map[description]] = df
            print(f"Loaded single point data: {description}")
        else:
            print(f"Warning: No single point data file found for {description}")
    
    # Load patch data if enabled
    patch_dfs = {}
    if enable_patch_analysis:
        if patch_radii is None:
            patch_radii = [0.001, 0.002, 0.005]
        
        for location in tracking_locations:
            patch_number = location['patch_number']
            face_index = location['face_indices'][0]
            description = location['description']
            
            if description not in location_map:
                continue
            
            location_key = location_map[description]
            patch_dfs[location_key] = {}
            
            for radius in patch_radii:
                radius_mm = radius * 1000
                radius_key = f"{radius_mm:.1f}mm"
                
                # Try different filename patterns
                patterns = [
                    f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}_(fixed_patch_{radius_mm:.1f}mm)_r2mm.csv",
                    f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}_fixed_patch_{radius_mm:.1f}mm.csv",
                    f"{subject_name}_patch{patch_number}_face{face_index}_{description.lower().replace(' ', '_')}_patch_{radius_mm:.1f}mm.csv"
                ]
                
                patch_file = None
                for pattern in patterns:
                    potential_file = output_dir / pattern
                    if potential_file.exists():
                        patch_file = potential_file
                        break
                
                if patch_file and patch_file.exists():
                    df = pd.read_csv(patch_file)
                    patch_dfs[location_key][radius_key] = df
                    print(f"Loaded patch data: {description} ({radius_key})")
                else:
                    print(f"Warning: No patch data file found for {description} ({radius_key})")
    
    return single_point_dfs, patch_dfs 