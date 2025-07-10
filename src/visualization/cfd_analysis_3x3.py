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
import matplotlib.colors as mcolors


def smart_label_position(ax, target_xy, text, existing_labels, data_points=None, 
                        margin_factor=0.08, min_distance=0.15):
    """
    Zone-based label positioning that guarantees no overlaps by using the entire plot area.
    
    Args:
        ax: matplotlib axis object
        target_xy: (x, y) tuple of the target point to annotate
        text: the text to display
        existing_labels: list of existing label positions [(x, y), ...]
        data_points: optional array of data points to avoid (less important now)
        margin_factor: fraction of plot range to use as margin from edges
        min_distance: minimum distance between labels (fraction of plot range)
    
    Returns:
        (xytext, ha, va): position and alignment for the label
    """
    x_target, y_target = target_xy
    
    # Get plot boundaries
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # Calculate margins and minimum distance
    x_margin = margin_factor * x_range
    y_margin = margin_factor * y_range
    min_dist = min_distance * min(x_range, y_range)
    
    # Estimate text bounding box for the larger font size
    font_size = 11  # Increased by 20% from 9 to ~11
    char_width = font_size * 0.6
    char_height = font_size * 1.2
    text_width = len(text) * char_width
    text_height = char_height
    
    # Convert text dimensions to data coordinates
    text_width_data = text_width * x_range / 800
    text_height_data = text_height * y_range / 600
    
    # Define strategic zones across the entire plot area
    # Use 6x4 grid of zones to distribute labels evenly
    zones = []
    n_cols = 6
    n_rows = 4
    
    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate zone center
            zone_x = xlim[0] + x_margin + (col + 0.5) * (x_range - 2*x_margin) / n_cols
            zone_y = ylim[0] + y_margin + (row + 0.5) * (y_range - 2*y_margin) / n_rows
            
            # Determine alignment based on zone position
            if col < n_cols // 3:
                ha = 'left'
            elif col >= 2 * n_cols // 3:
                ha = 'right'
            else:
                ha = 'center'
                
            if row < n_rows // 2:
                va = 'bottom'
            else:
                va = 'top'
            
            # Calculate priority based on distance to target (closer zones preferred)
            distance_to_target = np.sqrt((zone_x - x_target)**2 + (zone_y - y_target)**2)
            
            zones.append({
                'x': zone_x,
                'y': zone_y,
                'ha': ha,
                'va': va,
                'priority': distance_to_target
            })
    
    # Sort zones by priority (closer to target first, but we'll use any available zone)
    zones.sort(key=lambda z: z['priority'])
    
    def is_zone_available(zone):
        x, y, ha, va = zone['x'], zone['y'], zone['ha'], zone['va']
        
        # Calculate actual text bounds
        if ha == 'left':
            text_x_min, text_x_max = x, x + text_width_data
        elif ha == 'right':
            text_x_min, text_x_max = x - text_width_data, x
        else:  # center
            text_x_min, text_x_max = x - text_width_data/2, x + text_width_data/2
            
        if va == 'bottom':
            text_y_min, text_y_max = y, y + text_height_data
        elif va == 'top':
            text_y_min, text_y_max = y - text_height_data, y
        else:  # center
            text_y_min, text_y_max = y - text_height_data/2, y + text_height_data/2
        
        # Check boundaries
        if (text_x_min < xlim[0] + x_margin or text_x_max > xlim[1] - x_margin or 
            text_y_min < ylim[0] + y_margin or text_y_max > ylim[1] - y_margin):
            return False
        
        # Check distance from existing labels - this is the critical part for no overlaps
        for existing_x, existing_y in existing_labels:
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < min_dist:
                return False
        
        return True
    
    # Try zones in order until we find an available one
    for zone in zones:
        if is_zone_available(zone):
            return (zone['x'], zone['y']), zone['ha'], zone['va']
    
    # If all zones are somehow occupied (very unlikely with 24 zones), 
    # use a fallback position at the edge
    fallback_x = xlim[1] - x_margin - text_width_data/2
    fallback_y = ylim[1] - y_margin - text_height_data/2
    return (fallback_x, fallback_y), 'right', 'top'


def format_time_label(time_value):
    """Format time values to remove unnecessary decimal places."""
    if time_value == int(time_value):
        return f"{int(time_value)}s"
    else:
        # Remove trailing zeros and unnecessary decimal places
        return f"{time_value:.2f}s".rstrip('0').rstrip('.')

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
    LABEL_SIZE = 11.2  # Reduced by 20% from 14 to match main.py
    TITLE_SIZE = 17  # Increased by 20% from 14
    
    # Create colormap
    normalized_time_max = max([data['normalized_times'].max() for data in data_dict.values()])
    global_time_min = min([data['global_time_min'] for data in data_dict.values()])
    original_inhale_exhale = 1.034
    normalized_inhale_exhale = original_inhale_exhale - global_time_min
    
    norm = plt.Normalize(0, normalized_time_max)
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
        if loc['key'] not in data_dict:
            continue
            
        data = data_dict[loc['key']]
        
        # Get the smoothed data
        normalized_times = data['normalized_times']
        vdotn_smooth_mm = data['vdotn_smooth'] * 1000  # Convert to mm/s
        pressure_smooth = data['pressure_smooth']
        adotn_smooth_mm = data['adotn_smooth'] * 1000  # Convert to mm/s²
        
        # Initialize label tracking for this subplot
        ax1_labels = []
        ax2_labels = []  
        ax3_labels = []
        
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
                
                # Use smart positioning
                data_points = list(zip(vdotn_smooth_mm, pressure_smooth))
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
            
            # Mark p=0 crossings
            for p_cross in data['p_crossings']:
                idx_at_cross = np.abs(normalized_times - p_cross).argmin()
                v_at_cross = vdotn_smooth_mm[idx_at_cross]
                
                # Use smart positioning
                data_points = list(zip(vdotn_smooth_mm, pressure_smooth))
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
                
                # Use smart positioning
                data_points = list(zip(adotn_smooth_mm, pressure_smooth))
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
            
            # Mark p=0 crossings
            for p_cross in data['p_crossings']:
                idx_at_cross = np.abs(normalized_times - p_cross).argmin()
                a_at_cross = adotn_smooth_mm[idx_at_cross]
                
                # Use smart positioning
                data_points = list(zip(adotn_smooth_mm, pressure_smooth))
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
                
                # Use smart positioning
                data_points = list(zip(adotn_smooth_mm, vdotn_smooth_mm))
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
            
            # Mark a=0 crossings
            for a_cross in data['a_crossings']:
                idx_at_cross = np.abs(normalized_times - a_cross).argmin()
                v_at_cross = vdotn_smooth_mm[idx_at_cross]
                
                # Use smart positioning
                data_points = list(zip(adotn_smooth_mm, vdotn_smooth_mm))
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
    
    # Add Point-Patch Comparison Page
    if dfs and patch_dfs:
        print("Creating Point-Patch Comparison Page")
        single_point_data = process_data_for_page(dfs, is_patch=False)
        create_point_patch_comparison_page(single_point_data, patch_dfs, subject_name, pdf, pdfs_dir)
    
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
    
    # Add Point-Patch Comparison Page
    if dfs and patch_dfs:
        print("Creating Point-Patch Comparison Page")
        single_point_data = process_data_for_page(dfs, is_patch=False)
        create_point_patch_comparison_page(single_point_data, patch_dfs, subject_name, pdf, pdfs_dir)
    
    # Save standalone PDF
    if pdfs_dir:
        print(f"Saved to: {pdfs_dir / f'{subject_name}_cfd_analysis_3x3_with_markers.pdf'}")
    else:
        print(f"Saved to: {subject_name}_cfd_analysis_3x3_with_markers.pdf")


def create_point_patch_comparison_page(single_point_data, patch_dfs, subject_name, pdf, pdfs_dir=None, default_radius="2.0mm"):
    """
    Create a dedicated 3x3 comparison page showing single point data vs patch data overlays.
    
    3 rows (anatomical locations) × 3 columns (p vs v, p vs a, v vs a comparisons)
    
    Args:
        single_point_data: Processed single point data dictionary
        patch_dfs: Patch data dictionary
        subject_name: Subject name
        pdf: PDF object to save to
        pdfs_dir: Directory for standalone PDFs
        default_radius: Which patch radius to use for comparison (default: 2.0mm)
    """
    print(f"\nCreating Point-Patch Comparison Page (3x3 layout)...")
    
    # Create figure with 3x3 layout
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Define the locations
    locations = [
        {'key': 'posterior_border_of_soft_palate', 'description': 'Posterior Border of Soft Palate'},
        {'key': 'back_of_tongue', 'description': 'Back of Tongue'},
        {'key': 'superior_border_of_epiglottis', 'description': 'Superior Border of Epiglottis'}
    ]
    
    # Get patch data for the specified radius
    patch_data_for_comparison = {}
    if patch_dfs:
        patch_data_for_comparison = process_data_for_page(patch_dfs, is_patch=True, radius_key=default_radius)
    
    if not patch_data_for_comparison:
        print(f"Warning: No patch data found for radius {default_radius}, skipping comparison page")
        return
    
    # Find mutual ranges across all locations (using both single point and patch data)
    v_max = 0
    a_max = 0 
    p_max = 0
    
    # Get ranges from single point data
    if single_point_data:
        for data in single_point_data.values():
            v_max = max(v_max, data['v_max'])
            a_max = max(a_max, data['a_max'])
            p_max = max(p_max, data['p_max'])
    
    # Get ranges from patch data
    for data in patch_data_for_comparison.values():
        v_max = max(v_max, data['v_max'])
        a_max = max(a_max, data['a_max'])
        p_max = max(p_max, data['p_max'])
    
    # Add margins
    v_max *= 1.05
    a_max *= 1.05
    p_max *= 1.05
    
    # Common settings
    LABEL_SIZE = 11.2
    TITLE_SIZE = 17
    
    # Create the 3x3 grid of comparison plots
    for i, loc in enumerate(locations):
        location_key = loc['key']
        
        # Get single point data for this location
        sp_data = single_point_data.get(location_key) if single_point_data else None
        patch_data = patch_data_for_comparison.get(location_key)
        
        if not patch_data:
            continue
            
        # Get patch data
        patch_vdotn_mm = patch_data['vdotn_smooth'] * 1000  # Convert to mm/s
        patch_pressure = patch_data['pressure_smooth']
        patch_adotn_mm = patch_data['adotn_smooth'] * 1000  # Convert to mm/s²
        
        # Get single point data if available
        if sp_data:
            sp_vdotn_mm = sp_data['vdotn_smooth'] * 1000
            sp_pressure = sp_data['pressure_smooth']
            sp_adotn_mm = sp_data['adotn_smooth'] * 1000
        
        # 1. Plot p vs v comparison
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot patch data as line (in time order)
        ax1.plot(patch_vdotn_mm, patch_pressure, 
                'c-', linewidth=2, alpha=0.8, label='Patch Average (2mm radius)')
        
        # Plot single point data as line (in time order)
        if sp_data:
            ax1.plot(sp_vdotn_mm, sp_pressure, 
                    'k-', linewidth=2, alpha=0.8, label='Single Point', zorder=5)
        
        ax1.set_xlim(-v_max, v_max)
        ax1.set_ylim(-p_max, p_max)
        ax1.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Pressure vs Velocity\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=LABEL_SIZE*0.8, loc='upper right')
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        
        # 2. Plot p vs a comparison
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot patch data as line (in time order)
        ax2.plot(patch_adotn_mm, patch_pressure, 
                'c-', linewidth=2, alpha=0.8, label='Patch Average (2mm radius)')
        
        # Plot single point data as line (in time order)
        if sp_data:
            ax2.plot(sp_adotn_mm, sp_pressure, 
                    'k-', linewidth=2, alpha=0.8, label='Single Point', zorder=5)
        
        ax2.set_xlim(-a_max, a_max)
        ax2.set_ylim(-p_max, p_max)
        ax2.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('Total Pressure (Pa)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title(f'Pressure vs Acceleration\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=LABEL_SIZE*0.8, loc='upper right')
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontweight('bold')
        
        # 3. Plot v vs a comparison
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot patch data as line (in time order)
        ax3.plot(patch_adotn_mm, patch_vdotn_mm, 
                'c-', linewidth=2, alpha=0.8, label='Patch Average (2mm radius)')
        
        # Plot single point data as line (in time order)
        if sp_data:
            ax3.plot(sp_adotn_mm, sp_vdotn_mm, 
                    'k-', linewidth=2, alpha=0.8, label='Single Point', zorder=5)
        
        ax3.set_xlim(-a_max, a_max)
        ax3.set_ylim(-v_max, v_max)
        ax3.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title(f'Velocity vs Acceleration\n{loc["description"]}', fontsize=TITLE_SIZE, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=LABEL_SIZE*0.8, loc='upper right')
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_SIZE)
        for label in ax3.get_xticklabels() + ax3.get_yticklabels():
            label.set_fontweight('bold')
    
    # Add overall title
    fig.suptitle(f'Point vs Patch Comparison Analysis\nSingle Point (Black Lines) vs Patch Average {default_radius} (Cyan Lines)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add note
    fig.text(0.5, 0.01, f'Comparison between single point measurements (black lines) and patch-averaged data (cyan lines, {default_radius} radius)', 
            fontsize=10, ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    
    print(f"Point-Patch Comparison Page completed.")


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