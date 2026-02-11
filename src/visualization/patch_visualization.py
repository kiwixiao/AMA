"""
Interactive 3D patch visualization for airway surface analysis.
This module provides functionality to visualize patch regions on 3D airway surfaces.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import sys

def visualize_patch_regions(subject_name: str, 
                          time_step: int = 100,
                          patch_radii: List[float] = None,
                          use_pipeline_data: bool = True,
                          normal_angle_threshold: float = 60.0,
                          output_dir: str = None,
                          hdf5_file_path: str = None) -> Optional[go.Figure]:
    """
    Create interactive 3D visualization of patch regions on airway surface.
    
    Args:
        subject_name: Subject identifier (e.g., 'OSAMRI007')
        time_step: Time step to visualize (default: 100)
        patch_radii: List of patch radii in meters (default: [0.001, 0.002, 0.005])
        use_pipeline_data: Whether to use pre-filtered data from pipeline (default: True)
        normal_angle_threshold: Normal angle threshold for filtering (default: 60.0)
        output_dir: Directory to save HTML file (default: current directory)
        hdf5_file_path: Path to HDF5 file (if None, tries to find it automatically)
        
    Returns:
        Plotly figure object if successful, None if failed
    """
    if patch_radii is None:
        patch_radii = [0.001, 0.002, 0.005]  # 1mm, 2mm, 5mm
    
    print(f"🎯 Visualizing patch regions for {subject_name} at time step {time_step}")
    
    if use_pipeline_data:
        print("📊 Using pre-filtered data from main pipeline (recommended)")
    else:
        print("🔄 Recomputing filtering (for debugging/validation)")
    
    # Try to load from HDF5 first, then fall back to CSV
    df = None
    data_source = None
    
    # Try HDF5 first
    if hdf5_file_path is None:
        hdf5_file_path = f"{subject_name}_cfd_data.h5"
    
    if Path(hdf5_file_path).exists():
        print(f"📊 Loading from HDF5 file: {hdf5_file_path}")
        try:
            from ..data_processing.trajectory import load_hdf5_data_for_html_plots
        except ImportError:
            # Handle direct execution or different import context
            import sys
            import os
            src_dir = os.path.dirname(os.path.dirname(__file__))
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from data_processing.trajectory import load_hdf5_data_for_html_plots
        
        df = load_hdf5_data_for_html_plots(hdf5_file_path, time_step)
        if df is not None:
            data_source = "HDF5 (from pipeline cache)"
        else:
            print("❌ Failed to load from HDF5, trying CSV fallback...")
    
    # Fall back to CSV if HDF5 failed
    if df is None:
        print(f"📊 Loading from CSV files...")
        # Load base airway surface
        patched_xyz_dir = Path(f'{subject_name}_xyz_tables')

        # Find the file that matches the timestep (handle scientific notation)
        base_file = None
        available_files = list(patched_xyz_dir.glob('*XYZ_Internal_Table_table_*.csv'))
        
        # Import timestep extraction function
        try:
            from ..main import extract_timestep_from_filename
        except ImportError:
            # Handle direct execution or different import context
            import sys
            import os
            src_dir = os.path.dirname(os.path.dirname(__file__))
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from main import extract_timestep_from_filename
        
        # Find file with matching timestep
        for file_path in available_files:
            try:
                file_timestep = extract_timestep_from_filename(file_path)
                if abs(file_timestep - time_step) < 1e-6:  # Allow small floating point differences
                    base_file = file_path
                    break
            except ValueError:
                continue
        
        if base_file is None:
            print(f"❌ Base surface file not found for timestep {time_step}")
            print(f"Available files in {patched_xyz_dir}:")
            for f in sorted(available_files)[:5]:  # Show first 5
                try:
                    ts = extract_timestep_from_filename(f)
                    print(f"   - {f.name} (timestep: {ts})")
                except ValueError:
                    print(f"   - {f.name} (timestep: unknown)")
            return None
        
        print(f"📁 Loading base surface from {base_file}")
        df = pd.read_csv(base_file)
        data_source = "CSV (patched files)"
    
    if df is None:
        print("❌ Failed to load data from any source")
        return None
    
    print(f"✅ Loaded {len(df):,} points from airway surface")
    
    # Load tracking locations
    try:
        from ..utils.file_processing import load_tracking_locations
    except ImportError:
        # Handle direct execution or different import context
        import sys
        import os
        src_dir = os.path.dirname(os.path.dirname(__file__))
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from utils.file_processing import load_tracking_locations
    
    tracking_config = load_tracking_locations(subject_name=subject_name)
    tracking_locations = tracking_config['locations']
    
    print(f"🎯 Found {len(tracking_locations)} tracking locations")
    
    # Create the 3D plot
    fig = go.Figure()
    
    # Add the full airway surface as a dark gray background
    print("🌫️  Adding full airway surface...")
    
    # Group points by patch number to create proper surface visualization
    colors_surface = ['#E8E8E8', '#D0D0D0', '#B8B8B8', '#A0A0A0', '#888888', '#707070', '#585858', '#404040']
    
    # Get unique patch numbers and create surface traces
    unique_patches = sorted(df['Patch Number'].unique())
    print(f"   Found {len(unique_patches)} surface patches")
    
    for i, patch_num in enumerate(unique_patches):
        patch_mask = df['Patch Number'] == patch_num
        patch_data = df[patch_mask]
        
        if len(patch_data) == 0:
            continue
        
        # Use a light gray color for background surface
        surface_color = colors_surface[i % len(colors_surface)]
        
        fig.add_trace(go.Scatter3d(
            x=patch_data['X (m)'],
            y=patch_data['Y (m)'], 
            z=patch_data['Z (m)'],
            mode='markers',
            marker=dict(
                size=1.5,  # Slightly larger for better visibility
                color=surface_color,
                opacity=0.4  # Semi-transparent
            ),
            name=f'Surface Patch {patch_num}',
            showlegend=False,  # Don't clutter legend with surface patches
            hoverinfo='skip'   # Skip hover for background surface
        ))
    
    # Color variations for different radii (light to lighter for patch regions)
    def get_radius_color(base_color: str, radius_idx: int) -> str:
        """Generate distinct colors for different radii of the same location"""
        color_variations = {
            'red': ['#FFB3B3', '#FF9999', '#FF8080'],      # Light to medium light red
            'blue': ['#B3D9FF', '#99CCFF', '#80C0FF'],     # Light to medium light blue  
            'green': ['#B3FFB3', '#99FF99', '#80FF80'],    # Light to medium light green
            'orange': ['#FFD9B3', '#FFCC99', '#FFC080'],   # Light to medium light orange
            'purple': ['#E6B3FF', '#D999FF', '#CC80FF'],   # Light to medium light purple
            'brown': ['#D9B3A3', '#CC9999', '#BF8080'],    # Light to medium light brown
            'pink': ['#FFB3E6', '#FF99D9', '#FF80CC'],     # Light to medium light pink
            'cyan': ['#B3FFFF', '#99FFFF', '#80FFFF']      # Light to medium light cyan
        }
        return color_variations.get(base_color, ['#CCCCCC', '#B3B3B3', '#999999'])[radius_idx]
    
    # Dark colors for center points
    def get_center_color(base_color: str) -> str:
        """Get dark color for center points"""
        dark_colors = {
            'red': '#800000',      # Dark red
            'blue': '#000080',     # Dark blue
            'green': '#006400',    # Dark green
            'orange': '#FF4500',   # Dark orange
            'purple': '#4B0082',   # Dark purple (indigo)
            'brown': '#8B4513',    # Dark brown (saddle brown)
            'pink': '#C71585',     # Dark pink (medium violet red)
            'cyan': '#008B8B'      # Dark cyan
        }
        return dark_colors.get(base_color, '#333333')
    
    # Base colors for each location
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    # Process each tracking location
    for loc_idx, location in enumerate(tracking_locations):
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        base_color = base_colors[loc_idx % len(base_colors)]
        
        print(f"\n🔍 Processing {description} (Patch {patch_number}, Face {face_index})")
        
        # Find center point
        center_mask = (df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)
        center_points = df[center_mask]
        
        if len(center_points) == 0:
            print(f"   ❌ Center point not found")
            continue
        
        center_point = (
            center_points['X (m)'].iloc[0],
            center_points['Y (m)'].iloc[0],
            center_points['Z (m)'].iloc[0]
        )
        
        print(f"   Center: ({center_point[0]:.6f}, {center_point[1]:.6f}, {center_point[2]:.6f})")
        
        # Add the center point as a large marker
        fig.add_trace(go.Scatter3d(
            x=[center_point[0]],
            y=[center_point[1]],
            z=[center_point[2]],
            mode='markers',
            marker=dict(
                size=12,
                color=get_center_color(base_color),  # Use dark color for center
                symbol='diamond',
                line=dict(width=3, color='black'),
                opacity=1.0
            ),
            name=f'{description} (Center)',
            showlegend=True,
            hovertemplate=f'<b>{description}</b><br>' +
                         f'Patch: {patch_number}<br>' +
                         f'Face: {face_index}<br>' +
                         'X: %{x:.6f}<br>' +
                         'Y: %{y:.6f}<br>' +
                         'Z: %{z:.6f}<br>' +
                         '<extra></extra>'
        ))
        
        # Process each radius
        for radius_idx, radius in enumerate(patch_radii):
            radius_mm = radius * 1000
            
            # Always use proper filtering for visualization to avoid disconnected patches
            try:
                from ..main import find_connected_points_with_normal_filter
            except ImportError:
                # Handle direct execution or different import context
                import sys
                import os
                src_dir = os.path.dirname(os.path.dirname(__file__))
                if str(src_dir) not in sys.path:
                    sys.path.insert(0, str(src_dir))
                from main import find_connected_points_with_normal_filter
            
            # Apply proper filtering (same as pipeline) to ensure connected patches
            patch_points = find_connected_points_with_normal_filter(
                df, center_point, radius, 
                connectivity_threshold=0.001,
                normal_angle_threshold=normal_angle_threshold
            )
            
            # Get pipeline count for comparison if available
            pipeline_count = "N/A"
            if use_pipeline_data:
                patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                patch_data_file = Path(f"{subject_name}_results/tracked_points/{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv")
                
                if patch_data_file.exists():
                    patch_df = pd.read_csv(patch_data_file)
                    pipeline_count = len(patch_df)
                
            visualization_count = len(patch_points)
            
            if len(patch_points) == 0:
                print(f"   ❌ {radius_mm:.1f}mm: No points found")
                continue
            
            # Add patch region points
            patch_color = get_radius_color(base_color, radius_idx)
            
            fig.add_trace(go.Scatter3d(
                x=patch_points['X (m)'],
                y=patch_points['Y (m)'],
                z=patch_points['Z (m)'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=patch_color,
                    opacity=0.8
                ),
                name=f'{description} ({radius_mm:.1f}mm)',
                showlegend=True,
                hovertemplate=f'<b>{description} ({radius_mm:.1f}mm)</b><br>' +
                             'X: %{x:.6f}<br>' +
                             'Y: %{y:.6f}<br>' +
                             'Z: %{z:.6f}<br>' +
                             '<extra></extra>'
            ))
            
            if use_pipeline_data and pipeline_count != "N/A":
                print(f"   ✅ {radius_mm:.1f}mm: {visualization_count} points (pipeline: {pipeline_count})")
            else:
                print(f"   ✅ {radius_mm:.1f}mm: {visualization_count} points (filtered)")
    
    # Update layout to reflect data source
    fig.update_layout(
        title=dict(
            text=f'{subject_name} - Patch Regions (t={time_step})<br>' +
                 f'<sub>Data Source: {data_source}</sub>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)', 
            zaxis_title='Z (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data',
            # Clean white background with no grid lines
            xaxis=dict(
                backgroundcolor='white',
                gridcolor='white',
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                backgroundcolor='white',
                gridcolor='white', 
                showgrid=False,
                zeroline=False
            ),
            zaxis=dict(
                backgroundcolor='white',
                gridcolor='white',
                showgrid=False,
                zeroline=False
            ),
            bgcolor='white'  # Scene background color
        ),
        width=1400,
        height=900,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        uirevision='constant',
        paper_bgcolor='white',  # Overall plot background
        plot_bgcolor='white'    # Plot area background
    )
    
    # Save the visualization (round timestep for clean filename)
    time_step_int = int(round(time_step))
    if output_dir:
        output_path = Path(output_dir) / f'{subject_name}_patch_regions_t{time_step_int}ms.html'
    else:
        output_path = f'{subject_name}_patch_regions_t{time_step_int}ms.html'

    fig.write_html(output_path)
    print(f"\n✅ Patch regions visualization saved as: {output_path}")

    # Also save to results directory if it exists
    results_interactive_dir = Path(f'{subject_name}_results/interactive')
    if results_interactive_dir.exists():
        results_output_path = results_interactive_dir / f'{subject_name}_patch_regions_t{time_step_int}ms.html'
        fig.write_html(results_output_path)
        print(f"📁 Also saved to: {results_output_path}")
    
    # Print summary
    patch_radii_mm = [f'{r*1000:.1f}mm' for r in patch_radii]
    print(f"\n📊 Visualization Summary:")
    print(f"   • Subject: {subject_name}")
    print(f"   • Time step: {time_step} ({time_step/1000:.3f}s)")
    print(f"   • Tracking locations: {len(tracking_locations)}")
    print(f"   • Patch radii: {patch_radii_mm}")
    print(f"   • Data source: {data_source}")
    print(f"   • Total airway surface points: {len(df):,}")
    
    return fig


def main():
    """Command line interface for patch visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize patch regions on 3D airway surface')
    parser.add_argument('--subject', default='OSAMRI007', help='Subject name (default: OSAMRI007)')
    parser.add_argument('--timestep', type=int, default=100, help='Time step to visualize (default: 100)')
    parser.add_argument('--patchradii', nargs='+', type=float, default=[1.0, 2.0, 5.0],
                       help='Patch radii in millimeters (default: 1.0 2.0 5.0)')
    parser.add_argument('--recompute', action='store_true',
                       help='Recompute filtering instead of using pipeline data')
    parser.add_argument('--normalangle', type=float, default=60.0,
                       help='Normal angle threshold in degrees (default: 60.0)')
    
    args = parser.parse_args()
    
    # Convert patch radii from mm to meters
    patch_radii = [r / 1000.0 for r in args.patchradii]
    
    # Run visualization
    fig = visualize_patch_regions(
        subject_name=args.subject,
        time_step=args.timestep,
        patch_radii=patch_radii,
        use_pipeline_data=not args.recompute,
        normal_angle_threshold=args.normalangle
    )
    
    if fig:
        print("\n✅ Visualization completed successfully!")
    else:
        print("\n❌ Visualization failed!")
        sys.exit(1)


if __name__ == '__main__':
    main() 