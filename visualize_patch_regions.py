#!/usr/bin/env python3
"""
Patch Regions Visualization

Streamlined visualization tool for viewing patch regions from the CFD analysis pipeline.
Uses pre-filtered patch data from the main pipeline by default for optimal performance.

Usage:
    python visualize_patch_regions.py --subject OSAMRI007 --timestep 100
    python visualize_patch_regions.py --subject OSAMRI007 --timestep 100 --recompute
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
from typing import List, Tuple
import argparse
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def load_tracking_locations(subject_name: str = None):
    """Load tracking locations from subject-specific JSON file."""
    from src.utils.file_processing import load_tracking_locations as load_tracking
    return load_tracking(subject_name=subject_name)

def calculate_surface_normal(points: np.ndarray, center_idx: int, k: int = 10) -> np.ndarray:
    """Calculate surface normal at a point using local neighborhood PCA."""
    if len(points) < 3:
        return np.array([0, 0, 1])
    
    nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
    distances, indices = nbrs.kneighbors([points[center_idx]])
    local_points = points[indices[0]]
    centered = local_points - local_points.mean(axis=0)
    
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        return normal / np.linalg.norm(normal)
    except:
        return np.array([0, 0, 1])

def find_connected_points_with_normal_filter(df: pd.DataFrame, center_point: Tuple[float, float, float], 
                                           radius: float, connectivity_threshold: float = 0.001,
                                           normal_angle_threshold: float = 60.0) -> pd.DataFrame:
    """Find connected points with surface normal filtering (same as pipeline filtering)."""
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
    
    # Stage 1: Initial connectivity check
    coords = candidate_points[['X (m)', 'Y (m)', 'Z (m)']].values
    clustering = DBSCAN(eps=connectivity_threshold, min_samples=1).fit(coords)
    candidate_points['cluster_label'] = clustering.labels_
    
    center_distances = cdist([center_point], coords)[0]
    center_point_idx = np.argmin(center_distances)
    center_cluster = candidate_points.iloc[center_point_idx]['cluster_label']
    connected_points = candidate_points[candidate_points['cluster_label'] == center_cluster].copy()
    
    # Stage 2: Surface normal filtering (for patches >= 1mm)
    if len(connected_points) > 3 and radius >= 0.001:
        coords_connected = connected_points[['X (m)', 'Y (m)', 'Z (m)']].values
        ref_radius = min(0.001, radius)
        ref_distances = np.sqrt(np.sum((coords_connected - np.array(center_point))**2, axis=1))
        ref_mask = ref_distances <= ref_radius
        
        if np.sum(ref_mask) >= 3:
            ref_points = coords_connected[ref_mask]
            ref_center_idx = np.argmin(ref_distances[ref_mask])
            reference_normal = calculate_surface_normal(ref_points, ref_center_idx)
            
            valid_indices = []
            for i, point_coords in enumerate(coords_connected):
                if ref_mask[i]:
                    valid_indices.append(i)
                else:
                    point_normal = calculate_surface_normal(coords_connected, i, k=8)
                    cos_angle = np.dot(reference_normal, point_normal)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))
                    
                    if angle_deg <= normal_angle_threshold:
                        valid_indices.append(i)
            
            if valid_indices:
                connected_points = connected_points.iloc[valid_indices].copy()
                
                # Stage 3: Final connectivity check
                if len(connected_points) > 1:
                    coords_filtered = connected_points[['X (m)', 'Y (m)', 'Z (m)']].values
                    final_clustering = DBSCAN(eps=connectivity_threshold, min_samples=1).fit(coords_filtered)
                    connected_points['final_cluster_label'] = final_clustering.labels_
                    
                    final_center_distances = cdist([center_point], coords_filtered)[0]
                    final_center_point_idx = np.argmin(final_center_distances)
                    final_center_cluster = connected_points.iloc[final_center_point_idx]['final_cluster_label']
                    
                    connected_points = connected_points[connected_points['final_cluster_label'] == final_center_cluster].copy()
                    connected_points = connected_points.drop('final_cluster_label', axis=1)
    
    connected_points['distance_from_center'] = distances[within_radius_mask][connected_points.index]
    connected_points = connected_points.drop('cluster_label', axis=1)
    
    return connected_points

def visualize_patch_regions(subject_name: str = "OSAMRI007", time_step: int = 100,
                           patch_radii: List[float] = [0.001, 0.002, 0.005],
                           use_pipeline_data: bool = True,
                           normal_angle_threshold: float = 60.0):
    """
    Create interactive visualization of patch regions.
    
    Args:
        subject_name: Subject name (e.g., "OSAMRI007")
        time_step: Time step to visualize (default: 100 = 0.1s)
        patch_radii: List of patch radii in meters (default: [1mm, 2mm, 5mm])
        use_pipeline_data: If True, use pre-filtered CSV data from pipeline (default: True)
        normal_angle_threshold: Normal angle threshold for filtering if recomputing (default: 60°)
    """
    
    print(f"🎯 Visualizing patch regions for {subject_name} at time step {time_step}")
    
    if use_pipeline_data:
        print("📊 Using pre-filtered data from main pipeline (recommended)")
    else:
        print("🔄 Recomputing patch filtering (slower)")
    
    # Load the patched time frame CSV file for the base surface
    xyz_file = Path(f"{subject_name}_xyz_tables_with_patches") / f"patched_XYZ_Internal_Table_table_{time_step}.csv"
    
    if not xyz_file.exists():
        print(f"❌ Error: File {xyz_file} not found")
        return None
    
    print(f"📁 Loading base surface from {xyz_file}")
    df = pd.read_csv(xyz_file, low_memory=False)
    print(f"✅ Loaded {len(df):,} points from patched airway surface")
    
    # Load tracking locations
    tracking_config = load_tracking_locations(subject_name=subject_name)
    tracking_locations = tracking_config['locations']
    print(f"🎯 Found {len(tracking_locations)} tracking locations")
    
    # Create the base surface plot
    fig = go.Figure()
    
    # Add the full airway surface as a dark gray background
    print("🌫️  Adding full airway surface...")
    fig.add_trace(go.Scatter3d(
        x=df['X (m)'],
        y=df['Y (m)'], 
        z=df['Z (m)'],
        mode='markers',
        marker=dict(
            size=1,
            color='#404040',  # Dark gray
            opacity=0.3
        ),
        name='Airway Surface',
        showlegend=True
    ))
    
    # Base colors for different tracking locations
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    # Color variations for different radii (light to lighter for patch regions)
    def get_radius_color(base_color: str, radius_idx: int) -> str:
        """Generate light colors for patch regions (1mm, 2mm, 5mm)"""
        color_variations = {
            'red': ['#FFB3B3', '#FF9999', '#FF8080'],      # Light red shades
            'blue': ['#B3D9FF', '#99CCFF', '#80BFFF'],     # Light blue shades  
            'green': ['#B3FFB3', '#99FF99', '#80FF80'],    # Light green shades
            'orange': ['#FFD9B3', '#FFCC99', '#FFBF80'],   # Light orange shades
            'purple': ['#D9B3FF', '#CC99FF', '#BF80FF'],   # Light purple shades
            'brown': ['#D2B48C', '#C4A484', '#B6947C'],    # Light brown shades
            'pink': ['#FFB3E6', '#FF99DD', '#FF80D4'],     # Light pink shades
            'cyan': ['#B3FFFF', '#99FFFF', '#80FFFF']      # Light cyan shades
        }
        return color_variations.get(base_color, ['#CCCCCC', '#BBBBBB', '#AAAAAA'])[radius_idx]
    
    # Dark colors for center points
    def get_center_color(base_color: str) -> str:
        """Generate dark colors for center points"""
        center_colors = {
            'red': '#800000',      # Dark red
            'blue': '#000080',     # Dark blue
            'green': '#006400',    # Dark green
            'orange': '#FF4500',   # Dark orange
            'purple': '#4B0082',   # Dark purple
            'brown': '#654321',    # Dark brown
            'pink': '#8B008B',     # Dark pink
            'cyan': '#008B8B'      # Dark cyan
        }
        return center_colors.get(base_color, '#333333')
    
    # Process each tracking location
    for idx, location in enumerate(tracking_locations):
        patch_number = location['patch_number']
        face_index = location['face_indices'][0]
        description = location['description']
        base_color = base_colors[idx % len(base_colors)]
        
        print(f"\n🔍 Processing {description} (Patch {patch_number}, Face {face_index})")
        
        # Find the center point
        center_condition = (df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)
        center_points = df[center_condition]
        
        if center_points.empty:
            print(f"⚠️  Warning: No center point found for Patch {patch_number}, Face {face_index}")
            continue
            
        center_point = (
            center_points['X (m)'].iloc[0],
            center_points['Y (m)'].iloc[0], 
            center_points['Z (m)'].iloc[0]
        )
        
        print(f"   Center: ({center_point[0]:.6f}, {center_point[1]:.6f}, {center_point[2]:.6f})")
        
        # Add the center point as a large marker with dark color
        center_color = get_center_color(base_color)
        fig.add_trace(go.Scatter3d(
            x=[center_point[0]],
            y=[center_point[1]],
            z=[center_point[2]],
            mode='markers',
            marker=dict(
                size=12,
                color=center_color,  # Use dark color for center
                symbol='diamond',
                line=dict(width=3, color='black'),
                opacity=1.0
            ),
            name=f'{description} (Center)',
            showlegend=True,
            hovertemplate=f'<b>{description}</b><br>' +
                         f'Patch: {patch_number}<br>' +
                         f'Face: {face_index}<br>' +
                         f'X: %{{x:.6f}}<br>' +
                         f'Y: %{{y:.6f}}<br>' +
                         f'Z: %{{z:.6f}}<br>' +
                         '<extra></extra>'
        ))
        
        # Process each radius
        for radius_idx, radius in enumerate(patch_radii):
            radius_mm = radius * 1000
            
            if use_pipeline_data:
                # Load pre-filtered data from pipeline CSV files
                results_dir = Path(f'{subject_name}_results/tracked_points')
                patch_description_for_filename = f"{description} (Fixed Patch {radius_mm:.1f}mm)"
                patch_csv_file = results_dir / f"{subject_name}_patch{patch_number}_face{face_index}_{patch_description_for_filename.lower().replace(' ', '_')}_r2mm.csv"
                
                if not patch_csv_file.exists():
                    print(f"   ⚠️  No pipeline data for {radius_mm:.1f}mm - {patch_csv_file.name}")
                    continue
                
                # Load the pipeline data to get point count
                patch_df = pd.read_csv(patch_csv_file)
                time_sec = time_step * 0.001
                time_step_data = patch_df[abs(patch_df['Time (s)'] - time_sec) < 0.0001]
                
                if time_step_data.empty:
                    print(f"   ⚠️  No data for time step {time_step} ({time_sec:.3f}s)")
                    continue
                
                # Reconstruct the patch points using the same filtering as pipeline
                patch_points = find_connected_points_with_normal_filter(
                    df, center_point, radius, 
                    connectivity_threshold=0.001, 
                    normal_angle_threshold=normal_angle_threshold
                )
                
                # Verify against pipeline data
                pipeline_count = time_step_data['Patch Points Count'].iloc[0] if 'Patch Points Count' in time_step_data.columns else len(patch_points)
                match_status = "✅" if len(patch_points) == pipeline_count else "⚠️"
                print(f"   {match_status} {radius_mm:.1f}mm: {len(patch_points)} points (pipeline: {pipeline_count})")
                
            else:
                # Recompute filtering
                print(f"   🔄 Computing {radius_mm:.1f}mm patch with filtering...")
                patch_points = find_connected_points_with_normal_filter(
                    df, center_point, radius, 
                    connectivity_threshold=0.001, 
                    normal_angle_threshold=normal_angle_threshold
                )
                print(f"   ✅ {radius_mm:.1f}mm: {len(patch_points)} filtered points")
            
            if patch_points.empty:
                continue
            
            # Get distinct color for this radius
            radius_color = get_radius_color(base_color, radius_idx)
            
            # Create size based on radius (smaller radius = larger points for visibility)
            size = 5 - (radius_idx * 0.7)  # 5, 4.3, 3.6
            opacity = 0.8
            
            # Add patch points
            fig.add_trace(go.Scatter3d(
                x=patch_points['X (m)'],
                y=patch_points['Y (m)'],
                z=patch_points['Z (m)'],
                mode='markers',
                marker=dict(
                    size=size,
                    color=radius_color,
                    opacity=opacity,
                    line=dict(width=0.5, color='black')
                ),
                name=f'{description} ({radius_mm:.1f}mm - {len(patch_points)} pts)',
                showlegend=True,
                hovertemplate=f'<b>{description}</b><br>' +
                             f'Radius: {radius_mm:.1f}mm<br>' +
                             f'Distance: %{{customdata:.6f}}m<br>' +
                             f'Patch: %{{text}}<br>' +
                             f'X: %{{x:.6f}}<br>' +
                             f'Y: %{{y:.6f}}<br>' +
                             f'Z: %{{z:.6f}}<br>' +
                             '<extra></extra>',
                customdata=patch_points['distance_from_center'],
                text=patch_points['Patch Number']
            ))
    
    # Update layout
    data_source = "Pipeline CSV Data" if use_pipeline_data else "Recomputed Filtering"
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
            # White background with no grid lines
            bgcolor='white',
            xaxis=dict(
                showgrid=False,
                showline=False,
                zeroline=False,
                showticklabels=True,
                backgroundcolor='white'
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                zeroline=False,
                showticklabels=True,
                backgroundcolor='white'
            ),
            zaxis=dict(
                showgrid=False,
                showline=False,
                zeroline=False,
                showticklabels=True,
                backgroundcolor='white'
            )
        ),
        width=1400,
        height=900,
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        uirevision='constant'
    )
    
    # Add annotation
    fig.add_annotation(
        text="🎯 <b>Patch Regions Visualization</b><br>" +
             f"<sub>Data Source: {data_source}</sub><br>" +
             "• 🎯 Diamond markers = tracking centers<br>" +
             "• 🔴 Colored circles = filtered patch regions<br>" +
             "• Patch sizes: 1mm, 2mm, 5mm radius (distinct colors)<br>" +
             f"• Multi-stage filtering: connectivity + normals (≤{normal_angle_threshold}°) + final connectivity<br>" +
             "• Double-click legend to toggle (view stays fixed)<br>" +
             "• Hover for distance and details<br>" +
             "• Rotate/zoom to explore in 3D",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Save the interactive plot
    output_file = f"{subject_name}_patch_regions_t{time_step}.html"
    fig.write_html(output_file)
    
    print(f"\n✅ Patch regions visualization saved as: {output_file}")
    
    # Also save to results directory if it exists
    results_dir = Path(f'{subject_name}_results/interactive')
    if results_dir.exists():
        results_output = results_dir / output_file
        fig.write_html(results_output)
        print(f"📁 Also saved to: {results_output}")
    
    # Print summary
    print(f"\n📊 Visualization Summary:")
    print(f"   • Subject: {subject_name}")
    print(f"   • Time step: {time_step} ({time_step * 0.001:.3f}s)")
    print(f"   • Tracking locations: {len(tracking_locations)}")
    print(f"   • Patch radii: {[f'{r*1000:.1f}mm' for r in patch_radii]}")
    print(f"   • Data source: {data_source}")
    print(f"   • Total airway surface points: {len(df):,}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description='Visualize patch regions from CFD analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Use pre-filtered data from pipeline (recommended)
  python visualize_patch_regions.py --subject OSAMRI007 --timestep 100
  
  # Recompute filtering (slower, for debugging)
  python visualize_patch_regions.py --subject OSAMRI007 --timestep 100 --recompute
  
  # Custom radii
  python visualize_patch_regions.py --subject OSAMRI007 --timestep 100 --radii 1 2 5
  
  # Different normal angle threshold (only affects recompute mode)
  python visualize_patch_regions.py --subject OSAMRI007 --timestep 100 --recompute --normalangle 45

Notes:
  - Default mode uses pre-filtered CSV data from main pipeline (fast)
  - Recompute mode recalculates filtering (slower, for validation)
  - Pipeline data typically starts around timestep 61 (0.061s)
  - Timestep 100 = 0.1s, timestep 200 = 0.2s, etc.
        """
    )
    
    parser.add_argument('--subject', default='OSAMRI007',
                       help='Subject name (default: OSAMRI007)')
    parser.add_argument('--timestep', type=int, default=100,
                       help='Time step to visualize (default: 100 = 0.1s)')
    parser.add_argument('--radii', type=float, nargs='+', default=[1, 2, 5],
                       help='Patch radii in mm (default: 1 2 5)')
    parser.add_argument('--recompute', action='store_true',
                       help='Recompute filtering instead of using pipeline CSV data')
    parser.add_argument('--normalangle', type=float, default=60.0,
                       help='Normal angle threshold in degrees (default: 60.0, only for recompute mode)')
    
    args = parser.parse_args()
    
    # Convert radii from mm to meters
    patch_radii = [r / 1000.0 for r in args.radii]
    
    print(f"🎯 Patch Regions Visualization")
    print(f"Subject: {args.subject}")
    print(f"Time step: {args.timestep} ({args.timestep * 0.001:.3f}s)")
    print(f"Patch radii: {[f'{r:.1f}mm' for r in args.radii]}")
    print(f"Mode: {'Recompute filtering' if args.recompute else 'Use pipeline data'}")
    if args.recompute:
        print(f"Normal angle threshold: {args.normalangle}°")
    print()
    
    fig = visualize_patch_regions(
        subject_name=args.subject,
        time_step=args.timestep,
        patch_radii=patch_radii,
        use_pipeline_data=not args.recompute,
        normal_angle_threshold=args.normalangle
    )
    
    if fig:
        print(f"\n🎉 Success! Open the HTML file in a web browser to explore the 3D visualization.")
    else:
        print(f"\n❌ Failed to create visualization.")

if __name__ == "__main__":
    main() 