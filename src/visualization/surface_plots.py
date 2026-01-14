import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pandas as pd
from typing import List, Tuple

def plot_3d_trajectory(xyz_file, trajectory_points, labels):
    """Plot 3D visualization of surface points and trajectory."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Read surface data
    df = pd.read_csv(xyz_file)
    
    # Get unique patches based on Face Index resets
    patches = []
    current_patch = []
    prev_face_idx = -1
    patch_face_indices = []
    patch_sizes = []
    patch_start_coords = []
    
    # First identify all the patches
    for idx, row in df.iterrows():
        face_idx = row['Face Index']
        if face_idx == 0 and prev_face_idx > 0:
            if current_patch:
                patch_array = np.array(current_patch)
                patches.append(patch_array)
                patch_face_indices.append(prev_face_idx)
                patch_sizes.append(len(current_patch))
                patch_start = patch_array[:10].mean(axis=0)
                patch_start_coords.append((patch_start[1], patch_start[2]))
            current_patch = []
        current_patch.append([row['X (m)'], row['Y (m)'], row['Z (m)']])
        prev_face_idx = face_idx
    
    if current_patch:
        patch_array = np.array(current_patch)
        patches.append(patch_array)
        patch_face_indices.append(prev_face_idx)
        patch_sizes.append(len(current_patch))
        patch_start = patch_array[:10].mean(axis=0)
        patch_start_coords.append((patch_start[1], patch_start[2]))
    
    # Sort patches by Y and Z coordinates
    patch_indices = list(range(len(patches)))
    patch_indices.sort(key=lambda i: (patch_start_coords[i][0], patch_start_coords[i][1]))
    
    # Take first 5 patches
    selected_indices = patch_indices[:5]
    
    # Create colormap
    cmap = plt.cm.Set2
    colors = [cmap(i) for i in np.linspace(0, 1, 5)]
    
    # Plot selected patches
    for i, patch_idx in enumerate(selected_indices):
        patch = patches[patch_idx]
        ax.scatter(patch[:, 0], patch[:, 1], patch[:, 2], 
                  c=[colors[i]], s=10, alpha=0.7)
        ax.scatter([], [], [], 
                  c=[colors[i]], s=100, alpha=0.7,
                  label=f'Patch {patch_idx+1}\n{patch_sizes[patch_idx]} points\nStart (Y:{patch_start_coords[patch_idx][0]:.4f},\nZ:{patch_start_coords[patch_idx][1]:.4f})')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('5 Most Negative Y,Z Surface Patches\nEach color represents a distinct surface patch')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.savefig('cfd_3d_trajectory.pdf', bbox_inches='tight')
    plt.close()

def plot_3d_interactive_all_patches(data_source, tracking_points, subject_name: str, output_dir: str = None, time_point: int = None):
    """
    Create an interactive 3D visualization of surface patches with original face indices.
    
    Args:
        data_source: Either a CSV file path or HDF5 file path
        tracking_points: List of tracking point configurations
        subject_name: Subject name for output files
        output_dir: Directory to save HTML file
        time_point: Time point to visualize (for HDF5 data)
    """
    print("\nCreating interactive 3D visualization...")
    
    # Create a dictionary mapping (patch_number, face_index) to description
    point_to_desc = {}
    for loc in tracking_points:
        patch_num = loc['patch_number']
        for face_idx in loc['face_indices']:
            point_to_desc[(patch_num, face_idx)] = loc['description']
    
    # Load data from appropriate source
    if str(data_source).endswith('.h5') or str(data_source).endswith('.hdf5'):
        # Load from HDF5 file
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
        
        print(f"📊 Loading data from HDF5 file: {data_source}")
        df = load_hdf5_data_for_html_plots(str(data_source), time_point)
        
        if df is None:
            print("❌ Failed to load HDF5 data")
            return
        
        data_source_type = "HDF5"
    else:
        # Load from CSV file (legacy support)
        print(f"📊 Loading data from CSV file: {data_source}")
        df = pd.read_csv(data_source, low_memory=False)
        data_source_type = "CSV"
    
    points = []
    face_indices = []
    patch_numbers = []
    tracked_points = []
    tracked_labels = []
    
    # Process data based on source type
    if data_source_type == "CSV":
        # Original CSV processing logic
        current_patch = 1
        prev_face_idx = -1
        
        for _, row in df.iterrows():
            point_coords = [row['X (m)'], row['Y (m)'], row['Z (m)']]
            face_idx = row['Face Index']
            
            # Detect new patch when Face Index resets to 0 after being > 0
            if face_idx == 0 and prev_face_idx > 0:
                current_patch += 1
            
            points.append(point_coords)
            face_indices.append(face_idx)
            patch_numbers.append(current_patch)
            
            # Check if this is a tracked point
            point_key = (current_patch, int(face_idx))
            if point_key in point_to_desc:
                tracked_points.append(point_coords)
                tracked_labels.append(f"{point_to_desc[point_key]}\n(Patch {point_key[0]}, Face {point_key[1]})")
                print(f"Found tracked point: {point_to_desc[point_key]} at patch {point_key[0]}, face {point_key[1]}")
            
            prev_face_idx = face_idx
    
    else:  # HDF5 data
        # HDF5 data already has Patch Number column added
        for _, row in df.iterrows():
            point_coords = [row['X (m)'], row['Y (m)'], row['Z (m)']]
            face_idx = row['Face Index']
            patch_num = row['Patch Number']
            
            points.append(point_coords)
            face_indices.append(face_idx)
            patch_numbers.append(patch_num)
            
            # Check if this is a tracked point
            point_key = (int(patch_num), int(face_idx))
            if point_key in point_to_desc:
                tracked_points.append(point_coords)
                tracked_labels.append(f"{point_to_desc[point_key]}\n(Patch {point_key[0]}, Face {point_key[1]})")
                print(f"Found tracked point: {point_to_desc[point_key]} at patch {point_key[0]}, face {point_key[1]}")
    
    points = np.array(points)
    face_indices = np.array(face_indices)
    patch_numbers = np.array(patch_numbers)
    
    print(f"Found {len(np.unique(patch_numbers))} patches")
    
    # Create Plotly figure
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    
    for patch_num in sorted(np.unique(patch_numbers)):
        patch_mask = patch_numbers == patch_num
        patch_points = points[patch_mask]
        patch_face_indices = face_indices[patch_mask]
        
        if len(patch_points) == 0:
            continue
            
        hover_text = [
            f'Patch {int(patch_num)}<br>'
            f'Original Face Index: {int(face_idx)}<br>'
            f'X: {x:.6f}<br>'
            f'Y: {y:.6f}<br>'
            f'Z: {z:.6f}'
            for face_idx, (x, y, z) in zip(patch_face_indices, patch_points)
        ]
        
        fig.add_trace(go.Scatter3d(
            x=patch_points[:, 0],
            y=patch_points[:, 1],
            z=patch_points[:, 2],
            mode='markers',
            marker=dict(
                size=1.5,  # Smaller points for better performance and visual clarity
                color=colors[int(patch_num) % len(colors)],  # Convert to int to avoid numpy.float64 indexing error
                opacity=0.8,  # Good visibility without being too heavy
                line=dict(width=0.2, color='rgba(0,0,0,0.2)'),  # Minimal outline
                sizemode='diameter'  # More consistent sizing
            ),
            name=f'Patch {int(patch_num)}<br>Face Indices: {int(np.min(patch_face_indices))}-{int(np.max(patch_face_indices))}',
            text=hover_text,
            hoverinfo='text'
        ))
    
    if tracked_points:
        tracked_points = np.array(tracked_points)
        fig.add_trace(go.Scatter3d(
            x=tracked_points[:, 0],
            y=tracked_points[:, 1],
            z=tracked_points[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond'
            ),
            text=tracked_labels,
            textposition="top center",
            name='Tracked Points',
            hovertext=[f"{label}" for label in tracked_labels],
            hoverinfo='text'
        ))
    
    # Update title to reflect data source
    time_info = f" (t={time_point})" if time_point is not None else ""
    fig.update_layout(
        title=f'Surface Patches with Tracked Points - {subject_name}{time_info}<br>Data Source: {data_source_type}<br>Each color represents a different patch<br><sub>Use selection tools in toolbar to select points</sub>',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),  # Better initial viewing angle
                up=dict(x=0, y=0, z=1)
            ),
            # Enhanced lighting for better depth perception
            xaxis=dict(
                backgroundcolor='rgba(240,240,240,0.5)',
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                backgroundcolor='rgba(240,240,240,0.5)',
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True,
                zeroline=False
            ),
            zaxis=dict(
                backgroundcolor='rgba(240,240,240,0.5)',
                gridcolor='rgba(200,200,200,0.3)',
                showgrid=True,
                zeroline=False
            ),
            bgcolor='rgba(248,248,248,1)'  # Light background for contrast
        ),
        width=1400,
        height=800,
        showlegend=True,
        legend=dict(
            title=f'🎯 Interactive 3D View ({data_source_type} Data):<br>• Rotate: Click and drag<br>• Zoom: Scroll wheel<br>• Pan: Shift + Click and drag<br>• Hover for point details',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        # 3D navigation mode (selection not supported in 3D scatter)
        dragmode='orbit'  # 3D rotation and navigation
    )
    
    # Save HTML file in organized directory structure
    # Include time point in filename for clarity (first file in breathing cycle)
    if time_point is not None:
        time_suffix = f"_first_breathing_cycle_t{time_point}ms"
    else:
        time_suffix = ""

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        html_filename = output_dir / f'{subject_name}_surface_patches_interactive{time_suffix}.html'
    else:
        html_filename = f'{subject_name}_surface_patches_interactive{time_suffix}.html'
    
    # Add JavaScript for navigation help panel (hover shows patch/face info via Plotly tooltip)
    custom_js = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        console.log('🎯 Surface patches 3D visualization loaded');

        // Add navigation help panel
        var helpDiv = document.createElement('div');
        helpDiv.id = 'navigation-help';
        helpDiv.innerHTML = `
            <h4 style="margin: 0 0 8px 0; color: #2c3e50;">🎯 3D Navigation</h4>
            <div style="font-size: 12px; line-height: 1.4;">
                <div style="margin-bottom: 6px;">
                    <strong>🔄 Rotate:</strong> Click and drag<br>
                    <strong>🔍 Zoom:</strong> Scroll wheel<br>
                    <strong>📱 Pan:</strong> Shift + Click and drag<br>
                    <strong>👆 Hover:</strong> Shows Patch & Face Index
                </div>
                <div style="color: #27ae60; font-size: 11px; font-style: italic;">
                    Use PyVista picker for fast point selection:<br>
                    python src/main.py --point-picker
                </div>
            </div>
        `;
        helpDiv.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.98);
            border: 2px solid #3498db;
            border-radius: 10px;
            padding: 12px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            max-width: 280px;
        `;
        document.body.appendChild(helpDiv);

        console.log('✅ Navigation help initialized');
    }});
    </script>
    """
    
    # Write HTML with custom JavaScript
    with open(html_filename, 'w') as f:
        html_content = fig.to_html(include_plotlyjs=True)
        # Insert custom JavaScript before closing body tag
        html_content = html_content.replace('</body>', custom_js + '</body>')
        f.write(html_content)
    
    print(f"Saved interactive visualization with persistent selection: {html_filename}")


def visualize_remesh_comparison(
    hdf5_path: str,
    before_timestep_ms: float,
    after_timestep_ms: float,
    tracking_locations: list,
    subject_name: str,
    output_dir: Path,
    remesh_event_num: int = 1
) -> str:
    """
    Generate side-by-side HTML comparing before/after remesh point mapping.

    Args:
        hdf5_path: Path to HDF5 file
        before_timestep_ms: Timestep (ms) before remesh
        after_timestep_ms: Timestep (ms) after remesh
        tracking_locations: List of tracking location dicts with post_remesh data
        subject_name: Subject name for output files
        output_dir: Directory to save HTML file
        remesh_event_num: Which remesh event (1-indexed)

    Returns:
        Path to generated HTML file
    """
    from plotly.subplots import make_subplots

    # Import data loading function
    try:
        from ..data_processing.trajectory import load_hdf5_data_for_html_plots
    except ImportError:
        import sys
        import os
        src_dir = os.path.dirname(os.path.dirname(__file__))
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from data_processing.trajectory import load_hdf5_data_for_html_plots

    print(f"\n📊 Generating remesh comparison visualization (Event {remesh_event_num})...")
    print(f"   Before: {before_timestep_ms}ms, After: {after_timestep_ms}ms")

    # Load data for both timesteps
    df_before = load_hdf5_data_for_html_plots(str(hdf5_path), int(before_timestep_ms))
    df_after = load_hdf5_data_for_html_plots(str(hdf5_path), int(after_timestep_ms))

    if df_before is None or df_after is None:
        print(f"❌ Failed to load HDF5 data for remesh comparison")
        return None

    # Create side-by-side subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[
            f'BEFORE Remesh (t={before_timestep_ms}ms)',
            f'AFTER Remesh (t={after_timestep_ms}ms)'
        ],
        horizontal_spacing=0.05
    )

    colors = px.colors.qualitative.Set3

    # Helper function to add surface traces
    def add_surface_traces(df, col, opacity=0.5):
        for patch_num in sorted(df['Patch Number'].unique()):
            patch_mask = df['Patch Number'] == patch_num
            patch_df = df[patch_mask]

            if len(patch_df) == 0:
                continue

            fig.add_trace(
                go.Scatter3d(
                    x=patch_df['X (m)'].values,
                    y=patch_df['Y (m)'].values,
                    z=patch_df['Z (m)'].values,
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=colors[int(patch_num) % len(colors)],
                        opacity=opacity
                    ),
                    name=f'Patch {int(patch_num)}',
                    showlegend=(col == 1),  # Only show legend for first plot
                    hoverinfo='skip'  # Skip hover for surface points
                ),
                row=1, col=col
            )

    # Add surface for both views
    add_surface_traces(df_before, col=1)
    add_surface_traces(df_after, col=2)

    # Add tracked points - BEFORE (original locations)
    before_points = []
    before_labels = []
    before_hovers = []

    for loc in tracking_locations:
        desc = loc.get('description', 'Unknown')
        patch = loc.get('patch_number', 0)
        face = loc.get('face_indices', [0])[0]
        coords = loc.get('coordinates', [0, 0, 0])

        if coords and coords != [0, 0, 0] and coords != [0.0, 0.0, 0.0]:
            before_points.append(coords)
            before_labels.append(desc)
            before_hovers.append(
                f"<b>{desc}</b><br>"
                f"Patch: {patch}<br>"
                f"Face: {face}<br>"
                f"X: {coords[0]:.6f}<br>"
                f"Y: {coords[1]:.6f}<br>"
                f"Z: {coords[2]:.6f}"
            )

    if before_points:
        before_points = np.array(before_points)
        fig.add_trace(
            go.Scatter3d(
                x=before_points[:, 0],
                y=before_points[:, 1],
                z=before_points[:, 2],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='diamond'),
                text=before_labels,
                textposition="top center",
                textfont=dict(size=10, color='darkred'),
                name='Original Points',
                hovertext=before_hovers,
                hoverinfo='text'
            ),
            row=1, col=1
        )

    # Add tracked points - AFTER (mapped locations)
    after_points = []
    after_labels = []
    after_hovers = []

    for loc in tracking_locations:
        desc = loc.get('description', 'Unknown')

        # Get post_remesh data (handle both single and multiple remesh)
        post_remesh = loc.get('post_remesh')
        post_remesh_list = loc.get('post_remesh_list', [])

        if post_remesh_list and len(post_remesh_list) >= remesh_event_num:
            mapping = post_remesh_list[remesh_event_num - 1]
        elif post_remesh:
            mapping = post_remesh
        else:
            continue

        if not mapping:
            continue

        new_patch = mapping.get('patch_number', 0)
        new_face = mapping.get('face_index', 0)
        new_coords = mapping.get('coordinates', [0, 0, 0])
        distance = mapping.get('distance_mm', mapping.get('distance', 0))

        if new_coords and new_coords != [0, 0, 0]:
            after_points.append(new_coords)
            after_labels.append(desc)
            after_hovers.append(
                f"<b>{desc}</b><br>"
                f"New Patch: {new_patch}<br>"
                f"New Face: {new_face}<br>"
                f"X: {new_coords[0]:.6f}<br>"
                f"Y: {new_coords[1]:.6f}<br>"
                f"Z: {new_coords[2]:.6f}<br>"
                f"<b>Mapping Distance: {distance:.3f}mm</b>"
            )

    if after_points:
        after_points = np.array(after_points)
        fig.add_trace(
            go.Scatter3d(
                x=after_points[:, 0],
                y=after_points[:, 1],
                z=after_points[:, 2],
                mode='markers+text',
                marker=dict(size=10, color='blue', symbol='diamond'),
                text=after_labels,
                textposition="top center",
                textfont=dict(size=10, color='darkblue'),
                name='Mapped Points',
                hovertext=after_hovers,
                hoverinfo='text'
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Remesh Point Mapping Verification - {subject_name} (Event {remesh_event_num})<br>'
                 f'<sub>Red diamonds = original points | Blue diamonds = mapped points | Hover for details</sub>',
            x=0.5
        ),
        width=1600,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Update 3D scene settings for both subplots
    scene_settings = dict(
        aspectmode='data',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)'
    )
    fig.update_scenes(scene_settings)

    # Save HTML
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_filename = output_dir / f'{subject_name}_remesh_{remesh_event_num}_comparison_t{int(before_timestep_ms)}_t{int(after_timestep_ms)}.html'

    fig.write_html(str(html_filename))
    print(f"✅ Saved remesh comparison: {html_filename}")

    return str(html_filename)