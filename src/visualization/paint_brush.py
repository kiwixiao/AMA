#!/usr/bin/env python3
"""
Interactive Paint Brush Tool for CFD Surface Analysis
Part of the main CFD analysis pipeline
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaintBrushTool:
    """Interactive paint brush tool for CFD surface analysis."""
    
    def __init__(self, subject_name: str):
        self.subject_name = subject_name
        self.results_dir = Path(f"{subject_name}_results")
        self.painted_regions_dir = self.results_dir / "painted_regions"
        self.painted_regions_dir.mkdir(parents=True, exist_ok=True)
        
    def create_interactive_tool(self, time_point: int = 100) -> Tuple[go.Figure, pd.DataFrame]:
        """
        Create an interactive paint brush tool for selecting surface regions.
        
        Args:
            time_point: Time point to analyze (e.g., 100 for t=0.1s)
            
        Returns:
            Tuple of (plotly figure, dataframe)
        """
        logger.info(f"Creating paint brush tool for {self.subject_name} at time point {time_point}")
        
        # Load data
        csv_file = Path(f"{self.subject_name}_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_{time_point}.csv")
        if not csv_file.exists():
            logger.error(f"Data file not found: {csv_file}")
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        df = pd.read_csv(csv_file, low_memory=False)
        logger.info(f"Loaded {len(df)} surface points")
        
        # Load tracking locations
        tracking_file = Path('tracking_locations.json')
        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                tracking_locations = json.load(f)
        else:
            logger.warning("tracking_locations.json not found, continuing without reference points")
            tracking_locations = {'locations': []}
        
        # Create the interactive plot
        fig = go.Figure()
        
        # Plot surface points by patch with distinct colors
        colors = px.colors.qualitative.Set3
        
        for patch_num in sorted(df['Patch Number'].unique()):
            patch_mask = df['Patch Number'] == patch_num
            patch_points = df[patch_mask]
            
            if len(patch_points) == 0:
                continue
            
            # Create custom data for JavaScript access
            custom_data = []
            hover_text = []
            for _, row in patch_points.iterrows():
                custom_data.append([
                    row['X (m)'], row['Y (m)'], row['Z (m)'],
                    patch_num, int(row['Face Index']),
                    row['Total Pressure (Pa)'], row['VdotN'],
                    row['Velocity: Magnitude (m/s)']
                ])
                hover_text.append(
                    f"<b>Patch {patch_num}</b><br>"
                    f"Face: {int(row['Face Index'])}<br>"
                    f"Position: ({row['X (m)']:.4f}, {row['Y (m)']:.4f}, {row['Z (m)']:.4f})<br>"
                    f"Pressure: {row['Total Pressure (Pa)']:.1f} Pa<br>"
                    f"Velocity: {row['Velocity: Magnitude (m/s)']:.5f} m/s<br>"
                    f"VdotN: {row['VdotN']:.5f}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=patch_points['X (m)'],
                y=patch_points['Y (m)'],
                z=patch_points['Z (m)'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors[patch_num % len(colors)],
                    opacity=0.8,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                name=f'Patch {patch_num}',
                text=hover_text,
                hoverinfo='text',
                customdata=custom_data
            ))
        
        # Add reference points if available
        self._add_reference_points(fig, df, tracking_locations)
        
        # Configure layout with selection tools
        self._configure_layout(fig, time_point)
        
        # Save the interactive plot
        output_file = self.results_dir / "interactive" / f"{self.subject_name}_paint_brush_t{time_point/1000:.1f}s.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced HTML with JavaScript
        html_content = self._create_enhanced_html(fig, time_point)
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Paint brush tool saved: {output_file}")
        
        return fig, df
    
    def _add_reference_points(self, fig: go.Figure, df: pd.DataFrame, tracking_locations: Dict):
        """Add reference tracking points to the plot."""
        point_to_desc = {}
        for loc in tracking_locations.get('locations', []):
            patch_num = loc['patch_number']
            for face_idx in loc['face_indices']:
                point_to_desc[(patch_num, face_idx)] = loc['description']
        
        tracked_points = []
        tracked_labels = []
        
        for (patch_num, face_idx), description in point_to_desc.items():
            point_data = df[
                (df['Patch Number'] == patch_num) & 
                (df['Face Index'] == face_idx)
            ]
            if len(point_data) > 0:
                point = point_data.iloc[0]
                tracked_points.append([point['X (m)'], point['Y (m)'], point['Z (m)']])
                tracked_labels.append(description)
        
        if tracked_points:
            tracked_points = np.array(tracked_points)
            fig.add_trace(go.Scatter3d(
                x=tracked_points[:, 0],
                y=tracked_points[:, 1],
                z=tracked_points[:, 2],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='darkred')
                ),
                text=tracked_labels,
                textposition="top center",
                name='📍 Reference Points',
                hovertext=[f"<b>{label}</b>" for label in tracked_labels],
                hoverinfo='text'
            ))
    
    def _configure_layout(self, fig: go.Figure, time_point: int):
        """Configure the plot layout with selection tools."""
        fig.update_layout(
            title=dict(
                text=f'🎨 Paint Brush Tool - {self.subject_name} (t={time_point/1000:.1f}s)<br>'
                     f'<span style="font-size:14px;">Select regions and save to pipeline</span>',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='white'
            ),
            width=1400,
            height=900,
            showlegend=True,
            # Enable selection tools - CRITICAL for 3D plots
            dragmode='select',
            selectdirection='d'
        )
        
        # Configure toolbar explicitly with all necessary tools
        fig.update_layout(
            modebar=dict(
                add=[
                    'select2d',      # Box select
                    'lasso2d',       # Lasso select
                ],
                remove=[
                    'autoScale2d', 'resetScale2d', 'toggleHover', 'toggleSpikelines',
                    'hoverClosestCartesian', 'hoverCompareCartesian'
                ],
                orientation='v',
                bgcolor='rgba(255,255,255,0.9)',
                color='#444',
                activecolor='#007bff'
            )
        )
        
        # Add instruction annotation with selection tips
        fig.add_annotation(
            text="🎨 <b>SELECTION INSTRUCTIONS</b><br>"
                 "1. Look for toolbar on the right side<br>"
                 "2. Click 📦 (Box Select) or 🖱️ (Lasso Select)<br>"
                 "3. Drag to select points<br>"
                 "4. Selected points will be highlighted<br>"
                 "5. Click 'Save Selection' button to save",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color="blue"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="blue",
            borderwidth=1
        )
    
    def _create_enhanced_html(self, fig: go.Figure, time_point: int) -> str:
        """Create enhanced HTML with JavaScript for selection handling."""
        html_content = fig.to_html(include_plotlyjs=True)
        
        # Add custom JavaScript for selection handling
        custom_js = f"""
        <script>
        let selectedPoints = [];
        let selectionData = {{
            coordinates: [],
            patch_info: [],
            statistics: {{}}
        }};
        
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('🎨 Paint Brush Tool Loaded!');
            console.log('📋 Instructions:');
            console.log('1. Use Box Select (📦) or Lasso Select (🖱️) tools in toolbar');
            console.log('2. Selected data will appear in console');
            console.log('3. Use "Save Selection" button to save to pipeline');
            
            // Add save button
            const saveButton = document.createElement('button');
            saveButton.innerHTML = '💾 Save Selection to Pipeline';
            saveButton.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                padding: 10px 20px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            `;
            saveButton.onclick = saveSelectionToPipeline;
            document.body.appendChild(saveButton);
            
            // Get the plotly div
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            
            // Listen for selection events
            plotDiv.on('plotly_selected', function(eventData) {{
                if (eventData && eventData.points) {{
                    selectedPoints = eventData.points;
                    processSelection(selectedPoints);
                }}
            }});
            
            // Listen for deselect events
            plotDiv.on('plotly_deselect', function() {{
                selectedPoints = [];
                selectionData = {{coordinates: [], patch_info: [], statistics: {{}}}};
                console.log('🧹 Selection cleared');
            }});
        }});
        
        function processSelection(points) {{
            console.log('\\n🎨 NEW SELECTION MADE!');
            console.log(`📊 Selected ${{points.length}} points`);
            
            selectionData.coordinates = [];
            selectionData.patch_info = [];
            
            let pressures = [];
            let velocities = [];
            let vdotns = [];
            
            points.forEach(point => {{
                const coord = [point.x, point.y, point.z];
                selectionData.coordinates.push(coord);
                
                if (point.customdata) {{
                    const [x, y, z, patch, face, pressure, vdotn, velocity] = point.customdata;
                    selectionData.patch_info.push({{
                        coordinates: coord,
                        patch_number: patch,
                        face_index: face,
                        pressure: pressure,
                        velocity: velocity,
                        vdotn: vdotn
                    }});
                    
                    pressures.push(pressure);
                    velocities.push(velocity);
                    vdotns.push(vdotn);
                }}
            }});
            
            // Calculate statistics
            if (pressures.length > 0) {{
                selectionData.statistics = {{
                    pressure_mean: pressures.reduce((a,b) => a+b) / pressures.length,
                    pressure_std: Math.sqrt(pressures.map(x => Math.pow(x - pressures.reduce((a,b) => a+b) / pressures.length, 2)).reduce((a,b) => a+b) / pressures.length),
                    velocity_mean: velocities.reduce((a,b) => a+b) / velocities.length,
                    velocity_std: Math.sqrt(velocities.map(x => Math.pow(x - velocities.reduce((a,b) => a+b) / velocities.length, 2)).reduce((a,b) => a+b) / velocities.length),
                    vdotn_mean: vdotns.reduce((a,b) => a+b) / vdotns.length,
                    vdotn_std: Math.sqrt(vdotns.map(x => Math.pow(x - vdotns.reduce((a,b) => a+b) / vdotns.length, 2)).reduce((a,b) => a+b) / vdotns.length)
                }};
                
                console.log('📊 SELECTION STATISTICS:');
                console.log(`Pressure: ${{selectionData.statistics.pressure_mean.toFixed(2)}} ± ${{selectionData.statistics.pressure_std.toFixed(2)}} Pa`);
                console.log(`Velocity: ${{selectionData.statistics.velocity_mean.toFixed(6)}} ± ${{selectionData.statistics.velocity_std.toFixed(6)}} m/s`);
                console.log(`VdotN: ${{selectionData.statistics.vdotn_mean.toFixed(6)}} ± ${{selectionData.statistics.vdotn_std.toFixed(6)}}`);
            }}
            
            window.selectedCoordinates = selectionData.coordinates;
            window.selectionStats = selectionData.statistics;
        }}
        
        function saveSelectionToPipeline() {{
            if (selectionData.coordinates.length === 0) {{
                alert('⚠️ No selection made! Please select points first.');
                return;
            }}
            
            const regionName = prompt('Enter a name for this painted region:', 'painted_region_1');
            if (!regionName) return;
            
            const selectionToSave = {{
                region_name: regionName,
                subject_name: "{self.subject_name}",
                time_point: {time_point},
                timestamp: new Date().toISOString(),
                coordinates: selectionData.coordinates,
                patch_info: selectionData.patch_info,
                statistics: selectionData.statistics,
                num_points: selectionData.coordinates.length
            }};
            
            // Create download link
            const dataStr = JSON.stringify(selectionToSave, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `${{regionName}}_t{time_point/1000:.1f}s.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            
            console.log('💾 Selection saved as JSON file!');
            console.log('📋 To process in Python pipeline:');
            console.log(`from src.visualization.paint_brush import PaintBrushTool`);
            console.log(`tool = PaintBrushTool("{self.subject_name}")`);
            console.log(`tool.process_saved_selection("${{regionName}}_t{time_point/1000:.1f}s.json")`);
        }}
        
        // Make functions available globally
        window.getSelectionData = () => selectionData;
        window.selectedCoordinates = [];
        window.saveSelectionToPipeline = saveSelectionToPipeline;
        </script>
        """
        
        # Insert custom JavaScript before closing body tag
        html_content = html_content.replace('</body>', custom_js + '</body>')
        
        return html_content
    
    def process_saved_selection(self, json_file: str, radius_mm: float = 2.0) -> Tuple[pd.DataFrame, Dict]:
        """
        Process a saved selection JSON file.
        
        Args:
            json_file: Path to the saved selection JSON file
            radius_mm: Expansion radius in millimeters
            
        Returns:
            Tuple of (selected points dataframe, statistics)
        """
        json_path = Path(json_file)
        if not json_path.exists():
            # Try in painted_regions directory
            json_path = self.painted_regions_dir / json_file
            if not json_path.exists():
                raise FileNotFoundError(f"Selection file not found: {json_file}")
        
        logger.info(f"Processing saved selection: {json_path}")
        
        with open(json_path, 'r') as f:
            selection_data = json.load(f)
        
        coordinates = selection_data['coordinates']
        time_point = selection_data['time_point']
        region_name = selection_data['region_name']
        
        return self.process_coordinates(coordinates, time_point, radius_mm, region_name)
    
    def process_coordinates(self, coordinates_list: List[List[float]], 
                          time_point: int, 
                          radius_mm: float = 2.0,
                          region_name: str = "painted_region") -> Tuple[pd.DataFrame, Dict]:
        """
        Process coordinates selected from the paint brush tool.
        
        Args:
            coordinates_list: List of [x, y, z] coordinates
            time_point: Time point used
            radius_mm: Expansion radius in millimeters
            region_name: Name for this region
            
        Returns:
            Tuple of (selected points dataframe, statistics)
        """
        logger.info(f"Processing {len(coordinates_list)} painted points with {radius_mm}mm radius")
        
        # Load the data
        csv_file = Path(f"{self.subject_name}_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_{time_point}.csv")
        df = pd.read_csv(csv_file, low_memory=False)
        
        radius_m = radius_mm / 1000.0
        all_selected = pd.DataFrame()
        
        # Find all points within radius of any painted coordinate
        for coord in coordinates_list:
            x, y, z = coord
            
            # Calculate distances
            distances = np.sqrt(
                (df['X (m)'] - x)**2 +
                (df['Y (m)'] - y)**2 +
                (df['Z (m)'] - z)**2
            )
            
            # Find nearby points
            nearby_points = df[distances <= radius_m]
            all_selected = pd.concat([all_selected, nearby_points], ignore_index=True)
        
        # Remove duplicates
        all_selected = all_selected.drop_duplicates()
        
        logger.info(f"Found {len(all_selected)} points in painted region")
        
        if len(all_selected) > 0:
            # Calculate statistics
            stats = {
                'region_name': region_name,
                'time_point': time_point,
                'num_points': len(all_selected),
                'pressure_mean': all_selected['Total Pressure (Pa)'].mean(),
                'pressure_std': all_selected['Total Pressure (Pa)'].std(),
                'velocity_mean': all_selected['Velocity: Magnitude (m/s)'].mean(),
                'velocity_std': all_selected['Velocity: Magnitude (m/s)'].std(),
                'vdotn_mean': all_selected['VdotN'].mean(),
                'vdotn_std': all_selected['VdotN'].std()
            }
            
            # Save results
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.painted_regions_dir / f"{region_name}_{timestamp}.csv"
            all_selected.to_csv(csv_file, index=False)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'region_name': region_name,
                'time_point': time_point,
                'num_painted_points': len(coordinates_list),
                'analysis_radius_mm': radius_mm,
                'painted_coordinates': coordinates_list,
                'total_points_in_region': len(all_selected),
                'statistics': stats
            }
            
            metadata_file = self.painted_regions_dir / f"{region_name}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved results: {csv_file}")
            logger.info(f"Saved metadata: {metadata_file}")
            
            return all_selected, stats
        else:
            logger.warning("No points found in painted region!")
            return pd.DataFrame(), {}
    
    def get_all_painted_regions(self) -> List[Dict]:
        """Get metadata for all painted regions."""
        metadata_files = list(self.painted_regions_dir.glob("*_metadata_*.json"))
        regions = []
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            regions.append(metadata)
        
        return sorted(regions, key=lambda x: x['timestamp'], reverse=True)

def create_paint_brush_tool(subject_name: str, time_point: int = 100) -> PaintBrushTool:
    """
    Convenience function to create and launch paint brush tool.
    
    Args:
        subject_name: Subject identifier
        time_point: Time point to analyze
        
    Returns:
        PaintBrushTool instance
    """
    tool = PaintBrushTool(subject_name)
    fig, df = tool.create_interactive_tool(time_point)
    return tool 