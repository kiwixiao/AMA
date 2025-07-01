#!/usr/bin/env python3
"""
2D Paint Brush Tool for CFD Surface Analysis
Uses 2D projections to enable better selection tools
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PaintBrush2D:
    """2D Paint brush tool with working selection."""
    
    def __init__(self, subject_name: str):
        self.subject_name = subject_name
        self.results_dir = Path(f"{subject_name}_results")
        self.painted_regions_dir = self.results_dir / "painted_regions"
        self.painted_regions_dir.mkdir(parents=True, exist_ok=True)
        
    def create_interactive_tool(self, time_point: int = 100, projection: str = 'xy'):
        """
        Create 2D paint brush tool with working selection.
        
        Args:
            time_point: Time point to analyze
            projection: Projection plane ('xy', 'xz', 'yz')
        """
        print(f"🎨 Creating 2D paint brush tool for {self.subject_name} at time point {time_point}")
        
        # Load data
        csv_file = Path(f"{self.subject_name}_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_{time_point}.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"Loaded {len(df)} surface points")
        
        # Create the 2D plot
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        
        # Set projection coordinates
        if projection == 'xy':
            x_col, y_col = 'X (m)', 'Y (m)'
            x_title, y_title = 'X (m)', 'Y (m)'
        elif projection == 'xz':
            x_col, y_col = 'X (m)', 'Z (m)'
            x_title, y_title = 'X (m)', 'Z (m)'
        else:  # yz
            x_col, y_col = 'Y (m)', 'Z (m)'
            x_title, y_title = 'Y (m)', 'Z (m)'
        
        # Plot points by patch
        for patch_num in sorted(df['Patch Number'].unique()):
            patch_points = df[df['Patch Number'] == patch_num]
            if len(patch_points) == 0:
                continue
            
            # Create custom data for selection
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
                    f"3D: ({row['X (m)']:.4f}, {row['Y (m)']:.4f}, {row['Z (m)']:.4f})<br>"
                    f"Pressure: {row['Total Pressure (Pa)']:.1f} Pa<br>"
                    f"Velocity: {row['Velocity: Magnitude (m/s)']:.5f} m/s<br>"
                    f"VdotN: {row['VdotN']:.5f}"
                )
            
            fig.add_trace(go.Scatter(
                x=patch_points[x_col],
                y=patch_points[y_col],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[patch_num % len(colors)],
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                name=f'Patch {patch_num}',
                text=hover_text,
                hoverinfo='text',
                customdata=custom_data
            ))
        
        # Configure layout with working selection tools
        fig.update_layout(
            title=dict(
                text=f'🎨 2D Paint Brush - {self.subject_name} (t={time_point/1000:.1f}s, {projection.upper()} view)<br>'
                     f'<span style="font-size:14px;">✅ Selection tools enabled - drag to select regions!</span>',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title=x_title,
            yaxis_title=y_title,
            width=1200,
            height=800,
            showlegend=True,
            # Enable selection - this works in 2D!
            dragmode='select',
                         selectdirection='d'
        )
        
        # Configure toolbar - 2D selection tools work!
        fig.update_layout(
            modebar=dict(
                add=[
                    'select2d',      # Box select - WORKS in 2D!
                    'lasso2d',       # Lasso select - WORKS in 2D!
                ],
                remove=['autoScale2d', 'resetScale2d'],
                orientation='v',
                bgcolor='rgba(255,255,255,0.9)',
                color='#444',
                activecolor='#007bff'
            )
        )
        
        # Add clear instructions
        fig.add_annotation(
            text="🎨 <b>WORKING SELECTION TOOLS!</b><br>"
                 "✅ Click 📦 Box Select for rectangles<br>"
                 "✅ Click 🖱️ Lasso Select for freehand<br>"
                 "✅ Drag to select points<br>"
                 "✅ Selected points turn highlighted<br>"
                 "✅ Use Save button to export selection",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=12, color="green"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="green",
            borderwidth=2
        )
        
        # Save HTML with JavaScript
        output_file = self.results_dir / "interactive" / f"{self.subject_name}_2d_paint_brush_{projection}_t{time_point/1000:.1f}s.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._create_enhanced_html(fig, time_point, projection)
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ 2D Paint brush tool saved: {output_file}")
        print(f"🎯 This version has WORKING selection tools!")
        
        return fig, df
    
    def _create_enhanced_html(self, fig: go.Figure, time_point: int, projection: str) -> str:
        """Create enhanced HTML with JavaScript for 2D selection."""
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
            console.log('🎨 2D Paint Brush Tool Loaded!');
            console.log('✅ Selection tools are ENABLED and WORKING!');
            console.log('📋 Instructions:');
            console.log('1. Click Box Select (📦) or Lasso Select (🖱️) in toolbar');
            console.log('2. Drag to select points - they will highlight!');
            console.log('3. Click Save button to export selection');
            
            // Add save button
            const saveButton = document.createElement('button');
            saveButton.innerHTML = '💾 Save Selection to Pipeline';
            saveButton.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                padding: 15px 25px;
                background: #28a745;
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
            console.log('\\n🎨 SELECTION SUCCESSFUL!');
            console.log(`✅ Selected ${{points.length}} points`);
            
            selectionData.coordinates = [];
            selectionData.patch_info = [];
            
            let pressures = [];
            let velocities = [];
            let vdotns = [];
            
            points.forEach(point => {{
                if (point.customdata) {{
                    const [x, y, z, patch, face, pressure, vdotn, velocity] = point.customdata;
                    const coord = [x, y, z];  // Always use 3D coordinates
                    selectionData.coordinates.push(coord);
                    
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
                alert('⚠️ No selection made! Please select points first using the selection tools.');
                return;
            }}
            
            const regionName = prompt('Enter a name for this painted region:', 'painted_region_1');
            if (!regionName) return;
            
            const selectionToSave = {{
                region_name: regionName,
                subject_name: "{self.subject_name}",
                time_point: {time_point},
                projection: "{projection}",
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
            console.log(`✅ Saved ${{selectionData.coordinates.length}} points`);
            alert(`✅ Selection saved! Downloaded: ${{regionName}}_t{time_point/1000:.1f}s.json`);
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

def create_2d_paint_brush(subject_name: str, time_point: int = 100, projection: str = 'xy'):
    """Create 2D paint brush tool with working selection."""
    tool = PaintBrush2D(subject_name)
    return tool.create_interactive_tool(time_point, projection) 