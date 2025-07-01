#!/usr/bin/env python3
"""
Enhanced 3D Paint Brush Tool for CFD Surface Analysis
Uses click-based selection and region expansion for 3D data
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

class PaintBrush3DEnhanced:
    """Enhanced 3D Paint brush tool with click-based selection."""
    
    def __init__(self, subject_name: str):
        self.subject_name = subject_name
        self.results_dir = Path(f"{subject_name}_results")
        self.painted_regions_dir = self.results_dir / "painted_regions"
        self.painted_regions_dir.mkdir(parents=True, exist_ok=True)
        
    def create_interactive_tool(self, time_point: int = 100):
        """Create enhanced 3D paint brush tool with click-based selection."""
        print(f"🎨 Creating Enhanced 3D Paint Brush for {self.subject_name} at time point {time_point}")
        
        # Load data
        csv_file = Path(f"{self.subject_name}_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_{time_point}.csv")
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"Loaded {len(df)} surface points")
        
        # Load tracking locations for reference points
        tracking_file = Path('tracking_locations.json')
        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                tracking_locations = json.load(f)
        else:
            tracking_locations = {'locations': []}
        
        # Create the 3D plot
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        
        # Plot surface points by patch
        for patch_num in sorted(df['Patch Number'].unique()):
            patch_points = df[df['Patch Number'] == patch_num]
            if len(patch_points) == 0:
                continue
            
            # Create custom data for click handling
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
                    f"VdotN: {row['VdotN']:.5f}<br>"
                    f"<i>Click to add to selection</i>"
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
        
        # Add reference points
        self._add_reference_points(fig, df, tracking_locations)
        
        # Configure layout for 3D viewing
        fig.update_layout(
            title=dict(
                text=f'🎨 Enhanced 3D Paint Brush - {self.subject_name} (t={time_point/1000:.1f}s)<br>'
                     f'<span style="font-size:14px;">Click points to select, use controls to expand regions</span>',
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
            showlegend=True
        )
        
        # Save HTML with enhanced JavaScript
        output_file = self.results_dir / "interactive" / f"{self.subject_name}_3d_enhanced_paint_brush_t{time_point/1000:.1f}s.html"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._create_enhanced_html(fig, time_point)
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Enhanced 3D Paint brush tool saved: {output_file}")
        print(f"🎯 Features: Click selection + region expansion + 3D visualization!")
        
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
    
    def _create_enhanced_html(self, fig: go.Figure, time_point: int) -> str:
        """Create enhanced HTML with advanced JavaScript for 3D selection."""
        html_content = fig.to_html(include_plotlyjs=True)
        
        # Add advanced JavaScript for 3D click-based selection
        custom_js = f"""
        <style>
        .control-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: white;
            border: 2px solid #007bff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            min-width: 250px;
        }}
        .control-panel h3 {{
            margin: 0 0 10px 0;
            color: #007bff;
            font-size: 16px;
        }}
        .control-row {{
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .control-button {{
            padding: 8px 15px;
            margin: 2px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }}
        .select-btn {{ background: #28a745; color: white; }}
        .clear-btn {{ background: #dc3545; color: white; }}
        .save-btn {{ background: #007bff; color: white; }}
        .expand-btn {{ background: #ffc107; color: black; }}
        .status {{
            margin-top: 10px;
            padding: 8px;
            border-radius: 5px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            font-size: 12px;
        }}
        </style>
        
        <div class="control-panel">
            <h3>🎨 3D Paint Brush Controls</h3>
            
            <div class="control-row">
                <label>Selection Mode:</label>
                <button class="control-button select-btn" onclick="toggleSelectionMode()">
                    <span id="mode-text">Click to Select</span>
                </button>
            </div>
            
            <div class="control-row">
                <label>Expansion Radius:</label>
                <input type="range" id="radius-slider" min="1" max="10" value="2" style="width: 100px;">
                <span id="radius-value">2mm</span>
            </div>
            
            <div class="control-row">
                <button class="control-button expand-btn" onclick="expandSelection()">
                    🔍 Expand Region
                </button>
                <button class="control-button clear-btn" onclick="clearSelection()">
                    🧹 Clear
                </button>
            </div>
            
            <div class="control-row">
                <button class="control-button save-btn" onclick="saveSelection()" style="width: 100%;">
                    💾 Save Selection to Pipeline
                </button>
            </div>
            
            <div class="status" id="status">
                Ready - Click points to start selection
            </div>
        </div>
        
        <script>
        let selectionMode = true;
        let selectedPoints = [];
        let allData = [];
        let selectedIndices = new Set();
        
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('🎨 Enhanced 3D Paint Brush Loaded!');
            console.log('✅ Click-based selection enabled');
            console.log('📋 Instructions:');
            console.log('1. Click points to add to selection');
            console.log('2. Use Expand Region to grow selection');
            console.log('3. Save selection when ready');
            
            // Setup radius slider
            const radiusSlider = document.getElementById('radius-slider');
            const radiusValue = document.getElementById('radius-value');
            radiusSlider.addEventListener('input', function() {{
                radiusValue.textContent = this.value + 'mm';
            }});
            
            // Get the plotly div and setup click handling
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            
            // Extract all data for processing
            plotDiv.on('plotly_afterplot', function() {{
                allData = [];
                const plotData = plotDiv.data;
                plotData.forEach((trace, traceIndex) => {{
                    if (trace.customdata) {{
                        trace.customdata.forEach((customPoint, pointIndex) => {{
                            allData.push({{
                                traceIndex: traceIndex,
                                pointIndex: pointIndex,
                                x: customPoint[0],
                                y: customPoint[1], 
                                z: customPoint[2],
                                patch: customPoint[3],
                                face: customPoint[4],
                                pressure: customPoint[5],
                                vdotn: customPoint[6],
                                velocity: customPoint[7]
                            }});
                        }});
                    }}
                }});
                console.log(`📊 Loaded ${{allData.length}} data points for selection`);
            }});
            
            // Handle point clicks
            plotDiv.on('plotly_click', function(eventData) {{
                if (!selectionMode) return;
                
                const point = eventData.points[0];
                if (point && point.customdata) {{
                    addPointToSelection(point);
                }}
            }});
        }});
        
        function toggleSelectionMode() {{
            selectionMode = !selectionMode;
            const modeText = document.getElementById('mode-text');
            const button = modeText.parentElement;
            
            if (selectionMode) {{
                modeText.textContent = 'Click to Select';
                button.className = 'control-button select-btn';
                updateStatus('Selection mode ON - click points to select');
            }} else {{
                modeText.textContent = 'Selection OFF';
                button.className = 'control-button clear-btn';
                updateStatus('Selection mode OFF - click to pan/zoom');
            }}
        }}
        
        function addPointToSelection(point) {{
            const customData = point.customdata;
            const pointKey = `${{customData[3]}}_${{customData[4]}}`; // patch_face
            
            if (!selectedIndices.has(pointKey)) {{
                selectedIndices.add(pointKey);
                selectedPoints.push({{
                    x: customData[0],
                    y: customData[1],
                    z: customData[2],
                    patch: customData[3],
                    face: customData[4],
                    pressure: customData[5],
                    vdotn: customData[6],
                    velocity: customData[7]
                }});
                
                updateStatus(`Added point from Patch ${{customData[3]}} - Total: ${{selectedPoints.length}} points`);
                highlightSelectedPoints();
            }}
        }}
        
        function expandSelection() {{
            if (selectedPoints.length === 0) {{
                alert('⚠️ No points selected! Click some points first.');
                return;
            }}
            
            const radius = parseFloat(document.getElementById('radius-slider').value) / 1000; // Convert mm to m
            const initialCount = selectedPoints.length;
            
            selectedPoints.forEach(selectedPoint => {{
                allData.forEach(dataPoint => {{
                    const pointKey = `${{dataPoint.patch}}_${{dataPoint.face}}`;
                    if (!selectedIndices.has(pointKey)) {{
                        const distance = Math.sqrt(
                            Math.pow(dataPoint.x - selectedPoint.x, 2) +
                            Math.pow(dataPoint.y - selectedPoint.y, 2) +
                            Math.pow(dataPoint.z - selectedPoint.z, 2)
                        );
                        
                        if (distance <= radius) {{
                            selectedIndices.add(pointKey);
                            selectedPoints.push(dataPoint);
                        }}
                    }}
                }});
            }});
            
            const newCount = selectedPoints.length - initialCount;
            updateStatus(`Expanded selection: +${{newCount}} points (Total: ${{selectedPoints.length}})`);
            highlightSelectedPoints();
        }}
        
        function highlightSelectedPoints() {{
            // This would require updating the plot traces to highlight selected points
            // For now, we'll just log the selection
            console.log(`🎨 Selection updated: ${{selectedPoints.length}} points selected`);
            
            if (selectedPoints.length > 0) {{
                const avgPressure = selectedPoints.reduce((sum, p) => sum + p.pressure, 0) / selectedPoints.length;
                const avgVelocity = selectedPoints.reduce((sum, p) => sum + p.velocity, 0) / selectedPoints.length;
                
                console.log(`📊 Selection stats - Pressure: ${{avgPressure.toFixed(2)}} Pa, Velocity: ${{avgVelocity.toFixed(6)}} m/s`);
            }}
        }}
        
        function clearSelection() {{
            selectedPoints = [];
            selectedIndices.clear();
            updateStatus('Selection cleared');
            console.log('🧹 Selection cleared');
        }}
        
        function saveSelection() {{
            if (selectedPoints.length === 0) {{
                alert('⚠️ No points selected! Click some points first.');
                return;
            }}
            
            const regionName = prompt('Enter a name for this painted region:', 'painted_region_1');
            if (!regionName) return;
            
            // Calculate statistics
            const pressures = selectedPoints.map(p => p.pressure);
            const velocities = selectedPoints.map(p => p.velocity);
            const vdotns = selectedPoints.map(p => p.vdotn);
            
            const statistics = {{
                pressure_mean: pressures.reduce((a,b) => a+b) / pressures.length,
                pressure_std: Math.sqrt(pressures.map(x => Math.pow(x - pressures.reduce((a,b) => a+b) / pressures.length, 2)).reduce((a,b) => a+b) / pressures.length),
                velocity_mean: velocities.reduce((a,b) => a+b) / velocities.length,
                velocity_std: Math.sqrt(velocities.map(x => Math.pow(x - velocities.reduce((a,b) => a+b) / velocities.length, 2)).reduce((a,b) => a+b) / velocities.length),
                vdotn_mean: vdotns.reduce((a,b) => a+b) / vdotns.length,
                vdotn_std: Math.sqrt(vdotns.map(x => Math.pow(x - vdotns.reduce((a,b) => a+b) / vdotns.length, 2)).reduce((a,b) => a+b) / vdotns.length)
            }};
            
            const selectionToSave = {{
                region_name: regionName,
                subject_name: "{self.subject_name}",
                time_point: {time_point},
                timestamp: new Date().toISOString(),
                coordinates: selectedPoints.map(p => [p.x, p.y, p.z]),
                patch_info: selectedPoints,
                statistics: statistics,
                num_points: selectedPoints.length,
                selection_method: "3d_click_and_expand"
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
            
            updateStatus(`Saved ${{selectedPoints.length}} points as ${{regionName}}`);
            alert(`✅ Selection saved! Downloaded: ${{regionName}}_t{time_point/1000:.1f}s.json`);
            
            console.log('💾 Selection saved!');
            console.log('📊 Statistics:', statistics);
        }}
        
        function updateStatus(message) {{
            document.getElementById('status').textContent = message;
        }}
        
        // Make functions available globally
        window.selectedPoints = selectedPoints;
        window.allData = allData;
        </script>
        """
        
        # Insert custom JavaScript and CSS before closing body tag
        html_content = html_content.replace('</body>', custom_js + '</body>')
        
        return html_content

def create_3d_enhanced_paint_brush(subject_name: str, time_point: int = 100):
    """Create enhanced 3D paint brush tool."""
    tool = PaintBrush3DEnhanced(subject_name)
    return tool.create_interactive_tool(time_point) 