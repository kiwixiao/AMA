#!/usr/bin/env python3
"""
Interactive point selection tool for CFD surface analysis.
Extends existing Plotly visualizations with paint brush and lasso selection capabilities.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

class InteractivePointSelector:
    """Interactive tool for selecting surface points using paint brush and lasso tools."""
    
    def __init__(self, subject_name: str, csv_file: Path, tracking_locations: Dict):
        self.subject_name = subject_name
        self.csv_file = csv_file
        self.tracking_locations = tracking_locations
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.selected_points = []
        self.app = None
        
    def create_enhanced_surface_plot(self) -> go.Figure:
        """Create enhanced surface plot with selection capabilities."""
        fig = go.Figure()
        
        # Create point-to-description mapping
        point_to_desc = {}
        for loc in self.tracking_locations['locations']:
            patch_num = loc['patch_number']
            for face_idx in loc['face_indices']:
                point_to_desc[(patch_num, face_idx)] = loc['description']
        
        # Plot all surface points with selection capabilities
        colors = px.colors.qualitative.Set3
        
        for patch_num in sorted(self.df['Patch Number'].unique()):
            patch_mask = self.df['Patch Number'] == patch_num
            patch_points = self.df[patch_mask]
            
            if len(patch_points) == 0:
                continue
            
            # Create hover text with detailed information
            hover_text = [
                f'Patch {patch_num}<br>'
                f'Face Index: {int(face_idx)}<br>'
                f'X: {x:.6f}<br>'
                f'Y: {y:.6f}<br>'
                f'Z: {z:.6f}<br>'
                f'Pressure: {pressure:.2f} Pa<br>'
                f'VdotN: {vdotn:.6f}'
                for face_idx, x, y, z, pressure, vdotn in zip(
                    patch_points['Face Index'],
                    patch_points['X (m)'],
                    patch_points['Y (m)'],
                    patch_points['Z (m)'],
                    patch_points['Total Pressure (Pa)'],
                    patch_points['VdotN']
                )
            ]
            
            fig.add_trace(go.Scatter3d(
                x=patch_points['X (m)'],
                y=patch_points['Y (m)'],
                z=patch_points['Z (m)'],
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
                # Note: selectedpoints not supported for Scatter3d, using built-in selection instead
            ))
        
        # Add tracked points as reference
        tracked_points = []
        tracked_labels = []
        
        for patch_num, face_idx in point_to_desc.keys():
            point_data = self.df[
                (self.df['Patch Number'] == patch_num) & 
                (self.df['Face Index'] == face_idx)
            ]
            if len(point_data) > 0:
                point = point_data.iloc[0]
                tracked_points.append([point['X (m)'], point['Y (m)'], point['Z (m)']])
                tracked_labels.append(f"{point_to_desc[(patch_num, face_idx)]}")
        
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
                name='Reference Points',
                hovertext=tracked_labels,
                hoverinfo='text'
            ))
        
        # Configure layout for selection
        fig.update_layout(
            title=f'Interactive Point Selection - {self.subject_name}<br>'
                  f'<span style="font-size:14px;">Use selection tools to paint/select regions of interest</span>',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            width=1400,
            height=900,
            showlegend=True,
            # Enable selection tools
            dragmode='select',  # Default to lasso select
            selectdirection='diagonal'
        )
        
        # Add selection tools configuration
        fig.update_layout(
            modebar=dict(
                add=[
                    'lasso2d', 'select2d',  # 2D selection tools
                    'pan3d', 'orbitRotation', 'tableRotation',  # 3D navigation
                    'zoom3d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                    'autoScale2d', 'resetScale2d'
                ]
            )
        )
        
        return fig
    
    def create_dash_app(self) -> dash.Dash:
        """Create Dash application with interactive selection."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Create the enhanced plot
        fig = self.create_enhanced_surface_plot()
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2(f"Interactive Surface Selection - {self.subject_name}"),
                    html.P("Use the selection tools to paint regions of interest on the 3D surface."),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    # Selection tools panel
                    dbc.Card([
                        dbc.CardHeader("Selection Tools"),
                        dbc.CardBody([
                            dbc.ButtonGroup([
                                dbc.Button("Lasso Select", id="lasso-btn", color="primary", size="sm"),
                                dbc.Button("Box Select", id="box-btn", color="secondary", size="sm"),
                                dbc.Button("Clear Selection", id="clear-btn", color="warning", size="sm"),
                            ]),
                            html.Hr(),
                            html.Div(id="selection-info"),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Analysis Radius (mm):"),
                                    dcc.Slider(
                                        id="radius-slider",
                                        min=0.5,
                                        max=10.0,
                                        step=0.5,
                                        value=2.0,
                                        marks={i: f"{i}mm" for i in range(1, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ])
                            ]),
                            html.Hr(),
                            dbc.Button("Process Selected Region", id="process-btn", 
                                     color="success", size="lg", className="w-100")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    # 3D plot
                    dcc.Graph(
                        id="surface-plot",
                        figure=fig,
                        style={'height': '800px'}
                    )
                ], width=9)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id="processing-results")
                ])
            ])
        ], fluid=True)
        
        # Callbacks
        @app.callback(
            Output("selection-info", "children"),
            [Input("surface-plot", "selectedData")]
        )
        def update_selection_info(selected_data):
            if not selected_data or 'points' not in selected_data:
                return html.P("No points selected. Use lasso or box select tools.")
            
            num_selected = len(selected_data['points'])
            return html.Div([
                html.P(f"Selected: {num_selected} points"),
                html.Small("Click 'Process Selected Region' to analyze this selection.")
            ])
        
        @app.callback(
            Output("surface-plot", "figure"),
            [Input("lasso-btn", "n_clicks"),
             Input("box-btn", "n_clicks"),
             Input("clear-btn", "n_clicks")],
            [State("surface-plot", "figure")]
        )
        def update_selection_mode(lasso_clicks, box_clicks, clear_clicks, current_fig):
            ctx = callback_context
            if not ctx.triggered:
                return current_fig
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == "lasso-btn":
                current_fig['layout']['dragmode'] = 'lasso'
            elif button_id == "box-btn":
                current_fig['layout']['dragmode'] = 'select'
            elif button_id == "clear-btn":
                # Clear all selections - Note: Scatter3d doesn't support selectedpoints
                # Selection clearing will be handled by Plotly's built-in mechanisms
                pass
            
            return current_fig
        
        @app.callback(
            Output("processing-results", "children"),
            [Input("process-btn", "n_clicks")],
            [State("surface-plot", "selectedData"),
             State("radius-slider", "value")]
        )
        def process_selected_region(n_clicks, selected_data, radius_mm):
            if not n_clicks or not selected_data:
                return ""
            
            # Extract selected point indices and coordinates
            selected_points = selected_data['points']
            
            if not selected_points:
                return dbc.Alert("No points selected!", color="warning")
            
            # Process the selection
            results = self.process_selection(selected_points, radius_mm / 1000.0)
            
            return dbc.Card([
                dbc.CardHeader("Processing Results"),
                dbc.CardBody([
                    html.H5(f"Analyzed {len(selected_points)} selected points"),
                    html.P(f"Analysis radius: {radius_mm} mm"),
                    html.Hr(),
                    html.Pre(results['summary'], style={'background-color': '#f8f9fa', 'padding': '10px'})
                ])
            ])
        
        self.app = app
        return app
    
    def process_selection(self, selected_points: List[Dict], radius: float) -> Dict:
        """Process the selected points and generate analysis."""
        # Extract coordinates from selected points
        coordinates = []
        point_indices = []
        
        for point in selected_points:
            # Get the trace and point index
            trace_index = point.get('curveNumber', 0)
            point_index = point.get('pointNumber', 0)
            
            # Extract coordinates
            x = point.get('x', 0)
            y = point.get('y', 0) 
            z = point.get('z', 0)
            
            coordinates.append((x, y, z))
            point_indices.append((trace_index, point_index))
        
        # Calculate statistics for selected region
        selected_df = self.get_points_from_selection(coordinates, radius)
        
        if len(selected_df) == 0:
            return {'summary': 'No points found in selected region.'}
        
        # Calculate comprehensive statistics
        stats = {
            'num_points': len(selected_df),
            'pressure_mean': selected_df['Total Pressure (Pa)'].mean(),
            'pressure_std': selected_df['Total Pressure (Pa)'].std(),
            'velocity_mean': selected_df['Velocity: Magnitude (m/s)'].mean(),
            'velocity_std': selected_df['Velocity: Magnitude (m/s)'].std(),
            'vdotn_mean': selected_df['VdotN'].mean(),
            'vdotn_std': selected_df['VdotN'].std()
        }
        
        # Create summary
        summary = f"""Selected Region Analysis:
        
Number of points: {stats['num_points']}
Analysis radius: {radius*1000:.1f} mm

Pressure Statistics:
  Mean: {stats['pressure_mean']:.2f} ± {stats['pressure_std']:.2f} Pa
  
Velocity Statistics:
  Mean: {stats['velocity_mean']:.6f} ± {stats['velocity_std']:.6f} m/s
  
VdotN Statistics:
  Mean: {stats['vdotn_mean']:.6f} ± {stats['vdotn_std']:.6f}
"""
        
        # Save selection for further processing
        self.save_selected_region(selected_df, coordinates, radius)
        
        return {'summary': summary, 'data': selected_df, 'stats': stats}
    
    def get_points_from_selection(self, coordinates: List[Tuple], radius: float) -> pd.DataFrame:
        """Get all points within radius of any selected coordinate."""
        all_selected = pd.DataFrame()
        
        for coord in coordinates:
            # Find points within radius of this coordinate
            distances = np.sqrt(
                (self.df['X (m)'] - coord[0])**2 +
                (self.df['Y (m)'] - coord[1])**2 +
                (self.df['Z (m)'] - coord[2])**2
            )
            
            nearby_points = self.df[distances <= radius]
            all_selected = pd.concat([all_selected, nearby_points], ignore_index=True)
        
        # Remove duplicates
        all_selected = all_selected.drop_duplicates()
        return all_selected
    
    def save_selected_region(self, selected_df: pd.DataFrame, coordinates: List[Tuple], radius: float):
        """Save selected region data for further processing."""
        output_dir = Path(f"{self.subject_name}_results") / "interactive_selections"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the selected region data
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selected_region_{timestamp}.csv"
        filepath = output_dir / filename
        
        selected_df.to_csv(filepath, index=False)
        
        # Save selection metadata
        metadata = {
            'timestamp': timestamp,
            'num_selected_points': len(coordinates),
            'analysis_radius_m': radius,
            'selection_coordinates': coordinates,
            'total_points_in_region': len(selected_df),
            'csv_file': str(filepath)
        }
        
        metadata_file = output_dir / f"selection_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved selected region: {filepath}")
        print(f"Saved metadata: {metadata_file}")
    
    def run_app(self, port: int = 8050, debug: bool = True):
        """Run the Dash application."""
        if not self.app:
            self.app = self.create_dash_app()
        
        print(f"Starting interactive selection tool...")
        print(f"Open your browser to: http://localhost:{port}")
        print("Use the selection tools to paint regions of interest!")
        
        self.app.run_server(debug=debug, port=port)

def create_interactive_selector(subject_name: str, 
                              csv_file: Path, 
                              tracking_locations_file: Path = Path("tracking_locations.json")) -> InteractivePointSelector:
    """Create an interactive point selector for the given subject."""
    
    # Load tracking locations
    with open(tracking_locations_file, 'r') as f:
        tracking_locations = json.load(f)
    
    return InteractivePointSelector(subject_name, csv_file, tracking_locations)

if __name__ == "__main__":
    # Example usage
    subject_name = "OSAMRI007"
    csv_file = Path(f"{subject_name}_xyz_tables/XYZ_Internal_Table_table_100.csv")
    
    selector = create_interactive_selector(subject_name, csv_file)
    selector.run_app() 