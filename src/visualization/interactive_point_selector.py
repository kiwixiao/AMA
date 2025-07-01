"""
Interactive 3D Point Selector using PyVista

This module provides an interactive 3D visualization where users can:
1. Click on points to select and highlight them persistently
2. Rotate the view to inspect selected points from different angles
3. Extract patch and face index information for selected points
4. Add selected points to tracking locations

Usage:
    python -c "from src.visualization.interactive_point_selector import run_interactive_selector; run_interactive_selector('OSAMRI007', 100)"
"""

import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import sys
import os

# Configure PyVista for better performance
pv.set_plot_theme("document")  # Clean, professional theme
pv.global_theme.background = 'white'
pv.global_theme.window_size = [1200, 800]
pv.global_theme.font.size = 12


class InteractivePointSelector:
    """Interactive 3D point selector using PyVista for anatomical landmark selection."""
    
    def __init__(self, subject_name: str, timestep: int = 100):
        self.subject_name = subject_name
        self.timestep = timestep
        self.selected_points = []  # List of selected point info
        self.point_actors = []     # List of highlight actors
        self.plotter = None
        self.mesh_data = None
        self.patch_info = {}       # Maps point indices to patch/face info
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load the XYZ data and patch information."""
        print(f"🎯 Loading data for {self.subject_name} at timestep {self.timestep}")
        
        # Load the XYZ data
        csv_file = f"{self.subject_name}_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_{self.timestep}.csv"
        
        if not Path(csv_file).exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")
            
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} points from {csv_file}")
        
        # Extract coordinates and patch information
        points = df[['X (m)', 'Y (m)', 'Z (m)']].values
        face_indices = df['Face Index'].values
        
        # Detect patches based on Face Index resets
        patch_numbers = []
        current_patch = 1
        prev_face_idx = -1
        
        for face_idx in face_indices:
            if face_idx == 0 and prev_face_idx > 0:
                current_patch += 1
            patch_numbers.append(current_patch)
            prev_face_idx = face_idx
        
        # Store patch information for each point
        for i, (patch_num, face_idx) in enumerate(zip(patch_numbers, face_indices)):
            self.patch_info[i] = {
                'patch': int(patch_num),
                'face_index': int(face_idx),
                'coordinates': points[i].tolist()
            }
        
        # Create PyVista point cloud
        self.mesh_data = pv.PolyData(points)
        
        # Add patch information as scalars for coloring
        self.mesh_data['patch_number'] = np.array(patch_numbers)
        self.mesh_data['face_index'] = face_indices
        
        print(f"✅ Loaded {len(np.unique(patch_numbers))} patches")
        
    def _on_point_pick(self, *args):
        """Callback when a point is picked/clicked."""
        print(f"🔍 Debug: Callback received {len(args)} arguments: {args}")
        
        # PyVista may pass different arguments depending on the version
        # Try to extract the point_id from the arguments
        point_id = None
        
        if len(args) >= 1:
            # First argument might be point_id
            potential_id = args[0]
            if hasattr(potential_id, '__len__') and len(potential_id) > 0:
                point_id = potential_id[0]  # Take first element if it's an array
            elif hasattr(potential_id, '__len__') and len(potential_id) == 0:
                print("⚠️ Empty selection received")
                return  # Empty selection
            else:
                point_id = potential_id
        
        if point_id is None:
            print("❌ Could not extract point_id from callback arguments")
            return
        
        # Convert to int if it's a numpy scalar
        try:
            point_id = int(point_id)
        except (ValueError, TypeError) as e:
            print(f"❌ Could not convert point_id to int: {point_id}, error: {e}")
            return
        
        if point_id < 0 or point_id >= len(self.patch_info):
            print(f"⚠️ Invalid point ID: {point_id} (valid range: 0-{len(self.patch_info)-1})")
            return
            
        print(f"\n🎯 Point Clicked - Debug Info:")
        print(f"   Raw point_id: {point_id} (type: {type(point_id)})")
        print(f"   Total points available: {len(self.patch_info)}")
        
        if point_id not in self.patch_info:
            print(f"❌ Point ID {point_id} not found in patch_info!")
            print(f"Available range: 0 to {len(self.patch_info)-1}")
            return
            
        point_info = self.patch_info[point_id]
        coords = point_info['coordinates']
        patch_num = point_info['patch']
        face_idx = point_info['face_index']
        
        print(f"✅ Point Found:")
        print(f"   Point ID: {point_id}")
        print(f"   Coordinates: ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f})")
        print(f"   Patch: {patch_num}")
        print(f"   Face Index: {face_idx}")
        
        # Check if point is already selected
        existing_idx = None
        for i, selected in enumerate(self.selected_points):
            if selected['point_id'] == point_id:
                existing_idx = i
                break
                
        if existing_idx is not None:
            # Deselect the point
            print(f"🔄 Deselecting point {point_id} (removing from selection)")
            self.selected_points.pop(existing_idx)
            
            # Remove highlight sphere and text label
            if existing_idx < len(self.point_actors):
                # Remove highlight sphere
                self.plotter.remove_actor(self.point_actors[existing_idx]['sphere'])
                # Remove text label
                self.plotter.remove_actor(self.point_actors[existing_idx]['text'])
                self.point_actors.pop(existing_idx)
        else:
            # Select the point
            print(f"✅ Selecting point {point_id} (adding to selection)")
            
            selection_info = {
                'point_id': point_id,
                'coordinates': coords,
                'patch': patch_num,
                'face_index': face_idx
            }
            self.selected_points.append(selection_info)
            
            # Add larger highlight sphere for better visibility
            highlight_sphere = pv.Sphere(radius=0.004, center=coords)  # Larger sphere
            sphere_actor = self.plotter.add_mesh(
                highlight_sphere,
                color='red',
                opacity=0.9,
                name=f'selected_point_{point_id}'
            )
            
            # Add text label showing patch and face info
            label_text = f"P{patch_num}:F{face_idx}"
            text_position = [coords[0], coords[1], coords[2] + 0.008]  # Slightly above the point
            text_actor = self.plotter.add_point_labels(
                [text_position], 
                [label_text],
                point_color='red',
                point_size=0,  # Hide the point, just show text
                font_size=16,
                text_color='darkred',
                shape_color='white',
                shape_opacity=0.8,
                margin=3
            )
            
            # Store both actors for easy removal
            actor_pair = {
                'sphere': sphere_actor,
                'text': text_actor,
                'point_id': point_id
            }
            self.point_actors.append(actor_pair)
            
        # Update status and display selection info
        self._update_status()
        self._display_selection_summary()
        
    def _update_status(self):
        """Update the status display with current selections."""
        if not self.selected_points:
            print(f"📊 Status: No points selected")
            return
            
        print(f"📊 Status: {len(self.selected_points)} point(s) selected")
        for i, point in enumerate(self.selected_points):
            print(f"   {i+1}. Patch {point['patch']}, Face {point['face_index']} at ({point['coordinates'][0]:.4f}, {point['coordinates'][1]:.4f}, {point['coordinates'][2]:.4f})")
    
    def _display_selection_summary(self):
        """Display a summary of all selected points for manual tracking file updates."""
        if not self.selected_points:
            return
            
        print(f"\n📋 SELECTION SUMMARY (for manual tracking locations update):")
        print(f"{'='*60}")
        print(f"Subject: {self.subject_name}")
        print(f"Timestep: {self.timestep}")
        print(f"Selected Points: {len(self.selected_points)}")
        print(f"{'='*60}")
        
        for i, point in enumerate(self.selected_points):
            print(f"{i+1}. Patch {point['patch']}, Face {point['face_index']}")
            print(f"   Coordinates: ({point['coordinates'][0]:.6f}, {point['coordinates'][1]:.6f}, {point['coordinates'][2]:.6f})")
            print(f"   JSON Key: \"{point['patch']},{point['face_index']}\"")
            print(f"   Suggested Name: \"Selected_Point_{i+1}_Patch{point['patch']}_Face{point['face_index']}\"")
            print()
        
        print(f"{'='*60}")
        print(f"💡 Copy the JSON keys and names above to update your tracking locations file!")
        print(f"{'='*60}\n")
    
    def _save_selections(self):
        """Save selected points to tracking locations."""
        if not self.selected_points:
            print("❌ No points selected to save")
            return
            
        # Load existing tracking locations
        tracking_file = f"{self.subject_name}_tracking_locations.json"
        
        if Path(tracking_file).exists():
            with open(tracking_file, 'r') as f:
                tracking_data = json.load(f)
        else:
            tracking_data = {}
            
        print(f"\n💾 Saving {len(self.selected_points)} selected points...")
        
        for i, point in enumerate(self.selected_points):
            # Create a descriptive name
            point_name = f"Selected_Point_{i+1}_Patch{point['patch']}_Face{point['face_index']}"
            
            # Add to tracking locations
            key = f"{point['patch']},{point['face_index']}"
            tracking_data[key] = point_name
            
            print(f"   ✅ Saved: {point_name}")
            
        # Save updated tracking locations
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
            
        print(f"💾 Tracking locations saved to: {tracking_file}")
        
    def _clear_selections(self):
        """Clear all selected points."""
        print("🔄 Clearing all selections...")
        
        # Remove all highlight actors (both spheres and text labels)
        for actor_pair in self.point_actors:
            self.plotter.remove_actor(actor_pair['sphere'])
            self.plotter.remove_actor(actor_pair['text'])
            
        self.selected_points.clear()
        self.point_actors.clear()
        
        print("✅ All selections cleared")
        
    def _print_help(self):
        """Print help information."""
        help_text = """
🎯 Interactive Point Selector - Controls:

Mouse Controls:
  • Right-click on point → Select/deselect point (red highlight + label)
  • Left drag           → Rotate view
  • Right drag          → Pan view  
  • Scroll wheel        → Zoom in/out
  • Middle-click        → Alternative point selection

Keyboard Controls:
  • 'h' or '?'        → Show this help
  • 's'              → Save selected points to tracking locations
  • 'c'              → Clear all selections
  • 'r'              → Reset camera view
  • 'q' or ESC       → Quit selector

Features:
  ✅ Large, easy-to-click points
  ✅ Click again to remove wrong selections
  ✅ Patch and Face numbers shown in 3D scene (P##:F##)
  ✅ Selection summary printed for manual JSON updates
  ✅ Persistent highlighting while rotating view

Workflow:
  1. Click points to select them (red sphere + P##:F## label)
  2. Rotate view to inspect from different angles
  3. Click again to deselect wrong choices
  4. Check terminal for selection summary with JSON keys
  5. Press 's' to auto-save OR manually update tracking JSON
  6. Press 'q' to quit

The selection summary shows exact JSON keys for manual updates!
        """
        print(help_text)
        
    def run(self):
        """Run the interactive point selector."""
        print(f"\n🚀 Starting Interactive Point Selector")
        print(f"Subject: {self.subject_name}, Timestep: {self.timestep}")
        
        # Create plotter
        self.plotter = pv.Plotter(title=f"Interactive Point Selector - {self.subject_name}")
        
        # Add the point cloud with patch coloring - larger points for easier selection
        self.plotter.add_mesh(
            self.mesh_data,
            scalars='patch_number',
            point_size=8,  # Much larger points for easier clicking
            render_points_as_spheres=True,
            pickable=True,
            cmap='tab20',  # Distinct colors for different patches
            opacity=0.8
        )
        
        # Enable point picking with more robust settings
        self.plotter.enable_point_picking(
            callback=self._on_point_pick,
            show_message=True,  # Show picking message for debugging
            font_size=14,
            color='darkred',
            use_mesh=True,  # Use the mesh for picking
            show_point=True  # Show the picked point
        )
        
        # Add text instructions
        instructions = (
            "🎯 RIGHT-CLICK points to select/deselect (P##:F## labels)\n"
            "🔄 LEFT-DRAG to rotate view and inspect selections\n"
            "📋 Check terminal for selection summary\n"
            "💾 Press 's' to save, 'h' for help"
        )
        
        self.plotter.add_text(
            instructions,
            position='upper_left',
            font_size=12,
            color='black'
        )
        
        # Add keyboard callbacks
        self.plotter.add_key_event('s', self._save_selections)
        self.plotter.add_key_event('c', self._clear_selections)
        self.plotter.add_key_event('h', self._print_help)
        self.plotter.add_key_event('question', self._print_help)
        
        # Set up camera for good initial view
        self.plotter.camera.position = (0.1, 0.1, 0.1)
        self.plotter.camera.focal_point = (0, 0, 0)
        
        # Show help initially
        self._print_help()
        
        # Start the interactive session
        print("\n🎯 Interactive selector is ready!")
        print("Click on points to select them, then rotate to inspect.")
        
        self.plotter.show()
        
        return self.selected_points


def run_interactive_selector(subject_name: str, timestep: int = 100):
    """
    Run the interactive point selector.
    
    Args:
        subject_name: Subject identifier (e.g., 'OSAMRI007')
        timestep: Time step to visualize (default: 100)
    
    Returns:
        List of selected point information
    """
    try:
        selector = InteractivePointSelector(subject_name, timestep)
        return selector.run()
    except Exception as e:
        print(f"❌ Error running interactive selector: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        subject = sys.argv[1]
        timestep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    else:
        subject = "OSAMRI007"
        timestep = 100
        
    run_interactive_selector(subject, timestep) 