#!/usr/bin/env python3
"""
Interactive STL Surface Painter

This tool provides an interactive 3D interface for painting arbitrary regions
on STL surfaces. Users can select and paint surface areas, save painted regions,
and export coordinates for CFD analysis mapping.

Usage:
    python -m src.surface_painting.stl_painter --stl path/to/surface.stl
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pickle


class STLPainter:
    """Interactive STL surface painting tool."""
    
    def __init__(self, stl_path: Path, scale_factor: float = 0.001):
        """
        Initialize the STL painter.
        
        Args:
            stl_path: Path to STL file
            scale_factor: Scale factor to apply to STL coordinates (e.g., 0.001 for mm to m)
        """
        self.stl_path = stl_path
        self.scale_factor = scale_factor
        self.mesh = None
        self.plotter = None
        
        # Painting state
        self.painted_points = set()
        self.painted_regions = {}  # region_name -> set of point indices
        self.current_region = "region_1"
        self.brush_size = 5.0  # mm
        self.painting_mode = False
        
        # Colors
        self.default_color = [0.8, 0.8, 0.8]  # Light gray
        self.painted_color = [1.0, 0.2, 0.2]  # Red
        self.region_colors = {
            'region_1': [1.0, 0.2, 0.2],  # Red
            'region_2': [0.2, 1.0, 0.2],  # Green  
            'region_3': [0.2, 0.2, 1.0],  # Blue
            'region_4': [1.0, 1.0, 0.2],  # Yellow
            'region_5': [1.0, 0.2, 1.0],  # Magenta
        }
        
        self.load_stl()
        
    def load_stl(self):
        """Load and prepare STL mesh."""
        print(f"Loading STL file: {self.stl_path}")
        
        try:
            # Load STL mesh
            self.mesh = pv.read(str(self.stl_path))
            
            # Apply scale factor
            if self.scale_factor != 1.0:
                self.mesh.points *= self.scale_factor
                print(f"Applied scale factor: {self.scale_factor}")
            
            # Initialize point colors (all default)
            n_points = self.mesh.n_points
            colors = np.tile(self.default_color, (n_points, 1))
            self.mesh.point_data['colors'] = colors
            
            print(f"Loaded mesh: {n_points:,} points, {self.mesh.n_cells:,} cells")
            print(f"Bounds: X[{self.mesh.bounds[0]:.3f}, {self.mesh.bounds[1]:.3f}] "
                  f"Y[{self.mesh.bounds[2]:.3f}, {self.mesh.bounds[3]:.3f}] "
                  f"Z[{self.mesh.bounds[4]:.3f}, {self.mesh.bounds[5]:.3f}]")
            
        except Exception as e:
            print(f"Error loading STL: {e}")
            raise
    
    def setup_plotter(self):
        """Setup PyVista plotter with interactive controls."""
        self.plotter = pv.Plotter()
        
        # Add mesh with point colors
        self.plotter.add_mesh(
            self.mesh, 
            scalars='colors',
            rgb=True,
            point_size=3,
            render_points_as_spheres=True,
            pickable=True
        )
        
        # Setup camera and lighting
        self.plotter.camera.position = (0.1, 0.1, 0.1)
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.add_axes()
        
        # Add text instructions
        instructions = [
            "STL Surface Painter Controls:",
            "• Click and drag to paint",
            "• 'p' - Toggle painting mode",
            "• '1-5' - Switch regions",
            "• 'c' - Clear current region",
            "• 'C' - Clear all regions", 
            "• 's' - Save painted regions",
            "• 'l' - Load painted regions",
            "• 'e' - Export coordinates",
            "• 'q' - Quit"
        ]
        
        self.plotter.add_text(
            "\n".join(instructions),
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        # Add status text
        self.update_status_text()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup keyboard and mouse callbacks."""
        # Register callbacks using proper PyVista method
        self.plotter.add_key_event('p', lambda: self.toggle_painting_mode())
        self.plotter.add_key_event('1', lambda: self.switch_region('region_1'))
        self.plotter.add_key_event('2', lambda: self.switch_region('region_2'))
        self.plotter.add_key_event('3', lambda: self.switch_region('region_3'))
        self.plotter.add_key_event('4', lambda: self.switch_region('region_4'))
        self.plotter.add_key_event('5', lambda: self.switch_region('region_5'))
        self.plotter.add_key_event('c', lambda: self.clear_current_region())
        self.plotter.add_key_event('C', lambda: self.clear_all_regions())
        self.plotter.add_key_event('s', lambda: self.save_painted_regions())
        self.plotter.add_key_event('l', lambda: self.load_painted_regions())
        self.plotter.add_key_event('e', lambda: self.export_coordinates())
        self.plotter.add_key_event('q', lambda: self.safe_quit())
        
        # Enable point picking for painting
        self.plotter.enable_point_picking(callback=self.paint_picked_point, show_message=False)
    
    def update_status_text(self):
        """Update status information display."""
        if hasattr(self, 'status_actor'):
            self.plotter.remove_actor(self.status_actor)
        
        # Count painted points per region
        region_counts = {}
        for region_name, points in self.painted_regions.items():
            region_counts[region_name] = len(points)
        
        total_painted = sum(region_counts.values())
        
        status_lines = [
            f"Painting Mode: {'ON' if self.painting_mode else 'OFF'}",
            f"Current Region: {self.current_region}",
            f"Brush Size: {self.brush_size:.1f}mm",
            f"Total Painted: {total_painted:,} points",
            "",
            "Painted Regions:"
        ]
        
        for region_name, count in region_counts.items():
            color_name = self.get_color_name(region_name)
            percentage = (count / self.mesh.n_points) * 100 if count > 0 else 0
            status_lines.append(f"  {region_name}: {count:,} pts ({percentage:.1f}%) {color_name}")
        
        self.status_actor = self.plotter.add_text(
            "\n".join(status_lines),
            position='upper_right',
            font_size=10,
            color='yellow'
        )
    
    def get_color_name(self, region_name: str) -> str:
        """Get color name for region."""
        color_map = {
            'region_1': 'Red',
            'region_2': 'Green', 
            'region_3': 'Blue',
            'region_4': 'Yellow',
            'region_5': 'Magenta'
        }
        return color_map.get(region_name, 'Unknown')
    
    def toggle_painting_mode(self):
        """Toggle painting mode on/off."""
        self.painting_mode = not self.painting_mode
        self.update_status_text()
        print(f"Painting mode: {'ON' if self.painting_mode else 'OFF'}")
    
    def switch_region(self, region_name: str):
        """Switch to a different region."""
        self.current_region = region_name
        self.update_status_text()
        print(f"Switched to {region_name}")
    
    def paint_picked_point(self, point):
        """Handle point picking for painting."""
        if self.painting_mode and point is not None:
            self.paint_at_point(point)
    
    def safe_quit(self):
        """Safely quit with save prompt."""
        if self.painted_regions:
            print("\n💾 Saving painted regions before exit...")
            try:
                self.save_painted_regions()
                self.export_coordinates()
                print("✅ Saved successfully!")
            except Exception as e:
                print(f"❌ Save error: {e}")
        
        self.plotter.close()
    
    def paint_at_point(self, clicked_point: np.ndarray):
        """Paint points near the clicked location."""
        # Find points within brush size
        distances = np.linalg.norm(self.mesh.points - clicked_point, axis=1)
        brush_radius = self.brush_size * self.scale_factor  # Convert to mesh units
        
        points_to_paint = np.where(distances <= brush_radius)[0]
        
        if len(points_to_paint) == 0:
            return
        
        # Add to current region
        if self.current_region not in self.painted_regions:
            self.painted_regions[self.current_region] = set()
        
        # Count new points (not already painted)
        new_points = set(points_to_paint) - self.painted_regions[self.current_region]
        self.painted_regions[self.current_region].update(points_to_paint)
        
        # Update colors
        self.update_mesh_colors()
        
        # Show feedback
        total_in_region = len(self.painted_regions[self.current_region])
        if len(new_points) > 0:
            print(f"Painted {len(new_points)} new points in {self.current_region} (total: {total_in_region:,})")
        
        self.update_status_text()
    
    def update_mesh_colors(self):
        """Update mesh point colors based on painted regions."""
        # Start with default colors
        colors = np.tile(self.default_color, (self.mesh.n_points, 1))
        
        # Apply region colors
        for region_name, point_indices in self.painted_regions.items():
            if region_name in self.region_colors:
                region_color = self.region_colors[region_name]
                for idx in point_indices:
                    colors[idx] = region_color
        
        # Update mesh
        self.mesh.point_data['colors'] = colors
        self.plotter.render()
    
    def clear_current_region(self):
        """Clear the current region."""
        if self.current_region in self.painted_regions:
            del self.painted_regions[self.current_region]
            self.update_mesh_colors()
            self.update_status_text()
            print(f"Cleared {self.current_region}")
    
    def clear_all_regions(self):
        """Clear all painted regions."""
        self.painted_regions.clear()
        self.update_mesh_colors()
        self.update_status_text()
        print("Cleared all regions")
    
    def save_painted_regions(self, filename: str = None):
        """Save painted regions to file."""
        if filename is None:
            filename = f"{self.stl_path.stem}_painted_regions.pkl"
        
        save_data = {
            'painted_regions': {name: list(points) for name, points in self.painted_regions.items()},
            'stl_path': str(self.stl_path),
            'scale_factor': self.scale_factor,
            'brush_size': self.brush_size,
            'region_colors': self.region_colors
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved painted regions to: {filename}")
    
    def load_painted_regions(self, filename: str = None):
        """Load painted regions from file."""
        if filename is None:
            filename = f"{self.stl_path.stem}_painted_regions.pkl"
        
        if not Path(filename).exists():
            print(f"No saved regions found: {filename}")
            return
        
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
            
            # Convert back to sets
            self.painted_regions = {
                name: set(points) for name, points in save_data['painted_regions'].items()
            }
            
            self.update_mesh_colors()
            self.update_status_text()
            print(f"Loaded painted regions from: {filename}")
            
        except Exception as e:
            print(f"Error loading painted regions: {e}")
    
    def export_coordinates(self, filename: str = None):
        """Export painted region coordinates for CFD mapping."""
        if filename is None:
            filename = f"{self.stl_path.stem}_painted_coordinates.json"
        
        if not self.painted_regions:
            print("No painted regions to export!")
            return
        
        try:
            export_data = {
                'stl_file': str(self.stl_path),
                'scale_factor': float(self.scale_factor),
                'coordinate_units': 'meters' if self.scale_factor == 0.001 else 'original',
                'regions': {}
            }
            
            for region_name, point_indices in self.painted_regions.items():
                if len(point_indices) == 0:
                    continue
                    
                # Convert set to sorted list for consistent output
                indices_list = sorted(list(point_indices))
                
                # Get coordinates for painted points
                coordinates = self.mesh.points[indices_list]
                
                export_data['regions'][region_name] = {
                    'point_count': int(len(indices_list)),
                    'point_indices': [int(idx) for idx in indices_list],
                    'coordinates': [[float(x) for x in coord] for coord in coordinates.tolist()],
                    'centroid': [float(x) for x in coordinates.mean(axis=0).tolist()],
                    'bounds': {
                        'x_min': float(coordinates[:, 0].min()),
                        'x_max': float(coordinates[:, 0].max()),
                        'y_min': float(coordinates[:, 1].min()),
                        'y_max': float(coordinates[:, 1].max()),
                        'z_min': float(coordinates[:, 2].min()),
                        'z_max': float(coordinates[:, 2].max())
                    },
                    'color': list(self.region_colors.get(region_name, [1.0, 0.0, 0.0]))
                }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Exported coordinates to: {filename}")
            
            # Print summary
            total_points = sum(len(indices) for indices in self.painted_regions.values())
            print(f"📊 Exported {len(self.painted_regions)} regions with {total_points:,} total points")
            
        except Exception as e:
            print(f"❌ Error exporting coordinates: {e}")
            import traceback
            traceback.print_exc()
    
    def show(self):
        """Show the interactive painter."""
        self.setup_plotter()
        print("\nStarting STL Surface Painter...")
        print("Click and drag to paint when painting mode is ON (press 'p')")
        print("Press 's' to save or 'e' to export coordinates")
        
        try:
            self.plotter.show()
        finally:
            # Auto-save when closing
            if self.painted_regions:
                print("\n🔄 Auto-saving painted regions...")
                try:
                    self.save_painted_regions()
                    self.export_coordinates()
                    print("✅ Painted regions saved successfully!")
                except Exception as e:
                    print(f"❌ Error during auto-save: {e}")
            else:
                print("\n📝 No painted regions to save.")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Interactive STL Surface Painter for CFD Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Controls:
  • Click and drag to paint (when painting mode is ON)
  • 'p' - Toggle painting mode ON/OFF
  • '1-5' - Switch between regions (different colors)
  • 'c' - Clear current region
  • 'C' - Clear all regions
  • 's' - Save painted regions to file
  • 'l' - Load painted regions from file
  • 'e' - Export coordinates for CFD mapping
  • 'q' - Quit

Usage Examples:
  # Paint STL surface with automatic scaling
  python -m src.surface_painting.stl_painter --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl
  
  # Paint with custom brush size
  python -m src.surface_painting.stl_painter --stl surface.stl --brush-size 2.0
        """
    )
    
    parser.add_argument('--stl', required=True, type=Path,
                        help='Path to STL file')
    
    parser.add_argument('--scale-factor', type=float, default=0.001,
                        help='Scale factor for STL coordinates (default: 0.001 for mm to m)')
    
    parser.add_argument('--brush-size', type=float, default=5.0,
                        help='Brush size in millimeters (default: 5.0)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.stl.exists():
        print(f"Error: STL file not found: {args.stl}")
        return
    
    # Create and show painter
    try:
        painter = STLPainter(args.stl, args.scale_factor)
        painter.brush_size = args.brush_size
        painter.show()
        
    except ImportError as e:
        print(f"Error: Missing required dependency - {e}")
        print("Please install PyVista: pip install pyvista")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 