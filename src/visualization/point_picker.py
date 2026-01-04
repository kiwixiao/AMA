"""
PyVista-based Point Picker for CFD Surface Data

This module provides a fast, interactive point selection tool using PyVista/VTK.
Unlike Plotly's web-based visualization, PyVista uses native VTK picking which
handles millions of points efficiently.

Usage:
    python src/main.py --point-picker --subject SUBJECT_NAME

Or standalone:
    python -m src.visualization.point_picker --subject SUBJECT_NAME
"""

import numpy as np
import json
from pathlib import Path
import pyvista as pv
import h5py
from typing import Optional, List, Dict, Tuple


class PointPicker:
    """Interactive point picker using PyVista for fast selection on large meshes."""

    def __init__(self, subject_name: str, results_dir: Optional[Path] = None):
        """
        Initialize the point picker.

        Args:
            subject_name: Name of the subject (e.g., '2mmeshOSAMRI007')
            results_dir: Optional path to results directory
        """
        self.subject_name = subject_name
        self.results_dir = results_dir or Path(f"{subject_name}_results")
        self.h5_path = self.results_dir / f"{subject_name}_cfd_data.h5"
        self.tracking_json_path = self.results_dir / f"{subject_name}_tracking_locations.json"

        # Storage for picked points
        self.picked_points: List[Dict] = []
        self.current_data = None
        self.plotter = None
        self.cloud = None
        self.data_scale = 1.0  # Will be updated when data is loaded
        self._last_pick_time = 0  # For debouncing

    def load_available_timesteps(self) -> List[Tuple[int, float]]:
        """
        Load available timesteps from HDF5 file.

        Returns:
            List of (index, time_in_ms) tuples
        """
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        timesteps = []
        with h5py.File(self.h5_path, 'r') as f:
            if 'time_points' in f:
                # New format: time_points array with cfd_data[timestep, point, column]
                time_points = f['time_points'][:]
                for i, t in enumerate(time_points):
                    time_ms = int(round(t * 1000))  # Convert seconds to ms
                    timesteps.append((i, time_ms))
            else:
                # Old format: timestep_X groups
                for key in f.keys():
                    if key.startswith('timestep_'):
                        try:
                            ts = int(key.replace('timestep_', ''))
                            timesteps.append((ts, ts))
                        except ValueError:
                            continue

        return timesteps

    def load_timestep_data(self, timestep_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load point cloud data for a specific timestep.

        Args:
            timestep_idx: Index into the timestep array

        Returns:
            Tuple of (points, patch_numbers, face_indices)
        """
        with h5py.File(self.h5_path, 'r') as f:
            if 'cfd_data' in f:
                # New format: cfd_data[timestep, point, column]
                # Get column names from attributes
                col_names = list(f.attrs.get('column_names', []))

                # Find column indices
                x_idx = col_names.index('X (m)') if 'X (m)' in col_names else col_names.index('Position[X] (m)')
                y_idx = col_names.index('Y (m)') if 'Y (m)' in col_names else col_names.index('Position[Y] (m)')
                z_idx = col_names.index('Z (m)') if 'Z (m)' in col_names else col_names.index('Position[Z] (m)')
                patch_idx = col_names.index('Patch Number') if 'Patch Number' in col_names else None
                face_idx = col_names.index('Face Index') if 'Face Index' in col_names else None

                data = f['cfd_data'][timestep_idx]  # Shape: (points, columns)

                x = data[:, x_idx]
                y = data[:, y_idx]
                z = data[:, z_idx]
                points = np.column_stack([x, y, z])

                if patch_idx is not None:
                    patch_numbers = data[:, patch_idx].astype(int)
                else:
                    patch_numbers = np.zeros(len(x), dtype=int)

                if face_idx is not None:
                    face_indices = data[:, face_idx].astype(int)
                else:
                    face_indices = np.arange(len(x))
            else:
                # Old format: timestep_X groups
                grp = f[f'timestep_{timestep_idx}']
                x = grp['X'][:]
                y = grp['Y'][:]
                z = grp['Z'][:]
                points = np.column_stack([x, y, z])

                if 'Patch Number' in grp:
                    patch_numbers = grp['Patch Number'][:]
                else:
                    patch_numbers = np.zeros(len(x), dtype=int)

                face_indices = np.arange(len(x))

        return points, patch_numbers, face_indices

    def create_point_cloud(self, points: np.ndarray, patch_numbers: np.ndarray,
                          face_indices: np.ndarray) -> pv.PolyData:
        """Create PyVista point cloud with scalar data."""
        cloud = pv.PolyData(points)
        cloud['Patch Number'] = patch_numbers
        cloud['Face Index'] = face_indices

        # Store bounds for marker sizing
        bounds = cloud.bounds
        self.data_scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

        return cloud

    def on_pick(self, picked_cells):
        """Callback when a cell/point is picked via cell picking."""
        if self.current_data is None:
            return

        if picked_cells is None or picked_cells.n_cells == 0:
            return

        points, patch_numbers, face_indices = self.current_data

        # Get the picked point - for point clouds, cell index == point index
        # Get the center of the picked cell (which is the point itself for vertices)
        picked_point = np.array(picked_cells.center)

        # Find the closest point in our data to get the correct index
        distances = np.linalg.norm(points - picked_point, axis=1)
        point_id = np.argmin(distances)

        # Check if we already picked this point
        for existing in self.picked_points:
            if existing['face_index'] == int(face_indices[point_id]):
                print(f"\n⚠️  Point already selected (Face Index: {face_indices[point_id]})")
                return

        picked_info = {
            'patch_number': int(patch_numbers[point_id]),
            'face_index': int(face_indices[point_id]),
            'coordinates': [float(points[point_id, 0]), float(points[point_id, 1]), float(points[point_id, 2])],
            'description': f"Location {len(self.picked_points) + 1} - UPDATE THIS DESCRIPTION"
        }

        self.picked_points.append(picked_info)

        print(f"\n{'='*50}")
        print(f"📍 Point {len(self.picked_points)} Selected:")
        print(f"   Patch Number: {picked_info['patch_number']}")
        print(f"   Face Index: {picked_info['face_index']}")
        print(f"   Coordinates: ({picked_info['coordinates'][0]:.6f}, "
              f"{picked_info['coordinates'][1]:.6f}, {picked_info['coordinates'][2]:.6f})")
        print(f"{'='*50}")

        # Add visible marker at picked point - just a single highlighted point
        if self.plotter is not None:
            # Create a single point marker at the exact location
            marker = pv.PolyData(points[point_id:point_id+1])
            self.plotter.add_mesh(
                marker,
                color='red',
                point_size=15,  # Slightly larger than cloud points (8)
                render_points_as_spheres=True,
                name=f'picked_{len(self.picked_points)}'
            )
            self.plotter.render()

    def _handle_pick(self, point_id: int, points: np.ndarray,
                     patch_numbers: np.ndarray, face_indices: np.ndarray,
                     selection_mode: str = 'click'):
        """Handle a picked point by ID.

        Args:
            selection_mode: 'click' for single point, 'brush' for brush mode
        """
        import time
        # Debounce - ignore picks within 0.5 seconds of each other
        current_time = time.time()
        if current_time - self._last_pick_time < 0.5:
            return
        self._last_pick_time = current_time

        # Check if we already picked this point
        for existing in self.picked_points:
            if existing['face_index'] == int(face_indices[point_id]):
                print(f"\n⚠️  Point already selected (Face Index: {face_indices[point_id]})")
                return

        picked_info = {
            'patch_number': int(patch_numbers[point_id]),
            'face_index': int(face_indices[point_id]),
            'coordinates': [float(points[point_id, 0]), float(points[point_id, 1]), float(points[point_id, 2])],
            'description': '',  # Will be set by dialog or left empty
            'selection_mode': selection_mode  # 'click' or 'brush'
        }

        self.picked_points.append(picked_info)
        point_num = len(self.picked_points)

        print(f"\n{'='*50}")
        print(f"📍 Point {point_num} Selected:")
        print(f"   Patch Number: {picked_info['patch_number']}")
        print(f"   Face Index: {picked_info['face_index']}")
        print(f"   Coordinates: ({picked_info['coordinates'][0]:.6f}, "
              f"{picked_info['coordinates'][1]:.6f}, {picked_info['coordinates'][2]:.6f})")
        print(f"{'='*50}")

        # Note: Naming happens after pressing 'q' to avoid VTK event conflicts

        # Add visible marker at picked point - just a highlighted point
        if self.plotter is not None:
            marker = pv.PolyData(points[point_id:point_id+1])
            self.plotter.add_mesh(
                marker,
                color='red',
                point_size=15,  # Slightly larger than cloud points
                render_points_as_spheres=True,
                name=f'marker_{len(self.picked_points)}'
            )

            # Add text label at the point - deep blue, close to point
            label_pos = points[point_id] + np.array([0.0002, 0.0002, 0.0002])  # Very close
            self.plotter.add_point_labels(
                pv.PolyData(label_pos.reshape(1, 3)),
                [f"#{point_num}"],
                font_size=20,
                point_color='red',
                point_size=0,  # Hide the point, we already have marker
                text_color='darkblue',
                shape_opacity=0.85,
                shape='rounded_rect',
                fill_shape=True,
                name=f'label_{point_num}'
            )

            # Update on-screen picked points display
            self._update_picked_display()
            self.plotter.render()

    def _get_point_name_dialog(self, point_num: int, point_info: dict) -> str:
        """Get name for a picked point via console input.

        Args:
            point_num: The point number (1-based)
            point_info: Dictionary with patch_number, face_index, coordinates

        Returns:
            The name entered by user, or empty string if skipped
        """
        try:
            # Use console input - works during VTK event loop
            import sys
            if sys.stdin.isatty():
                name = input(f"   📝 Name for #{point_num} (Enter to skip): ").strip()
                return name
            else:
                return ""
        except (EOFError, OSError):
            return ""

    def _update_picked_display(self):
        """Update the on-screen display of picked points."""
        if self.plotter is None or len(self.picked_points) == 0:
            return

        # Count by selection mode
        click_count = sum(1 for pt in self.picked_points if pt.get('selection_mode') == 'click')
        brush_count = sum(1 for pt in self.picked_points if pt.get('selection_mode') == 'brush')

        # Build display text - show last 5 points to fit on screen with larger font
        display_points = self.picked_points[-5:]
        start_idx = max(0, len(self.picked_points) - 5)

        lines = [f"PICKED: {len(self.picked_points)} total"]
        lines.append(f"Click: {click_count} | Brush: {brush_count}")
        lines.append("-" * 28)

        for i, pt in enumerate(display_points):
            idx = start_idx + i + 1
            mode_icon = "[C]" if pt.get('selection_mode') == 'click' else "[B]"
            name = pt.get('description', '')
            if name:
                lines.append(f"{mode_icon} #{idx}: {name}")
            else:
                lines.append(f"{mode_icon} #{idx}: P{pt['patch_number']}, F{pt['face_index']}")

        lines.append("-" * 28)
        lines.append("Press 'q' to save & exit")

        display_text = "\n".join(lines)

        # Update or add the display text - font_size 16 (twice as big)
        self.plotter.add_text(
            display_text,
            position='lower_right',
            font_size=16,
            color='darkblue',
            name='picked_display',
            font='courier'
        )

    def on_pick_legacy(self, point: np.ndarray):
        """Legacy callback for point picking (fallback)."""
        if self.current_data is None:
            return

        points, patch_numbers, face_indices = self.current_data

        # Find closest point in data
        distances = np.linalg.norm(points - point, axis=1)
        idx = np.argmin(distances)

        picked_info = {
            'patch_number': int(patch_numbers[idx]),
            'face_index': int(face_indices[idx]),
            'coordinates': [float(points[idx, 0]), float(points[idx, 1]), float(points[idx, 2])],
            'description': f"Location {len(self.picked_points) + 1} - UPDATE THIS DESCRIPTION"
        }

        self.picked_points.append(picked_info)

        print(f"\n{'='*50}")
        print(f"📍 Point {len(self.picked_points)} Selected:")
        print(f"   Patch Number: {picked_info['patch_number']}")
        print(f"   Face Index: {picked_info['face_index']}")
        print(f"   Coordinates: ({picked_info['coordinates'][0]:.6f}, "
              f"{picked_info['coordinates'][1]:.6f}, {picked_info['coordinates'][2]:.6f})")
        print(f"{'='*50}")

        # Add marker at picked point
        if self.plotter is not None:
            marker_size = self.data_scale * 0.015
            sphere = pv.Sphere(radius=marker_size, center=points[idx])
            self.plotter.add_mesh(sphere, color='red', name=f'picked_{len(self.picked_points)}')

    def save_tracking_locations(self):
        """Save picked points to tracking locations JSON."""
        # Load existing JSON if it exists
        if self.tracking_json_path.exists():
            with open(self.tracking_json_path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                "locations": [],
                "combinations": [],
                "_instructions": {
                    "step1": "Points were selected using PyVista point picker",
                    "step2": "Update the description for each location",
                    "step3": "Run --plotting to generate analysis"
                }
            }

        # Convert picked points to location format
        new_locations = []
        for i, pt in enumerate(self.picked_points):
            location = {
                "description": pt['description'],
                "patch_number": pt['patch_number'],
                "face_indices": [pt['face_index']],
                "coordinates": pt['coordinates']
            }
            new_locations.append(location)

        # Replace or append locations
        if new_locations:
            data["locations"] = new_locations

        # Save
        with open(self.tracking_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Saved {len(new_locations)} locations to: {self.tracking_json_path}")

    def run(self, timestep_ms: Optional[int] = None):
        """
        Run the interactive point picker.

        Args:
            timestep_ms: Specific timestep in ms to load. If None, will prompt user.
        """
        print(f"\n{'='*60}")
        print(f"  PyVista Point Picker for {self.subject_name}")
        print(f"{'='*60}")

        # Load available timesteps - returns list of (index, time_ms) tuples
        print(f"\nLoading data from: {self.h5_path}")
        timesteps = self.load_available_timesteps()
        print(f"Found {len(timesteps)} timesteps")

        if len(timesteps) == 0:
            print("❌ No timesteps found in HDF5 file")
            return

        # Create lookup dict: time_ms -> index
        time_to_idx = {t_ms: idx for idx, t_ms in timesteps}
        available_times = [t_ms for _, t_ms in timesteps]

        # Select timestep
        timestep_idx = None
        selected_time_ms = None

        if timestep_ms is not None:
            # User provided specific timestep
            if timestep_ms in time_to_idx:
                timestep_idx = time_to_idx[timestep_ms]
                selected_time_ms = timestep_ms
            else:
                # Find closest
                closest = min(available_times, key=lambda x: abs(x - timestep_ms))
                print(f"Timestep {timestep_ms}ms not found, using closest: {closest}ms")
                timestep_idx = time_to_idx[closest]
                selected_time_ms = closest
        else:
            # Prompt user
            first_time = available_times[0]
            last_time = available_times[-1]
            print(f"\nAvailable timesteps: {first_time}ms to {last_time}ms")
            print(f"First 10 (ms): {available_times[:10]}")

            while True:
                try:
                    user_input = input(f"\nEnter timestep in ms (default: {first_time}): ").strip()
                    if not user_input:
                        selected_time_ms = first_time
                    else:
                        selected_time_ms = int(user_input)

                    if selected_time_ms not in time_to_idx:
                        # Find closest
                        closest = min(available_times, key=lambda x: abs(x - selected_time_ms))
                        print(f"Timestep {selected_time_ms}ms not found, closest is {closest}ms")
                        use_closest = input(f"Use {closest}ms? (y/n, default: y): ").strip().lower()
                        if use_closest != 'n':
                            selected_time_ms = closest
                        else:
                            continue

                    timestep_idx = time_to_idx[selected_time_ms]
                    break
                except ValueError:
                    print("Please enter a valid integer (time in milliseconds)")

        print(f"\nLoading timestep {selected_time_ms}ms (index {timestep_idx})...")
        points, patch_numbers, face_indices = self.load_timestep_data(timestep_idx)
        self.current_data = (points, patch_numbers, face_indices)

        print(f"Loaded {len(points):,} points with {len(np.unique(patch_numbers))} patches")

        # Create point cloud
        cloud = self.create_point_cloud(points, patch_numbers, face_indices)

        # Create plotter
        self.plotter = pv.Plotter(title=f"Point Picker - {self.subject_name} (t={selected_time_ms}ms)")

        # Add point cloud with larger points for better visibility
        # Using size 10 - good balance between visibility and not overlapping too much
        self.plotter.add_mesh(
            cloud,
            scalars='Patch Number',
            cmap='tab20',
            point_size=10,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            pickable=True,
            name='point_cloud'
        )

        # Store cloud reference and mesh actor for picking
        self.cloud = cloud

        # Create VTK point picker for accurate picking
        import vtk
        point_picker = vtk.vtkPointPicker()
        point_picker.SetTolerance(0.005)  # Picking tolerance

        # Brush mode state - activated by holding 'b' key
        brush_mode = {'active': False, 'points_this_stroke': set()}

        # Create brush indicator (2D circle that shows when brush mode is active)
        brush_indicator = None

        def pick_point_at(x, y, selection_mode='click'):
            """Pick a single point at screen coordinates (depth-aware)."""
            point_picker.Pick(x, y, 0, self.plotter.renderer)
            point_id = point_picker.GetPointId()

            if point_id >= 0 and point_id < len(points):
                # In brush mode, skip already picked points in this stroke
                if brush_mode['active'] and point_id in brush_mode['points_this_stroke']:
                    return
                brush_mode['points_this_stroke'].add(point_id)
                self._handle_pick(point_id, points, patch_numbers, face_indices, selection_mode)

        # Left-click for single point selection
        def on_left_click(obj, event):
            """Handle left click for single point picking."""
            if not brush_mode['active']:
                x, y = self.plotter.iren.interactor.GetEventPosition()
                brush_mode['points_this_stroke'].clear()
                pick_point_at(x, y, 'click')

        self.plotter.iren.interactor.AddObserver('LeftButtonPressEvent', on_left_click)

        # Brush mode: hold 'b' key + move mouse to paint
        def on_key_press(obj, event):
            """Handle key press for brush mode."""
            nonlocal brush_indicator
            key = self.plotter.iren.interactor.GetKeySym()
            if key == 'b' and not brush_mode['active']:
                brush_mode['active'] = True
                brush_mode['points_this_stroke'].clear()
                # Show brush mode indicator
                self.plotter.add_text(
                    "🎨 BRUSH MODE - Move mouse to paint",
                    position='lower_left',
                    font_size=14,
                    color='red',
                    name='brush_indicator'
                )
                # Change cursor to crosshair for visual feedback
                render_window = self.plotter.iren.interactor.GetRenderWindow()
                if render_window:
                    render_window.SetCurrentCursor(vtk.VTK_CURSOR_CROSSHAIR)
                print("\n🎨 Brush mode ON - move mouse over points to select")

        def on_key_release(obj, event):
            """Handle key release for brush mode."""
            key = self.plotter.iren.interactor.GetKeySym()
            if key == 'b' and brush_mode['active']:
                count = len(brush_mode['points_this_stroke'])
                print(f"🎨 Brush mode OFF - selected {count} new points")
                brush_mode['active'] = False
                brush_mode['points_this_stroke'].clear()
                # Remove brush mode indicator
                self.plotter.remove_actor('brush_indicator')
                # Restore default cursor
                render_window = self.plotter.iren.interactor.GetRenderWindow()
                if render_window:
                    render_window.SetCurrentCursor(vtk.VTK_CURSOR_DEFAULT)

        def on_mouse_move(obj, event):
            """Pick points while brush mode is active (depth-aware)."""
            if brush_mode['active']:
                x, y = self.plotter.iren.interactor.GetEventPosition()
                pick_point_at(x, y, 'brush')

        # Add observers for brush mode
        self.plotter.iren.interactor.AddObserver('KeyPressEvent', on_key_press)
        self.plotter.iren.interactor.AddObserver('KeyReleaseEvent', on_key_release)
        self.plotter.iren.interactor.AddObserver('MouseMoveEvent', on_mouse_move)

        # Add instructions text
        self.plotter.add_text(
            "Left-click: pick | Hold 'b': brush | Drag: rotate | 'q': SAVE & EXIT",
            position='upper_left',
            font_size=12,
            color='black',
            name='instructions'
        )

        # Add info panel
        self.plotter.add_text(
            f"Subject: {self.subject_name}\nTime: {selected_time_ms}ms\nPoints: {len(points):,}\nPatches: {len(np.unique(patch_numbers))}",
            position='upper_right',
            font_size=9,
            color='blue',
            name='info'
        )

        print("\n" + "="*60)
        print("  INSTRUCTIONS")
        print("="*60)
        print("  • Left-click to pick single point (depth-aware)")
        print("  • Hold 'b' + move mouse = BRUSH mode (paint multiple points)")
        print("  • Left-drag to rotate view")
        print("  • Scroll to zoom in/out")
        print("  • Shift + Left-drag to pan")
        print("  • Press 'q' to FINISH and AUTO-SAVE")
        print("="*60)
        print(f"\n📁 Will save to: {self.tracking_json_path}")

        # Show the plotter (blocks until closed)
        self.plotter.show()

        # After closing, name and save the picked points
        if self.picked_points:
            print(f"\n📊 You selected {len(self.picked_points)} points")
            print("\n" + "="*60)
            print("  NAME YOUR POINTS (press Enter to skip)")
            print("="*60)

            import sys
            for i, pt in enumerate(self.picked_points):
                mode = "[C]" if pt.get('selection_mode') == 'click' else "[B]"
                print(f"\n#{i+1} {mode} Patch {pt['patch_number']}, Face {pt['face_index']}")

                try:
                    if sys.stdin.isatty():
                        name = input(f"   Name for #{i+1}: ").strip()
                        pt['description'] = name
                    else:
                        pt['description'] = ""
                except (EOFError, OSError):
                    pt['description'] = ""

            # Summary
            named_count = sum(1 for pt in self.picked_points if pt.get('description'))
            print(f"\n📊 Summary: {named_count} named, {len(self.picked_points) - named_count} unnamed")

            self.save_tracking_locations()
            print(f"\n✅ Ready for Phase 2! Run:")
            print(f"   python src/main.py --subject {self.subject_name} --plotting")
        else:
            print("\n⚠️  No points were selected. Nothing saved.")


def run_point_picker(subject_name: str, timestep: Optional[int] = None,
                     results_dir: Optional[Path] = None):
    """
    Convenience function to run the point picker.

    Args:
        subject_name: Name of the subject
        timestep: Optional specific timestep
        results_dir: Optional results directory path
    """
    picker = PointPicker(subject_name, results_dir)
    picker.run(timestep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyVista Point Picker for CFD Data")
    parser.add_argument("--subject", required=True, help="Subject name (e.g., 2mmeshOSAMRI007)")
    parser.add_argument("--timestep", type=int, help="Specific timestep to load")
    parser.add_argument("--results-dir", help="Path to results directory")

    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else None
    run_point_picker(args.subject, args.timestep, results_dir)
