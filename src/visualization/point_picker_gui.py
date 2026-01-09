"""
PyVista Point Picker with Qt GUI

A proper GUI application with:
- Left panel: List of picked points with editable names
- Right panel: 3D PyVista view for point selection

Usage:
    python src/main.py --point-picker --subject SUBJECT_NAME
"""

import numpy as np
import json
from pathlib import Path
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QLineEdit, QPushButton, QLabel,
    QSplitter, QGroupBox, QMessageBox, QFrame
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
import h5py
from typing import Optional, List, Dict, Tuple
import sys


class PointPickerGUI(QMainWindow):
    """Main GUI window for point picking."""

    def __init__(self, subject_name: str, results_dir: Optional[Path] = None,
                 h5_path: Optional[Path] = None, use_light_h5: bool = False):
        super().__init__()
        self.subject_name = subject_name
        self.results_dir = results_dir or Path(f"{subject_name}_results")

        # Determine H5 file path
        if h5_path:
            self.h5_path = Path(h5_path)
        elif use_light_h5:
            # Try lightweight H5 first
            light_path = self.results_dir / f"{subject_name}_cfd_data_light.h5"
            if light_path.exists():
                self.h5_path = light_path
                print(f"📦 Using lightweight HDF5: {light_path}")
            else:
                self.h5_path = self.results_dir / f"{subject_name}_cfd_data.h5"
                print(f"⚠️  Lightweight H5 not found, using full: {self.h5_path}")
        else:
            self.h5_path = self.results_dir / f"{subject_name}_cfd_data.h5"

        # Use picked_points.json if it exists, otherwise fall back to tracking_locations.json
        picked_points_path = self.results_dir / f"{subject_name}_picked_points.json"
        if picked_points_path.exists():
            self.tracking_json_path = picked_points_path
            print(f"📄 Using picked_points.json format")
        else:
            self.tracking_json_path = self.results_dir / f"{subject_name}_tracking_locations.json"

        # Data storage
        self.picked_points: List[Dict] = []
        self.current_data = None
        self.points = None
        self.patch_numbers = None
        self.face_indices = None

        self.setWindowTitle(f"Point Picker - {subject_name}")
        self.setGeometry(100, 100, 1400, 800)

        self._setup_ui()
        self._setup_vtk_picker()

    def _setup_ui(self):
        """Set up the Qt user interface."""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === LEFT PANEL: Point list and controls ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Title
        title = QLabel(f"📍 Point Picker: {self.subject_name}")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        left_layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "• Left-click in 3D view to pick point\n"
            "• Hold 'b' + move mouse for brush mode\n"
            "• Double-click name to edit\n"
            "• Press Save when done"
        )
        instructions.setStyleSheet("color: #666; padding: 5px;")
        left_layout.addWidget(instructions)

        # Picked points list
        list_group = QGroupBox("Picked Points")
        list_layout = QVBoxLayout(list_group)

        self.point_list = QListWidget()
        self.point_list.setFont(QFont("Courier", 11))
        self.point_list.itemDoubleClicked.connect(self._edit_point_name)
        list_layout.addWidget(self.point_list)

        # Count label
        self.count_label = QLabel("0 points selected")
        list_layout.addWidget(self.count_label)

        left_layout.addWidget(list_group)

        # Name input section
        name_group = QGroupBox("Name Selected Point")
        name_layout = QVBoxLayout(name_group)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name (e.g., 'larynx', 'soft palate')")
        self.name_input.returnPressed.connect(self._apply_name)
        name_layout.addWidget(self.name_input)

        apply_btn = QPushButton("Apply Name")
        apply_btn.clicked.connect(self._apply_name)
        name_layout.addWidget(apply_btn)

        left_layout.addWidget(name_group)

        # Buttons row 1
        btn_layout1 = QHBoxLayout()

        undo_btn = QPushButton("↩ Undo Last")
        undo_btn.setStyleSheet("background-color: #FF9800; color: white;")
        undo_btn.clicked.connect(self._undo_last)
        btn_layout1.addWidget(undo_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        btn_layout1.addWidget(clear_btn)

        left_layout.addLayout(btn_layout1)

        # Buttons row 2
        btn_layout2 = QHBoxLayout()

        save_btn = QPushButton("💾 Save && Exit")
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        save_btn.clicked.connect(self._save_and_exit)
        btn_layout2.addWidget(save_btn)

        left_layout.addLayout(btn_layout2)

        # Status
        self.status_label = QLabel(f"Will save to: {self.tracking_json_path}")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left_panel)

        # === RIGHT PANEL: PyVista 3D view ===
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # PyVista Qt interactor
        self.plotter = QtInteractor(right_panel)
        right_layout.addWidget(self.plotter.interactor)

        splitter.addWidget(right_panel)

        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([350, 1050])

    def _setup_vtk_picker(self):
        """Set up VTK point picking."""
        import vtk
        self.vtk_picker = vtk.vtkPointPicker()
        self.vtk_picker.SetTolerance(0.005)

        # Brush mode state
        self.brush_mode = {'active': False, 'points_this_stroke': set()}

        # Add observers
        self.plotter.interactor.AddObserver('LeftButtonPressEvent', self._on_left_click)
        self.plotter.interactor.AddObserver('KeyPressEvent', self._on_key_press)
        self.plotter.interactor.AddObserver('KeyReleaseEvent', self._on_key_release)
        self.plotter.interactor.AddObserver('MouseMoveEvent', self._on_mouse_move)

    def load_timestep_data(self, timestep_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load point cloud data for a specific timestep."""
        with h5py.File(self.h5_path, 'r') as f:
            if 'cfd_data' in f:
                # WSL-compatible column name decoding (handles bytes vs strings)
                raw_col_names = f.attrs.get('column_names', [])
                col_names = [
                    (c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else str(c)).strip('\x00')
                    for c in raw_col_names
                ]

                x_idx = col_names.index('X (m)') if 'X (m)' in col_names else col_names.index('Position[X] (m)')
                y_idx = col_names.index('Y (m)') if 'Y (m)' in col_names else col_names.index('Position[Y] (m)')
                z_idx = col_names.index('Z (m)') if 'Z (m)' in col_names else col_names.index('Position[Z] (m)')
                patch_idx = col_names.index('Patch Number') if 'Patch Number' in col_names else None
                face_idx = col_names.index('Face Index') if 'Face Index' in col_names else None

                # Handle variable point counts per timestep
                if 'point_counts' in f:
                    n = int(f['point_counts'][timestep_idx])
                    data = f['cfd_data'][timestep_idx, :n, :]
                else:
                    data = f['cfd_data'][timestep_idx]

                x, y, z = data[:, x_idx], data[:, y_idx], data[:, z_idx]
                points = np.column_stack([x, y, z])

                patch_numbers = data[:, patch_idx].astype(int) if patch_idx else np.zeros(len(x), dtype=int)
                face_indices = data[:, face_idx].astype(int) if face_idx else np.arange(len(x))
            else:
                grp = f[f'timestep_{timestep_idx}']
                points = np.column_stack([grp['X'][:], grp['Y'][:], grp['Z'][:]])
                patch_numbers = grp.get('Patch Number', np.zeros(len(points)))
                face_indices = np.arange(len(points))

        return points, patch_numbers, face_indices

    def load_available_timesteps(self) -> List[Tuple[int, float]]:
        """Load available timesteps from HDF5."""
        timesteps = []
        with h5py.File(self.h5_path, 'r') as f:
            if 'time_points' in f:
                time_points = f['time_points'][:]
                for i, t in enumerate(time_points):
                    timesteps.append((i, int(round(t * 1000))))
            else:
                for key in f.keys():
                    if key.startswith('timestep_'):
                        ts = int(key.replace('timestep_', ''))
                        timesteps.append((ts, ts))
        return timesteps

    def load_and_display(self, timestep_idx: int, time_ms: int):
        """Load data and display in 3D view."""
        self.points, self.patch_numbers, self.face_indices = self.load_timestep_data(timestep_idx)
        self.current_data = (self.points, self.patch_numbers, self.face_indices)

        # Create point cloud
        cloud = pv.PolyData(self.points)
        cloud['Patch Number'] = self.patch_numbers

        # Display
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

        # Add info text
        self.plotter.add_text(
            f"Time: {time_ms}ms | Points: {len(self.points):,}",
            position='upper_right',
            font_size=10,
            color='black'
        )

        self.plotter.reset_camera()
        self.status_label.setText(f"Loaded {len(self.points):,} points at t={time_ms}ms")

    def _pick_point_at(self, x, y, selection_mode='click'):
        """Pick a point at screen coordinates."""
        if self.points is None:
            return

        self.vtk_picker.Pick(x, y, 0, self.plotter.renderer)
        point_id = self.vtk_picker.GetPointId()

        if 0 <= point_id < len(self.points):
            # Skip duplicates in brush mode
            if self.brush_mode['active'] and point_id in self.brush_mode['points_this_stroke']:
                return
            self.brush_mode['points_this_stroke'].add(point_id)

            # Check if already picked
            for existing in self.picked_points:
                if existing['face_index'] == int(self.face_indices[point_id]):
                    return

            # Add point
            picked_info = {
                'patch_number': int(self.patch_numbers[point_id]),
                'face_index': int(self.face_indices[point_id]),
                'coordinates': self.points[point_id].tolist(),
                'description': '',
                'selection_mode': selection_mode,
                'point_id': point_id
            }
            self.picked_points.append(picked_info)

            # Save camera position before adding actors
            camera_pos = self.plotter.camera_position

            # Add marker
            marker = pv.PolyData(self.points[point_id:point_id+1])
            self.plotter.add_mesh(
                marker, color='red', point_size=15,
                render_points_as_spheres=True,
                reset_camera=False,
                name=f'marker_{len(self.picked_points)}'
            )

            # Add label
            point_num = len(self.picked_points)
            label_pos = self.points[point_id] + np.array([0.0002, 0.0002, 0.0002])
            self.plotter.add_point_labels(
                pv.PolyData(label_pos.reshape(1, 3)),
                [f"#{point_num}"],
                font_size=20,
                point_size=0,
                text_color='darkblue',
                shape_opacity=0.85,
                shape='rounded_rect',
                fill_shape=True,
                reset_camera=False,
                name=f'label_{point_num}'
            )

            # Restore camera position
            self.plotter.camera_position = camera_pos

            self._update_list()

    def _on_left_click(self, obj, event):
        """Handle left click for point picking."""
        if not self.brush_mode['active']:
            x, y = self.plotter.interactor.GetEventPosition()
            self.brush_mode['points_this_stroke'].clear()
            self._pick_point_at(x, y, 'click')

    def _on_key_press(self, obj, event):
        """Handle key press for brush mode."""
        import vtk
        key = self.plotter.interactor.GetKeySym()
        if key == 'b' and not self.brush_mode['active']:
            self.brush_mode['active'] = True
            self.brush_mode['points_this_stroke'].clear()
            self.status_label.setText("🎨 BRUSH MODE - Move mouse to paint")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            render_window = self.plotter.interactor.GetRenderWindow()
            if render_window:
                render_window.SetCurrentCursor(vtk.VTK_CURSOR_CROSSHAIR)

    def _on_key_release(self, obj, event):
        """Handle key release for brush mode."""
        import vtk
        key = self.plotter.interactor.GetKeySym()
        if key == 'b' and self.brush_mode['active']:
            count = len(self.brush_mode['points_this_stroke'])
            self.brush_mode['active'] = False
            self.brush_mode['points_this_stroke'].clear()
            self.status_label.setText(f"Brush: selected {count} points")
            self.status_label.setStyleSheet("color: #888; font-size: 10px;")
            render_window = self.plotter.interactor.GetRenderWindow()
            if render_window:
                render_window.SetCurrentCursor(vtk.VTK_CURSOR_DEFAULT)

    def _on_mouse_move(self, obj, event):
        """Pick points while brush mode is active."""
        if self.brush_mode['active']:
            x, y = self.plotter.interactor.GetEventPosition()
            self._pick_point_at(x, y, 'brush')

    def _update_list(self):
        """Update the point list widget."""
        self.point_list.clear()
        for i, pt in enumerate(self.picked_points):
            mode = "[C]" if pt.get('selection_mode') == 'click' else "[B]"
            name = pt.get('description', '')
            if name:
                text = f"#{i+1} {mode} {name}"
            else:
                text = f"#{i+1} {mode} P{pt['patch_number']}, F{pt['face_index']}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)  # Store index
            self.point_list.addItem(item)

        self.count_label.setText(f"{len(self.picked_points)} points selected")

    def _edit_point_name(self, item):
        """Handle double-click to edit point name."""
        idx = item.data(Qt.UserRole)
        if idx is not None and idx < len(self.picked_points):
            current_name = self.picked_points[idx].get('description', '')
            self.name_input.setText(current_name)
            self.name_input.setFocus()
            self.point_list.setCurrentItem(item)

    def _apply_name(self):
        """Apply name to selected point."""
        current_item = self.point_list.currentItem()
        if current_item:
            idx = current_item.data(Qt.UserRole)
            if idx is not None and idx < len(self.picked_points):
                name = self.name_input.text().strip()
                self.picked_points[idx]['description'] = name
                self._update_list()
                self.name_input.clear()

                # Update 3D label (preserve camera position)
                point_num = idx + 1
                pt = self.picked_points[idx]
                label_text = f"#{point_num}: {name}" if name else f"#{point_num}"
                label_pos = np.array(pt['coordinates']) + np.array([0.0002, 0.0002, 0.0002])

                # Save camera position
                camera_pos = self.plotter.camera_position

                # Remove old label and add new
                self.plotter.remove_actor(f'label_{point_num}')
                self.plotter.add_point_labels(
                    pv.PolyData(label_pos.reshape(1, 3)),
                    [label_text],
                    font_size=20,
                    point_size=0,
                    text_color='darkblue',
                    shape_opacity=0.85,
                    shape='rounded_rect',
                    fill_shape=True,
                    reset_camera=False,
                    name=f'label_{point_num}'
                )

                # Restore camera position
                self.plotter.camera_position = camera_pos

    def _undo_last(self):
        """Undo the last picked point."""
        if not self.picked_points:
            self.status_label.setText("Nothing to undo")
            return

        # Save camera position
        camera_pos = self.plotter.camera_position

        # Remove last point
        point_num = len(self.picked_points)
        self.plotter.remove_actor(f'marker_{point_num}')
        self.plotter.remove_actor(f'label_{point_num}')

        removed = self.picked_points.pop()
        self._update_list()

        # Restore camera position
        self.plotter.camera_position = camera_pos

        self.status_label.setText(f"Undid point #{point_num} (P{removed['patch_number']}, F{removed['face_index']})")

    def _clear_all(self):
        """Clear all picked points."""
        reply = QMessageBox.question(
            self, 'Clear All',
            'Clear all picked points?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Save camera position
            camera_pos = self.plotter.camera_position

            # Remove markers and labels
            for i in range(len(self.picked_points)):
                self.plotter.remove_actor(f'marker_{i+1}')
                self.plotter.remove_actor(f'label_{i+1}')
            self.picked_points.clear()
            self._update_list()

            # Restore camera position
            self.plotter.camera_position = camera_pos

    def _save_and_exit(self):
        """Save points to JSON and exit."""
        if not self.picked_points:
            QMessageBox.warning(self, 'No Points', 'No points to save!')
            return

        # Load existing JSON to preserve remesh_info and other metadata
        if self.tracking_json_path.exists():
            with open(self.tracking_json_path, 'r') as f:
                data = json.load(f)
            # Clear locations but preserve everything else (remesh_info, combinations, etc.)
            data["locations"] = []
        else:
            data = {
                "locations": [],
                "combinations": [],
                "_instructions": {
                    "step1": "Points selected using PyVista point picker",
                    "step2": "Run --plotting to generate analysis"
                }
            }

        for pt in self.picked_points:
            location = {
                "description": pt.get('description', ''),
                "patch_number": pt['patch_number'],
                "face_indices": [pt['face_index']],
                "coordinates": pt['coordinates']
            }
            data["locations"].append(location)

        # Save
        with open(self.tracking_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Show summary
        named = sum(1 for pt in self.picked_points if pt.get('description'))
        QMessageBox.information(
            self, 'Saved!',
            f"Saved {len(self.picked_points)} points ({named} named)\n\n"
            f"File: {self.tracking_json_path}\n\n"
            f"Ready for Phase 2:\n"
            f"python src/main.py --subject {self.subject_name} --plotting"
        )

        self.close()


def run_point_picker_gui(subject_name: str, timestep: Optional[int] = None,
                         results_dir: Optional[Path] = None,
                         use_light_h5: bool = False, h5_path: Optional[Path] = None):
    """Run the Qt-based point picker GUI."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create main window
    window = PointPickerGUI(subject_name, results_dir, h5_path=h5_path, use_light_h5=use_light_h5)

    # Load timesteps
    timesteps = window.load_available_timesteps()
    if not timesteps:
        print("❌ No timesteps found in HDF5 file")
        return

    # Select timestep
    time_to_idx = {t_ms: idx for idx, t_ms in timesteps}
    available_times = [t_ms for _, t_ms in timesteps]

    if timestep is not None and timestep in time_to_idx:
        timestep_idx = time_to_idx[timestep]
        time_ms = timestep
    else:
        # Use first available
        timestep_idx, time_ms = timesteps[0]
        if timestep is not None:
            closest = min(available_times, key=lambda x: abs(x - timestep))
            timestep_idx = time_to_idx[closest]
            time_ms = closest

    # Load and display
    window.load_and_display(timestep_idx, time_ms)
    window.show()

    print(f"\n{'='*60}")
    print(f"  Qt Point Picker GUI - {subject_name}")
    print(f"{'='*60}")
    print(f"  Loaded {time_ms}ms timestep")
    print(f"  Close window or click 'Save & Exit' when done")
    print(f"{'='*60}")

    app.exec_()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--timestep", type=int)
    parser.add_argument("--results-dir")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else None
    run_point_picker_gui(args.subject, args.timestep, results_dir)
