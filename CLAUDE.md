# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CFD (Computational Fluid Dynamics) analysis pipeline for respiratory airway flow visualization. Processes time-series CFD simulation data to analyze airflow patterns, pressure dynamics, and velocity/acceleration relationships in respiratory tract anatomy.

## Installation

```bash
# Clone and install as a package
git clone <repository-url>
cd newMetrics
pip install -e .

# Verify
ama --listsubjects
```

For development:
```bash
pip install -e ".[dev]"
```

## Three-Phase Workflow

```
Phase 1: Prepare     →  CSV files in       →  HDF5 + templates out
Phase 2: Pick Points  →  HDF5 in           →  picked_points.json out
  (manual edits)      →  fix configs if needed
Phase 3: Analyze      →  HDF5 + configs in  →  PDFs + HTMLs out
```

### Phase 1: Prepare (`--prepare`)
Convert raw CSV data to HDF5, detect breathing cycle and remesh, create templates.
```bash
ama --subject OSAMRI007 --prepare --flow-profile OSAMRI007FlowProfile_smoothed.csv
```
Outputs in `{SUBJECT}_results/`:
- `{SUBJECT}_cfd_data.h5` — full HDF5 (all timesteps, flow profile embedded)
- `{SUBJECT}_cfd_data_light.h5` — single-timestep HDF5 (portable, for point picking)
- `{SUBJECT}_metadata.json` — system metadata (remesh info, breathing cycle)
- `{SUBJECT}_picked_points.json` — empty template for landmark selection
- `interactive/` — HTML visualization for browsing the surface

### Phase 2: Pick Points (`--point-picker`)
Select anatomical landmarks using PyVista GUI.
```bash
ama --subject OSAMRI007 --point-picker --h5-file OSAMRI007_results/OSAMRI007_cfd_data.h5
```
Saves selections to `{SUBJECT}_results/{SUBJECT}_picked_points.json`.

**Between Phase 2 and 3**, you can manually edit:
- `picked_points.json` — fix coordinates, add/remove landmarks, correct patch/face values
- `metadata.json` — override `manual_remesh_timesteps_ms` if auto-detection was wrong

### Phase 3: Analyze (`--plotting`)
Generate all tracking data, PDF reports, and HTML visualizations.
```bash
ama --subject OSAMRI007 --plotting
```
No `--flow-profile` needed — reads from HDF5. Only needed as fallback for older HDF5 files.

### All-in-One (`--all`)
Run prepare + analyze in one pass (skips point picking, requires pre-configured tracking locations).
```bash
ama --subject OSAMRI007 --all --flow-profile OSAMRI007FlowProfile_smoothed.csv
```

### Other Commands
```bash
ama --subject OSAMRI007 --prepare --flow-profile profile.csv \
    --inhale-start 0.05 --transition 1.0 --exhale-end 2.2     # Manual breathing cycle
ama --subject OSAMRI007 --forcererun                  # Force rerun
ama --subject OSAMRI007 --patchradii 1.0 3.0 7.0 --normalangle 45.0
ama --subject OSAMRI007 --rawdir qDNS_xyz_tables      # Custom raw dir
ama --subject OSAMRI007 --xyz-path /data/xyz          # Custom XYZ path
ama --subject OSAMRI007 --disablevisualization        # Skip HTML output
ama --subject OSAMRI007 --disablepatchanalysis        # Single-point only
ama --subject OSAMRI007 --highlight-patches --patch-timestep 100
ama --subject OSAMRI007 --raw-surface --surface-timestep 50
ama --subject OSAMRI007 --has-remesh --remesh-before file1.csv --remesh-after file2.csv
ama --listsubjects
```

### Testing
```bash
python test_xyz_formats.py           # XYZ format handling
ama --listsubjects          # Basic functionality
```

## Architecture

### Monolithic Orchestrator Pattern

`src/main.py` (~5900 lines) is the monolithic entry point that owns the entire pipeline. It directly contains:
- All argparse CLI definitions
- Surface normal calculation (`calculate_surface_normal` using PCA via `NearestNeighbors`)
- Connected point finding with DBSCAN clustering and normal filtering
- Point-in-circle filtering (radius + patch + normal constraints)
- All plotting orchestration for PDFs and HTML output
- Breathing cycle detection integration
- Remesh handling (auto-detection + manual override)

### Module Dependency Graph

```
src/main.py (orchestrator)
├── src/utils/file_processing.py       ← File I/O, CSV loading, breathing cycle detection,
│                                        flow profile search, subject name extraction
├── src/utils/parallel_csv_processing.py ← ProcessPoolExecutor-based parallel tracking
│                                        with memory-aware auto-tuning (psutil)
├── src/utils/signal_processing.py     ← Zero-crossing detection, smart label positioning
│                                        (shared by main.py and cfd_analysis_3x3.py)
├── src/data_processing/trajectory.py  ← HDF5 creation/reading, CSV→HDF5 conversion,
│                                        dimension scanning for remesh detection
├── src/analysis/patch_region_analysis.py ← Enhanced circle finding, patch statistics,
│                                          multi-radius patch analysis
├── src/visualization/
│   ├── surface_plots.py               ← Plotly 3D interactive HTML generation
│   ├── cfd_analysis_3x3.py            ← Multi-page 3x3 panel PDF reports
│   ├── patch_visualization.py         ← Patch region highlighting
│   ├── interactive_selection.py       ← Dash-based browser point selector
│   ├── point_picker.py               ← PyVista-based native 3D point picker
│   └── point_picker_gui.py           ← GUI wrapper for point picker
```

### Data Flow

```
Phase 1 (--prepare):
  {SUBJECT}_xyz_tables/*.csv  ──→  parallel_csv_processing  ──→  {SUBJECT}_results/{SUBJECT}_cfd_data.h5
  FlowProfile_smoothed.csv    ──→  file_processing           ──→  breathing cycle bounds
  HDF5 + cycle bounds         ──→  surface_plots              ──→  interactive HTML + template JSON

Phase 2 (--point-picker):
  {SUBJECT}_cfd_data.h5       ──→  PyVista GUI               ──→  picked_points.json

Phase 3 (--plotting):
  {SUBJECT}_cfd_data.h5                    ──→  parallel tracking (HDF5-based)  ──→  tracked_points/*.csv
  {SUBJECT}_picked_points.json             ──→  point/patch region extraction
  tracked_points/*.csv + flow profile      ──→  cfd_analysis_3x3               ──→  PDF reports
  tracked_points/*.csv + HDF5              ──→  surface_plots                   ──→  interactive HTML
```

### Key Processing Concepts

**Patch/Face Model**: CFD mesh data uses `Patch Number` to identify mesh boundary regions and `Face Index` for individual faces within a patch. Tracking locations are defined by (patch_number, face_indices, coordinates).

**Surface Normal Filtering**: Points within a radius are filtered using PCA-based normal calculation + DBSCAN connectivity clustering. This ensures tracked regions stay on the same anatomical surface rather than bleeding through thin walls. Controlled by `--normalangle` (default 60°).

**Parallel Processing Strategy**: `parallel_csv_processing.py` auto-selects between multiple HDF5 tracking methods based on dataset size and available memory. Uses `ProcessPoolExecutor` with `psutil`-based process count tuning. Methods include: memory-safe multicore, optimized multicore, and optimized index lookup.

**Remesh Handling**: Simulations may change mesh topology mid-run. Auto-detected by scanning CSV file dimensions for point count changes. Can be manually overridden via `manual_remesh_timesteps_ms` in `metadata.json`, or with CLI flags `--has-remesh` / `--remesh-before` / `--remesh-after`.

**Mesh Variant Subject Names**: Subject names may have mesh resolution prefixes (e.g., `2mmeshOSAMRI007`, `less1mmesh_OSAMRI007`). `extract_base_subject()` in `file_processing.py` strips these to find the base subject name for flow profile lookup.

## Data Structure

### Input Files
- `{SUBJECT}_xyz_tables/` — Raw CFD CSV files (one per timestep)
- `{SUBJECT}FlowProfile_smoothed.csv` — Breathing flow rate data (or base subject's)

### Output Structure
All results are self-contained in `{SUBJECT}_results/`:
- `{SUBJECT}_cfd_data.h5` — HDF5 cache (85% size reduction from CSV)
- `{SUBJECT}_cfd_data_light.h5` — Single-timestep HDF5 for portable point picking
- `{SUBJECT}_metadata.json` — System metadata (remesh, breathing cycle, do not edit unless overriding remesh)
- `{SUBJECT}_picked_points.json` — Landmark selections (editable)
- `tracked_points/` — Per-location CSV trajectory data
- `reports/` — PDF analysis reports (3x3 panels in both normalized and original time)
- `figures/` — PNG images
- `interactive/` — HTML 3D visualizations

### XYZ File Naming
Two formats supported:
- Integer (milliseconds): `XYZ_Internal_Table_table_2387.csv`
- Scientific notation (seconds): `XYZ_Internal_Table_table_2.300000e+00.csv`

## Common Issues

**HDF5 Lock Errors**: Remove `{SUBJECT}_results/{SUBJECT}_cfd_data.h5` and rerun with `--forcererun`.

**Memory Issues**: Set `export OMP_NUM_THREADS=4` to limit parallel processes.

**macOS sklearn threadpoolctl**: Known `'NoneType' object has no attribute 'split'` error from sklearn on macOS + conda. Pipeline handles this by falling back to default surface normals. Does not occur on Linux.

## Development Notes

- `src/main.py` is the entry point, invoked via the `ama` CLI command (installed via `pip install .`)
- Optional imports (`SURFACE_PLOTS_AVAILABLE`, `VISUALIZATION_AVAILABLE`, `CFD_ANALYSIS_AVAILABLE`) use try/except with fallback stubs so the pipeline degrades gracefully
- All coordinates are in meters (column names: `X (m)`, `Y (m)`, `Z (m)`)
- Area vectors use columns: `Area[i] (m^2)`, `Area[j] (m^2)`, `Area[k] (m^2)`
- Timestamps in filenames may be milliseconds (int) or seconds (float) — `extract_timestep_from_filename()` handles both
- Follow existing patterns for imports and error handling
- Use `parallel_csv_processing.py` utilities for any new data processing
- Maintain chronological file ordering when processing time-series data
- Respect breathing cycle boundaries when filtering data
