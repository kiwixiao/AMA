# AMA - Airway Metrics Analysis

A CFD analysis pipeline for studying airway dynamics during breathing cycles. Processes large-scale CFD simulation data to track anatomical landmarks, analyze pressure-velocity-acceleration relationships, and generate visualizations.

## Installation

### Quick Setup (recommended)

```bash
git clone https://github.com/kiwixiao/AMA.git
cd AMA
bash setup.sh
```

The setup script auto-detects your environment:
- **Conda available**: creates `ama` conda env from `environment.yml`, installs package
- **No conda**: creates `.venv` virtual environment, installs package with GUI deps

After setup, activate your environment and use the `ama` command:

```bash
# Conda
conda activate ama

# Or pip venv
source .venv/bin/activate

# Verify
ama --listsubjects
```

### Developer Setup

If you plan to modify source code:

```bash
bash setup.sh --dev
```

This uses editable install (`pip install -e .`) so changes to source files take effect immediately without reinstalling.

### Manual Install

```bash
pip install ".[gui]"    # with PyVista GUI support
pip install .           # core only (no point picker GUI)
```

## 3-Phase Workflow

### Phase 1: Prepare

Convert raw CSV data to HDF5, detect breathing cycle, auto-detect remesh events.

```bash
ama --subject OSAMRI007 --prepare --flow-profile OSAMRI007FlowProfile_smoothed.csv
```

Output: `OSAMRI007_results/` with HDF5 cache, interactive HTML, template tracking JSON.

### Phase 2: Select Points

Pick anatomical landmarks using the PyVista GUI or interactive HTML.

```bash
# Option A: PyVista GUI (interactive 3D picker)
ama --subject OSAMRI007 --point-picker

# Option B: Open the interactive HTML from Phase 1 in your browser
# Then manually edit OSAMRI007_results/OSAMRI007_picked_points.json
```

Update `{SUBJECT}_results/{SUBJECT}_picked_points.json` with correct patch numbers, face indices, and coordinates for each landmark.

### Phase 3: Analyze

Generate all plots, reports, and analysis using the tracking locations.

```bash
ama --subject OSAMRI007 --plotting
```

Output: PDF reports, PNG figures, HTML visualizations in `OSAMRI007_results/`.

### All-in-One

Run the complete pipeline in one pass (requires pre-configured tracking locations):

```bash
ama --subject OSAMRI007 --all --flow-profile OSAMRI007FlowProfile_smoothed.csv
```

## Data Structure

### Input

```
working_directory/
├── {SUBJECT}_xyz_tables/              # Raw CFD CSV files
└── {SUBJECT}FlowProfile_smoothed.csv  # Breathing flow data
```

### Output

```
{SUBJECT}_results/
├── {SUBJECT}_cfd_data.h5              # HDF5 cache (all timesteps)
├── {SUBJECT}_cfd_data_light.h5        # Light HDF5 (single timestep, portable)
├── {SUBJECT}_picked_points.json       # Tracking locations (EDIT THIS)
├── {SUBJECT}_metadata.json            # System metadata
├── {SUBJECT}_key_time_points.json     # Detected breathing cycle time points
├── tracked_points/                    # CSV trajectory data
├── figures/                           # PNG images
├── reports/                           # PDF analysis reports
└── interactive/                       # HTML 3D visualizations
```

## Advanced Options

```bash
# Custom patch radii
ama --subject OSAMRI007 --patchradii 1.0 3.0 7.0

# Custom raw data directory
ama --subject OSAMRI007 --rawdir qDNS_xyz_tables

# Custom XYZ table path
ama --subject OSAMRI007 --xyz-path /path/to/xyz_tables

# Adjust surface normal filtering angle
ama --subject OSAMRI007 --normalangle 45.0

# Disable interactive visualization
ama --subject OSAMRI007 --disablevisualization

# Highlight patch regions only
ama --subject OSAMRI007 --highlight-patches --patch-timestep 100

# Handle remeshed data manually
ama --subject OSAMRI007 --has-remesh --remesh-before file1.csv --remesh-after file2.csv

# Force complete rerun
ama --subject OSAMRI007 --forcererun

# List available subjects
ama --listsubjects
```

## Tracking Locations

Phase 1 auto-generates a template in `{SUBJECT}_results/{SUBJECT}_picked_points.json`:

```json
{
  "locations": [
    {
      "description": "Posterior border of soft palate",
      "patch_number": 17,
      "face_indices": [12220],
      "coordinates": [-0.0101, 0.0081, 0.0386]
    }
  ],
  "combinations": []
}
```

To find patch/face values:
1. Open the interactive HTML from Phase 1 in your browser
2. Hover over points to see Patch Number and Face Index
3. Update the JSON file with your landmark values
4. Or use `ama --point-picker` for the PyVista GUI

## System Requirements

- **Python**: 3.9+
- **RAM**: 16GB+ (32GB+ for large datasets)
- **CPU**: 8+ cores recommended
- **Storage**: 50GB+ free space

## Troubleshooting

**Memory errors**: Reduce parallel processes with `export OMP_NUM_THREADS=4`

**HDF5 lock errors**: Remove the locked file and rerun with `--forcererun`

**macOS sklearn warning**: Known macOS + conda issue with threadpoolctl. Pipeline handles it gracefully with fallback defaults.

## License

MIT
