# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated CFD (Computational Fluid Dynamics) analysis pipeline for respiratory flow visualization and analysis. The system processes time-series CFD simulation data to analyze airflow patterns, pressure dynamics, and velocity/acceleration relationships in respiratory tract anatomy.

## Environment Setup

### Primary Setup Method
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate cfd-analysis

# Or using setup script
./setup.sh
```

### Alternative Setup
```bash
# Using pip
python -m venv cfd-env
source cfd-env/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Verification
```bash
python src/main.py --listsubjects
```

## Core Commands

### Two-Phase Workflow (Recommended)

The pipeline uses a two-phase workflow for better control:

```bash
# Phase 1: Patch Selection - Convert CSV to HDF5 and create interactive visualization
python src/main.py --subject OSAMRI007 --patch-selection --flow-profile OSAMRI007FlowProfile_smoothed.csv

# After Phase 1: Update {SUBJECT}_results/{SUBJECT}_tracking_locations.json with correct patch/face values

# Phase 2: Plotting - Generate all analysis and plots
python src/main.py --subject OSAMRI007 --plotting --flow-profile OSAMRI007FlowProfile_smoothed.csv
```

### Legacy Mode (Single Command)
```bash
# Complete analysis for a subject (requires pre-configured tracking locations)
python src/main.py --subject OSAMRI007

# Force complete rerun (overwrite existing data)
python src/main.py --subject OSAMRI007 --forcererun

# List available subjects
python src/main.py --listsubjects
```

### Advanced Analysis Options
```bash
# Custom patch analysis with different radii
python src/main.py --subject OSAMRI007 --patchradii 1.0 3.0 7.0

# Custom raw data directory
python src/main.py --subject OSAMRI007 --rawdir qDNS_xyz_tables

# Disable interactive visualization for faster processing
python src/main.py --subject OSAMRI007 --disablevisualization

# Highlight patch regions only (quick visualization)
python src/main.py --subject OSAMRI007 --highlight-patches --patch-timestep 100

# Handle remeshed data (when mesh changes during simulation)
python src/main.py --subject OSAMRI007 --has-remesh --remesh-before file1.csv --remesh-after file2.csv
```

### Testing
```bash
# Test XYZ file format handling
python test_xyz_formats.py

# Basic functionality test
python src/main.py --listsubjects
```

## Architecture Overview

### Core Entry Point
- `src/main.py` - Main pipeline orchestration and execution

### Key Modules
- `src/utils/file_processing.py` - File I/O, data preprocessing, breathing cycle detection
- `src/utils/parallel_csv_processing.py` - High-performance parallel processing with 8-10x speedup
- `src/data_processing/trajectory.py` - HDF5 caching and trajectory analysis
- `src/visualization/` - Interactive and static visualizations
- `src/surface_painting/` - 3D surface painting and region analysis

### Data Processing Pipeline (Two-Phase)

**Phase 1: Patch Selection (`--patch-selection`)**
1. CSV tables converted to HDF5 with 85% size reduction
2. Breathing cycle detection from flow profile
3. Interactive HTML generation for manual point selection
4. Template tracking locations JSON creation

**Phase 2: Plotting (`--plotting`)**
1. Load HDF5 cache (skip CSV processing)
2. Load tracking locations from results folder
3. Surface-normal filtering for anatomical landmark tracking
4. Pressure-velocity-acceleration correlation analysis
5. Generate PDF reports (both normalized and original time versions)
6. Interactive HTML 3D visualizations with patch regions

## Data Structure

### Expected Input Structure
```
project_root/
├── {SUBJECT}_xyz_tables/                    # Raw CFD data (CSV files)
├── {SUBJECT}FlowProfile.csv                 # Breathing flow data
├── {SUBJECT}FlowProfile_smoothed.csv        # Smoothed flow data
└── {SUBJECT}_tracking_locations.json        # Anatomical landmark definitions
```

### Generated Output Structure
```
project_root/
├── {SUBJECT}_xyz_tables_with_patches/       # Processed CFD data with patch numbers
└── {SUBJECT}_results/                       # All analysis results (self-contained)
    ├── {SUBJECT}_cfd_data.h5                # HDF5 cache (85% size reduction)
    ├── {SUBJECT}_tracking_locations.json    # Tracking locations (editable)
    ├── {SUBJECT}_key_time_points.json       # Detected key time points
    ├── tracked_points/                      # CSV trajectory data
    ├── figures/                             # PNG images
    ├── reports/                             # PDF analysis reports
    │   ├── {SUBJECT}_3x3_panel_smoothed_with_markers_normalized_time.pdf
    │   ├── {SUBJECT}_3x3_panel_smoothed_with_markers_original_time.pdf
    │   └── ...                              # Other PDF reports
    └── interactive/                         # HTML visualizations
        ├── {SUBJECT}_surface_patches_interactive_first_breathing_cycle_t{time}ms.html
        └── {SUBJECT}_patch_regions_t{time}ms.html
```

## Performance Characteristics

### Optimization Features
- **Parallel Processing**: 4-8x faster than sequential processing
- **HDF5 Caching**: 10-20x faster access for repeated operations
- **Memory-Efficient**: ~2GB per process, handles large datasets without overflow
- **Storage Optimization**: 85% compression with HDF5 format

### System Requirements
- **RAM**: 16GB+ recommended (32GB+ for large datasets)
- **CPU**: 8+ cores recommended for optimal parallel processing
- **Storage**: 50GB+ free space for processing large CFD datasets

## Key Configuration Files

### Tracking Locations
Phase 1 (`--patch-selection`) auto-generates a template in `{SUBJECT}_results/{SUBJECT}_tracking_locations.json`:
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
  "combinations": [],
  "_instructions": {
    "step1": "Open the interactive HTML visualization",
    "step2": "Hover over points to see Patch Number and Face Index",
    "step3": "Update patch_number, face_indices, and coordinates",
    "step4": "Run --plotting to generate analysis"
  }
}
```

### Key Output Files
- **Interactive HTML**: `{SUBJECT}_surface_patches_interactive_first_breathing_cycle_t{time}ms.html`
- **3x3 Panels** (both time versions for traceability):
  - `{SUBJECT}_3x3_panel_smoothed_with_markers_normalized_time.pdf` - Time starts at 0s
  - `{SUBJECT}_3x3_panel_smoothed_with_markers_original_time.pdf` - Original timestamps

## Common Issues and Solutions

### Memory Issues
```bash
# Reduce parallel processes if memory limited
export OMP_NUM_THREADS=4
python src/main.py --subject OSAMRI007
```

### HDF5 Lock Errors
```bash
# Remove locked files and force rerun
rm -f *_cfd_data.h5
python src/main.py --subject OSAMRI007 --forcererun
```

### macOS sklearn threadpoolctl Error
On macOS with conda, you may see `'NoneType' object has no attribute 'split'` errors from sklearn.
This is a known macOS + conda issue with threadpoolctl library version detection.
The pipeline handles this gracefully by falling back to default surface normals.
This issue does not occur on Linux.

### File Naming Conventions
The system handles multiple XYZ table naming formats:
- Integer format: `XYZ_Internal_Table_table_2387.csv` (milliseconds)
- Scientific notation: `XYZ_Internal_Table_table_2.300000e+00.csv` (seconds)

## Development Notes

### Code Style
- Follow existing patterns in the codebase for imports and error handling
- Use parallel processing utilities from `utils/parallel_csv_processing.py`
- Maintain chronological file ordering when processing time-series data

### Dependencies
The project uses scientific computing stack (numpy, pandas, scipy) with specialized libraries:
- HDF5 processing: h5py, tables
- Visualization: matplotlib, plotly, seaborn, pyvista
- Performance: numba, scikit-learn
- System monitoring: psutil, tqdm

### File Processing
- Always use the existing file processing utilities for CFD data
- Maintain compatibility with both CSV and HDF5 formats
- Respect breathing cycle boundaries when filtering time-series data