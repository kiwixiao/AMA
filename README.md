# CFD Airway Analysis Pipeline

A comprehensive computational fluid dynamics (CFD) analysis pipeline for studying airway dynamics during breathing cycles. This pipeline processes large-scale CFD data to track anatomical landmarks, analyze pressure-velocity relationships, and generate detailed visualizations.

## 🚀 Features

- **High-Performance Processing**: Parallel processing with 8-10x speedup over sequential methods
- **HDF5 Caching**: Intelligent caching system reducing data size by 85% and providing 10-20x faster access
- **Advanced Tracking**: Surface-normal filtering for anatomical landmark tracking
- **Interactive Visualizations**: 3D HTML visualizations with patch region analysis
- **Comprehensive Analysis**: Pressure, velocity, acceleration correlation analysis
- **Batch Processing**: Process multiple subjects with consistent parameters

## 📋 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **RAM**: 16GB+ recommended (32GB+ for large datasets)
- **CPU**: 8+ cores recommended for optimal parallel processing
- **Storage**: 50GB+ free space for processing large CFD datasets

### Software Requirements
- Python 3.9+
- Conda (recommended) or pip

## 🛠️ Installation

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd newMetrics

# Create and activate conda environment
conda env create -f environment.yml
conda activate cfd-analysis
```

### Option 2: Pip Installation

```bash
# Clone the repository
git clone <repository-url>
cd newMetrics

# Create virtual environment
python -m venv cfd-env
source cfd-env/bin/activate  # Linux/macOS
# or
cfd-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verification

Test the installation:
```bash
python src/main.py --listsubjects
```

## 📁 Data Structure

The pipeline expects the following directory structure:

```
project_root/
├── {SUBJECT}_xyz_tables/                    # Raw CFD data (CSV files)
├── {SUBJECT}FlowProfile.csv                 # Breathing flow data
├── {SUBJECT}FlowProfile_smoothed.csv        # Smoothed flow data
├── {SUBJECT}_tracking_locations.json        # Anatomical landmark definitions
└── src/                                     # Source code
```

**Generated outputs:**
```
project_root/
├── {SUBJECT}_xyz_tables_with_patches/       # Processed CFD data with patch numbers
└── {SUBJECT}_results/                       # All results (self-contained folder)
    ├── {SUBJECT}_cfd_data.h5                # HDF5 cache (85% size reduction)
    ├── {SUBJECT}_tracking_locations.json    # Tracking locations (editable)
    ├── tracked_points/                      # CSV trajectory data
    ├── figures/                             # PNG images
    ├── reports/                             # PDF analysis reports
    └── interactive/                         # HTML visualizations
```

## 🎯 Usage

### Two-Phase Workflow (Recommended)

The pipeline uses a two-phase workflow for better control and manual verification:

#### Phase 1: Patch Selection
```bash
# Convert CSV to HDF5 and create interactive visualization for point selection
python src/main.py --subject OSAMRI007 --patch-selection --flow-profile OSAMRI007FlowProfile_smoothed.csv
```

This phase:
- Converts raw CSV files to HDF5 format (85% size reduction)
- Detects breathing cycle from flow profile
- Creates interactive HTML visualization for manual patch/face selection
- Generates template tracking locations JSON file in `{SUBJECT}_results/`

After Phase 1, open the interactive HTML and update `{SUBJECT}_results/{SUBJECT}_tracking_locations.json` with the correct patch numbers and face indices.

#### Phase 2: Plotting and Analysis
```bash
# Generate all analysis and plots using the updated tracking locations
python src/main.py --subject OSAMRI007 --plotting --flow-profile OSAMRI007FlowProfile_smoothed.csv
```

This phase:
- Reads HDF5 file (skips CSV processing)
- Loads tracking locations from results folder
- Generates all analysis, tracking, and visualization outputs

### Quick Start (Legacy Mode)

```bash
# Complete analysis for a subject (requires pre-configured tracking locations)
python src/main.py --subject OSAMRI007

# Force complete rerun (overwrite existing data)
python src/main.py --subject OSAMRI007 --forcererun

# List available subjects
python src/main.py --listsubjects
```

### Advanced Usage

```bash
# Custom patch analysis with different radii
python src/main.py --subject OSAMRI007 --patchradii 1.0 3.0 7.0

# Custom raw data directory
python src/main.py --subject OSAMRI007 --rawdir qDNS_xyz_tables

# Adjust surface normal filtering
python src/main.py --subject OSAMRI007 --normalangle 45.0

# Disable interactive visualization for faster processing
python src/main.py --subject OSAMRI007 --disablevisualization

# Highlight patch regions only (quick visualization)
python src/main.py --subject OSAMRI007 --highlight-patches --patch-timestep 100

# Handle remeshed data (when mesh changes during simulation)
python src/main.py --subject OSAMRI007 --has-remesh --remesh-before file1.csv --remesh-after file2.csv
```

### Batch Processing

```bash
# Process multiple subjects
for subject in OSAMRI007 OSAMRI008 OSAMRI009; do
    python src/main.py --subject $subject
done
```

## 🔧 Configuration

### Tracking Locations

The pipeline automatically generates a template tracking locations file during Phase 1 (`--patch-selection`).
Edit `{SUBJECT}_results/{SUBJECT}_tracking_locations.json` to define anatomical landmarks:

```json
{
  "locations": [
    {
      "description": "Posterior border of soft palate",
      "patch_number": 17,
      "face_indices": [12220],
      "coordinates": [-0.0101, 0.0081, 0.0386]
    },
    {
      "description": "Back of tongue",
      "patch_number": 25,
      "face_indices": [8450],
      "coordinates": [-0.0085, 0.0120, 0.0290]
    }
  ],
  "combinations": [],
  "_instructions": {
    "step1": "Open the interactive HTML visualization in your browser",
    "step2": "Hover over points to see Patch Number and Face Index",
    "step3": "Update each location's patch_number, face_indices, and coordinates",
    "step4": "Update the description to something meaningful",
    "step5": "Run the pipeline again with --plotting to generate analysis"
  }
}
```

**How to find patch/face values:**
1. Run Phase 1 to generate the interactive HTML
2. Open `{SUBJECT}_results/interactive/{SUBJECT}_surface_patches_interactive_first_breathing_cycle_t{time}ms.html`
3. Hover over points to see Patch Number and Face Index
4. Update the JSON file with the values for your anatomical landmarks

### Performance Tuning

The pipeline automatically detects optimal settings:
- **Process count**: Based on CPU cores and available memory
- **Chunk size**: Optimized for file sizes and memory constraints
- **Caching strategy**: Intelligent HDF5 conversion for repeated access

## 📊 Output Files

### Data Files
- **CSV trajectories**: `{SUBJECT}_results/tracked_points/`
- **HDF5 cache**: `{SUBJECT}_results/{SUBJECT}_cfd_data.h5` (compressed, fast access)
- **Key time points**: `{SUBJECT}_results/{SUBJECT}_key_time_points.json`
- **Tracking locations**: `{SUBJECT}_results/{SUBJECT}_tracking_locations.json`

### Visualizations
- **Interactive HTML**:
  - `{SUBJECT}_surface_patches_interactive_first_breathing_cycle_t{time}ms.html` - Full surface with patch selection
  - `{SUBJECT}_patch_regions_t{time}ms.html` - Highlighted patch regions
- **PNG figures**: Individual plots in `{SUBJECT}_results/figures/`

### PDF Reports (`{SUBJECT}_results/reports/`)
- **3x3 Panel Analysis** (pressure, velocity, acceleration correlations):
  - `{SUBJECT}_3x3_panel_smoothed_with_markers_normalized_time.pdf` - Time starts at 0s
  - `{SUBJECT}_3x3_panel_smoothed_with_markers_original_time.pdf` - Original timestamps for traceability
  - `{SUBJECT}_3x3_panel_clean.pdf` - Clean version without markers
  - `{SUBJECT}_3x3_panel_smooth.pdf` - Smoothed version
- **Correlation Analysis**:
  - `{SUBJECT}_pressure_motion_correlation_50ms.pdf` - 50ms window
  - `{SUBJECT}_pressure_motion_correlation_100ms.pdf` - 100ms window
- **CFD Analysis**:
  - `{SUBJECT}_cfd_analysis.pdf` - Comprehensive CFD analysis
  - `{SUBJECT}_cfd_analysis_3x3.pdf` - Multi-page 3x3 panels
  - `{SUBJECT}_cfd_analysis_3x3_with_markers.pdf` - With zero-crossing markers
- **Other**:
  - `{SUBJECT}_airway_surface_velocity.pdf` - Surface velocity analysis
  - `{SUBJECT}_clean_flow_profile.pdf` - Breathing flow profile

## ⚡ Performance

### Typical Performance (24mmesh dataset):
- **Total runtime**: ~21 minutes (was 24+ hours sequential)
- **Parallel efficiency**: 8x speedup with 15 processes
- **Memory usage**: ~2GB per process
- **Storage reduction**: 85% with HDF5 compression

### Optimization Features:
- **Parallel CSV processing**: 4-8x faster than sequential
- **HDF5 caching**: 10-20x faster for repeated access
- **Intelligent file filtering**: Process only breathing cycle data
- **Memory-efficient chunking**: Handle large files without memory overflow

## 🔍 Troubleshooting

### Common Issues

**1. Memory errors with large files:**
```bash
# Reduce parallel processes if memory limited
export OMP_NUM_THREADS=4
python src/main.py --subject OSAMRI007
```

**2. Missing dependencies:**
```bash
# Reinstall environment
conda env remove -n cfd-analysis
conda env create -f environment.yml
```

**3. File permission errors:**
```bash
# Ensure write permissions
chmod -R 755 .
```

**4. HDF5 lock errors:**
```bash
# Remove locked files from results folder
rm -f OSAMRI007_results/OSAMRI007_cfd_data.h5
python src/main.py --subject OSAMRI007 --forcererun
```

### Performance Issues

**Slow processing:**
- Check available RAM (16GB+ recommended)
- Verify SSD storage for better I/O
- Monitor CPU usage during parallel processing

**Large output files:**
- Use HDF5 caching to reduce storage
- Enable compression in visualization settings
- Clean up intermediate files after processing

## 🧪 Testing

Run the test suite:
```bash
# Basic functionality test
python src/main.py --listsubjects

# Test with small dataset
python src/main.py --subject OSAMRI007 --disablepatchanalysis

# Verify parallel processing
python src/main.py --subject OSAMRI007 --forcererun
```

## 📚 Documentation

### Key Components

- **`src/main.py`**: Main pipeline orchestration
- **`src/utils/parallel_csv_processing.py`**: High-performance parallel processing
- **`src/utils/file_processing.py`**: File I/O and data preprocessing
- **`src/data_processing/trajectory.py`**: HDF5 caching and trajectory analysis
- **`src/visualization/`**: Interactive and static visualizations

### Algorithm Details

**Surface Normal Filtering**: Uses PCA-based normal calculation with connectivity clustering to identify anatomically consistent surface patches.

**Parallel Processing**: Maintains chronological file ordering while processing multiple files simultaneously for optimal performance.

**HDF5 Caching**: Converts CSV data to compressed HDF5 format with intelligent column typing and error handling.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## 📄 License

[Add your license information here]

## 📧 Support

For questions or issues:
- Check the troubleshooting section above
- Review the generated log files in `{SUBJECT}_results/`
- Open an issue on the repository

## 🎯 Citation

If you use this pipeline in your research, please cite:
[Add citation information here] 