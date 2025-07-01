# Surface Painting Module Usage Guide

This module provides interactive STL surface painting capabilities for CFD analysis. It's designed as an **add-on feature** that doesn't affect your existing research pipeline.

## Overview

The surface painting workflow consists of three main steps:

1. **Paint STL surfaces** interactively to define regions of interest
2. **Map painted regions** to CFD data using spatial coordinates
3. **Analyze mapped regions** to extract statistics and insights

## Quick Start

### Step 1: Paint STL Surface

```bash
# Paint an STL surface interactively
python -m src.surface_painting.stl_painter --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl

# Custom brush size (5mm default)
python -m src.surface_painting.stl_painter --stl surface.stl --brush-size 2.0
```

**Interactive Controls:**
- `p` - Toggle painting mode ON/OFF
- `1-5` - Switch between different colored regions
- Click and drag to paint when painting mode is ON
- `c` - Clear current region
- `C` - Clear all regions
- `s` - Save painted regions
- `l` - Load saved regions
- `e` - Export coordinates for mapping
- `q` - Quit

### Step 2: Map to CFD Data

```bash
# Map painted regions to CFD data
python -m src.surface_painting.coordinate_mapper \
  --painted-coords out_0100.000_ds_10_l2_aBE0.001_be_0.001_painted_coordinates.json \
  --csv-data less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_100.csv

# Custom tolerance levels (in millimeters)
python -m src.surface_painting.coordinate_mapper \
  --painted-coords coords.json \
  --csv-data data.csv \
  --tolerances 0.5 1.0 2.0

# Export regional CFD data subsets
python -m src.surface_painting.coordinate_mapper \
  --painted-coords coords.json \
  --csv-data data.csv \
  --export-cfd
```

### Step 3: Analyze Results

```bash
# Generate comprehensive analysis report
python -m src.surface_painting.regional_analyzer \
  --mapping-results region_mapping_results.json

# Custom output file
python -m src.surface_painting.regional_analyzer \
  --mapping-results results.json \
  --output my_analysis.pdf
```

## Complete Workflow Example

```bash
# 1. Paint regions on STL surface
python -m src.surface_painting.stl_painter \
  --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl \
  --brush-size 3.0

# 2. Map painted regions to CFD data with multiple tolerance levels
python -m src.surface_painting.coordinate_mapper \
  --painted-coords out_0100.000_ds_10_l2_aBE0.001_be_0.001_painted_coordinates.json \
  --csv-data less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_100.csv \
  --tolerances 1.0 2.0 5.0 \
  --export-cfd

# 3. Generate analysis report
python -m src.surface_painting.regional_analyzer \
  --mapping-results region_mapping_results.json \
  --output painted_regions_analysis.pdf
```

## Output Files

### STL Painter Outputs:
- `{stl_name}_painted_regions.pkl` - Saved painted regions (binary)
- `{stl_name}_painted_coordinates.json` - Coordinate data for mapping

### Coordinate Mapper Outputs:
- `region_mapping_results.json` - Mapping statistics and results
- `regional_cfd_data/` - Directory with CFD data subsets for each region
  - `{region_name}_{tolerance}_cfd_data.csv` - CFD data for specific region

### Regional Analyzer Outputs:
- `regional_analysis_report.pdf` - Comprehensive analysis report

## Integration with Existing Pipeline

This module is designed to **not interfere** with your existing research pipeline. It operates independently and can be used alongside your current analysis tools.

### Existing Pipeline Usage:
```bash
# Your existing commands work exactly the same
python src/main.py --subject less1mmeshOSAMRI007 --timestep 100
python src/main.py --visual-only --subject OSAMRI007 --visual-timestep 100
```

### Surface Painting Usage:
```bash
# New add-on functionality
python -m src.surface_painting.stl_painter --stl STLoutput/surface.stl
```

## Tips and Best Practices

### Painting Tips:
1. **Start with painting mode OFF** - Navigate and explore the 3D surface first
2. **Use appropriate brush size** - 2-5mm typically works well for airway surfaces
3. **Paint in multiple passes** - Build up regions gradually
4. **Save frequently** - Use `s` key to save your work
5. **Use different regions** - Switch between regions (1-5) for different anatomical areas

### Mapping Tips:
1. **Try multiple tolerance levels** - Start with 1-2mm, increase if needed
2. **Check mapping statistics** - Review how many CFD points are matched
3. **Export CFD data** - Use `--export-cfd` to get regional data subsets

### Analysis Tips:
1. **Review the PDF report** - Contains comprehensive regional statistics
2. **Compare regions** - Look for differences between painted areas
3. **Validate results** - Cross-check with your existing analysis

## Dependencies

The surface painting module requires:
- `pyvista` - For interactive 3D visualization
- `scipy` - For spatial mapping algorithms
- Standard scientific Python stack (numpy, pandas, matplotlib)

Install missing dependencies:
```bash
pip install pyvista scipy
```

## Troubleshooting

### Common Issues:

**PyVista not displaying:**
- Check if you're in a graphical environment
- Try different PyVista backends: `pv.set_jupyter_backend('panel')`

**No CFD points matched:**
- Increase tolerance levels (try 5-10mm)
- Check coordinate units (STL vs CSV)
- Verify STL and CSV files correspond to same geometry

**Painting mode not responsive:**
- Ensure painting mode is ON (press `p`)
- Check brush size (might be too small)
- Try clicking directly on surface mesh

## Support

This module is an add-on to your existing CFD analysis pipeline. If you encounter issues:

1. **Check file paths** - Ensure STL and CSV files exist
2. **Verify coordinate systems** - STL and CFD data should use consistent units
3. **Review tolerance settings** - Start with larger tolerances for testing
4. **Save work frequently** - Use the save/load functionality to preserve painted regions

The module is designed to complement your existing research workflow without modification. 