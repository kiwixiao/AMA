#!/usr/bin/env python3
"""
STL-CSV Matching Analysis Tool

This script analyzes the relationship between STL mesh files and CSV point cloud data
to determine surface connectivity and matching quality across different scenarios:

1. Perfect match: STL(t=0) + CSV(t=0) with identical coordinates
2. Time mismatch: STL(t=0) + CSV(t=10ms) - surface deformation
3. Cell center vs vertices: STL vertices ≠ CSV cell centers
4. Combination: Different times + different coordinate systems

Usage:
    python src/analyze_stl_csv_matching.py --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl --csv less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_100.csv
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def read_stl_file(stl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read STL file and extract vertices, triangles, and normals.
    
    Args:
        stl_path: Path to STL file
        
    Returns:
        Tuple of (vertices, triangles, normals)
    """
    print(f"📁 Reading STL file: {stl_path}")
    
    try:
        # Try using numpy-stl if available
        try:
            from stl import mesh
            stl_mesh = mesh.Mesh.from_file(str(stl_path))
            
            # Extract unique vertices and triangles
            vertices = stl_mesh.vectors.reshape(-1, 3)
            unique_vertices, vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
            triangles = vertex_indices.reshape(-1, 3)
            normals = stl_mesh.normals
            
            print(f"   ✅ STL loaded: {len(unique_vertices):,} vertices, {len(triangles):,} triangles")
            return unique_vertices, triangles, normals
            
        except ImportError:
            print("   ⚠️  numpy-stl not available, using basic parser")
            return read_stl_basic(stl_path)
            
    except Exception as e:
        print(f"   ❌ Error reading STL: {e}")
        return None, None, None

def read_stl_basic(stl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basic STL parser for ASCII format."""
    vertices = []
    normals = []
    
    with open(stl_path, 'r') as f:
        current_normal = None
        triangle_vertices = []
        
        for line in f:
            line = line.strip()
            
            if line.startswith('facet normal'):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                triangle_vertices = []
                
            elif line.startswith('vertex'):
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
                triangle_vertices.append(len(vertices) - 1)
                
            elif line.startswith('endfacet'):
                if current_normal and len(triangle_vertices) == 3:
                    normals.append(current_normal)
    
    vertices = np.array(vertices)
    unique_vertices, vertex_indices = np.unique(vertices, axis=0, return_inverse=True)
    triangles = vertex_indices.reshape(-1, 3)
    normals = np.array(normals)
    
    print(f"   ✅ STL loaded: {len(unique_vertices):,} vertices, {len(triangles):,} triangles")
    return unique_vertices, triangles, normals

def read_csv_file(csv_path: Path) -> pd.DataFrame:
    """
    Read CSV file with surface point data.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with surface points
    """
    print(f"📁 Reading CSV file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"   ✅ CSV loaded: {len(df):,} points")
        
        # Check required columns
        required_cols = ['X (m)', 'Y (m)', 'Z (m)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ❌ Missing columns: {missing_cols}")
            return None
            
        print(f"   📊 Coordinate range:")
        print(f"      X: {df['X (m)'].min():.6f} to {df['X (m)'].max():.6f}")
        print(f"      Y: {df['Y (m)'].min():.6f} to {df['Y (m)'].max():.6f}")
        print(f"      Z: {df['Z (m)'].min():.6f} to {df['Z (m)'].max():.6f}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        return None

def analyze_coordinate_matching(stl_vertices: np.ndarray, csv_coords: np.ndarray, 
                               tolerance_levels: List[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]) -> Dict:
    """
    Analyze how well STL vertices match CSV coordinates.
    
    Args:
        stl_vertices: STL vertex coordinates (N, 3)
        csv_coords: CSV point coordinates (M, 3)
        tolerance_levels: Distance tolerances to test
        
    Returns:
        Dictionary with matching statistics
    """
    print(f"\n🔍 Analyzing coordinate matching...")
    print(f"   STL vertices: {len(stl_vertices):,}")
    print(f"   CSV points: {len(csv_coords):,}")
    
    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(csv_coords)
    
    # Find closest CSV point for each STL vertex
    distances, indices = tree.query(stl_vertices)
    
    results = {
        'total_stl_vertices': len(stl_vertices),
        'total_csv_points': len(csv_coords),
        'min_distance': float(distances.min()),
        'max_distance': float(distances.max()),
        'mean_distance': float(distances.mean()),
        'median_distance': float(np.median(distances)),
        'tolerance_stats': {}
    }
    
    print(f"\n📊 Distance Statistics:")
    print(f"   Min distance: {results['min_distance']:.6f} m ({results['min_distance']*1000:.3f} mm)")
    print(f"   Max distance: {results['max_distance']:.6f} m ({results['max_distance']*1000:.3f} mm)")
    print(f"   Mean distance: {results['mean_distance']:.6f} m ({results['mean_distance']*1000:.3f} mm)")
    print(f"   Median distance: {results['median_distance']:.6f} m ({results['median_distance']*1000:.3f} mm)")
    
    print(f"\n🎯 Tolerance Analysis:")
    for tolerance in tolerance_levels:
        matches = np.sum(distances <= tolerance)
        percentage = (matches / len(stl_vertices)) * 100
        
        results['tolerance_stats'][tolerance] = {
            'matches': int(matches),
            'percentage': float(percentage)
        }
        
        print(f"   Within {tolerance:.0e} m ({tolerance*1000:.3f} mm): {matches:,} vertices ({percentage:.1f}%)")
    
    return results

def analyze_surface_connectivity(stl_vertices: np.ndarray, stl_triangles: np.ndarray, 
                                csv_coords: np.ndarray, tolerance: float = 1e-3) -> Dict:
    """
    Analyze surface connectivity by finding which CSV points correspond to STL triangles.
    
    Args:
        stl_vertices: STL vertex coordinates
        stl_triangles: STL triangle indices
        csv_coords: CSV point coordinates
        tolerance: Distance tolerance for matching
        
    Returns:
        Dictionary with connectivity analysis
    """
    print(f"\n🔗 Analyzing surface connectivity (tolerance: {tolerance*1000:.1f}mm)...")
    
    # Build KDTree for CSV points
    tree = cKDTree(csv_coords)
    
    # For each STL triangle, find corresponding CSV points
    triangle_matches = []
    total_csv_points_near_triangles = 0
    
    for i, triangle in enumerate(stl_triangles[:1000]):  # Limit to first 1000 for performance
        # Get triangle vertices
        tri_vertices = stl_vertices[triangle]
        
        # Calculate triangle centroid
        centroid = np.mean(tri_vertices, axis=0)
        
        # Find CSV points within tolerance of triangle vertices and centroid
        vertex_matches = []
        for vertex in tri_vertices:
            indices = tree.query_ball_point(vertex, tolerance)
            vertex_matches.extend(indices)
        
        centroid_matches = tree.query_ball_point(centroid, tolerance)
        
        # Combine and remove duplicates
        all_matches = list(set(vertex_matches + centroid_matches))
        triangle_matches.append(len(all_matches))
        total_csv_points_near_triangles += len(all_matches)
        
        if i % 200 == 0:
            print(f"   Processed {i+1:,} triangles...")
    
    results = {
        'triangles_analyzed': len(triangle_matches),
        'avg_csv_points_per_triangle': float(np.mean(triangle_matches)),
        'total_csv_points_near_triangles': total_csv_points_near_triangles,
        'csv_coverage_percentage': (total_csv_points_near_triangles / len(csv_coords)) * 100
    }
    
    print(f"   ✅ Analyzed {results['triangles_analyzed']:,} triangles")
    print(f"   📊 Average CSV points per triangle: {results['avg_csv_points_per_triangle']:.2f}")
    print(f"   🎯 CSV points near triangles: {results['total_csv_points_near_triangles']:,}")
    print(f"   📈 CSV coverage: {results['csv_coverage_percentage']:.1f}%")
    
    return results

def estimate_time_deformation(stl_vertices: np.ndarray, csv_coords: np.ndarray) -> Dict:
    """
    Estimate surface deformation by analyzing coordinate differences.
    
    Args:
        stl_vertices: STL vertex coordinates
        csv_coords: CSV point coordinates
        
    Returns:
        Dictionary with deformation analysis
    """
    print(f"\n⏱️  Analyzing potential time-based deformation...")
    
    # Find coordinate ranges for both datasets
    stl_bounds = {
        'x': (stl_vertices[:, 0].min(), stl_vertices[:, 0].max()),
        'y': (stl_vertices[:, 1].min(), stl_vertices[:, 1].max()),
        'z': (stl_vertices[:, 2].min(), stl_vertices[:, 2].max())
    }
    
    csv_bounds = {
        'x': (csv_coords[:, 0].min(), csv_coords[:, 0].max()),
        'y': (csv_coords[:, 1].min(), csv_coords[:, 1].max()),
        'z': (csv_coords[:, 2].min(), csv_coords[:, 2].max())
    }
    
    # Calculate bounding box differences
    bound_diffs = {}
    for axis in ['x', 'y', 'z']:
        stl_range = stl_bounds[axis][1] - stl_bounds[axis][0]
        csv_range = csv_bounds[axis][1] - csv_bounds[axis][0]
        
        center_diff = abs((stl_bounds[axis][0] + stl_bounds[axis][1])/2 - 
                         (csv_bounds[axis][0] + csv_bounds[axis][1])/2)
        
        bound_diffs[axis] = {
            'stl_range': float(stl_range),
            'csv_range': float(csv_range),
            'range_ratio': float(csv_range / stl_range) if stl_range > 0 else 1.0,
            'center_shift': float(center_diff)
        }
    
    results = {
        'stl_bounds': stl_bounds,
        'csv_bounds': csv_bounds,
        'bound_differences': bound_diffs,
        'overall_deformation': float(np.sqrt(sum([bd['center_shift']**2 for bd in bound_diffs.values()])))
    }
    
    print(f"   📏 Bounding box analysis:")
    for axis in ['x', 'y', 'z']:
        bd = bound_diffs[axis]
        print(f"      {axis.upper()}: STL range {bd['stl_range']*1000:.1f}mm, CSV range {bd['csv_range']*1000:.1f}mm")
        print(f"           Ratio: {bd['range_ratio']:.3f}, Center shift: {bd['center_shift']*1000:.3f}mm")
    
    print(f"   🎯 Overall deformation: {results['overall_deformation']*1000:.3f}mm")
    
    return results

def determine_scenario_type(matching_results: Dict, deformation_results: Dict) -> str:
    """
    Determine which scenario we're dealing with based on analysis results.
    
    Args:
        matching_results: Results from coordinate matching analysis
        deformation_results: Results from deformation analysis
        
    Returns:
        String describing the likely scenario
    """
    # Check for exact matches (perfect scenario)
    exact_matches = matching_results['tolerance_stats'].get(1e-6, {}).get('percentage', 0)
    close_matches = matching_results['tolerance_stats'].get(1e-3, {}).get('percentage', 0)
    
    overall_deformation = deformation_results['overall_deformation'] * 1000  # Convert to mm
    
    if exact_matches > 90:
        return "PERFECT MATCH: STL and CSV coordinates are nearly identical"
    elif close_matches > 80 and overall_deformation < 1.0:
        return "CELL CENTER vs VERTICES: Different coordinate systems but same timepoint"
    elif close_matches > 50 and overall_deformation > 1.0:
        return "TIME MISMATCH: Surface has deformed between STL and CSV timepoints"
    elif close_matches < 50:
        return "MAJOR MISMATCH: Significant differences in geometry or coordinate systems"
    else:
        return "MIXED SCENARIO: Combination of coordinate differences and deformation"

def generate_report(stl_path: Path, csv_path: Path, matching_results: Dict, 
                   connectivity_results: Dict, deformation_results: Dict, scenario: str) -> None:
    """Generate a comprehensive report of the analysis."""
    
    print(f"\n" + "="*80)
    print(f"📋 STL-CSV MATCHING ANALYSIS REPORT")
    print(f"="*80)
    
    print(f"\n📁 Input Files:")
    print(f"   STL: {stl_path}")
    print(f"   CSV: {csv_path}")
    
    print(f"\n🎯 Scenario Assessment:")
    print(f"   {scenario}")
    
    print(f"\n📊 Key Statistics:")
    print(f"   • STL vertices: {matching_results['total_stl_vertices']:,}")
    print(f"   • CSV points: {matching_results['total_csv_points']:,}")
    print(f"   • Mean distance: {matching_results['mean_distance']*1000:.3f}mm")
    print(f"   • Points within 1mm: {matching_results['tolerance_stats'].get(1e-3, {}).get('percentage', 0):.1f}%")
    print(f"   • Overall deformation: {deformation_results['overall_deformation']*1000:.3f}mm")
    
    if connectivity_results:
        print(f"   • Avg CSV points per triangle: {connectivity_results['avg_csv_points_per_triangle']:.2f}")
        print(f"   • CSV coverage: {connectivity_results['csv_coverage_percentage']:.1f}%")
    
    print(f"\n💡 Implications for Surface Analysis:")
    if "PERFECT MATCH" in scenario:
        print(f"   ✅ Can use STL connectivity for geodesic distance calculations")
        print(f"   ✅ Surface normals from STL are accurate for CSV points")
        print(f"   ✅ True surface-based patch filtering is possible")
    elif "CELL CENTER" in scenario:
        print(f"   ⚠️  Need interpolation to map STL connectivity to CSV cell centers")
        print(f"   ✅ Surface topology is still valid with proper mapping")
        print(f"   🔧 Geodesic distances possible with coordinate transformation")
    elif "TIME MISMATCH" in scenario:
        print(f"   ⚠️  Surface has deformed - STL topology may not be accurate")
        print(f"   🔧 Can use STL as approximate topology with distance corrections")
        print(f"   ❌ Exact geodesic distances not reliable")
    else:
        print(f"   ❌ STL topology not suitable for this CSV data")
        print(f"   🔧 Continue using Euclidean distance approximation")
        print(f"   💡 Consider finding matching STL file for this timepoint")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze STL-CSV matching for surface connectivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze first STL frame with first CSV timepoint
  python src/analyze_stl_csv_matching.py --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl --csv less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_1.csv
  
     # Test time mismatch scenario
   python src/analyze_stl_csv_matching.py --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl --csv less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_100.csv --tolerance 0.002
   
   # Convert STL from mm to meters
   python src/analyze_stl_csv_matching.py --stl STLoutput/out_0100.000_ds_10_l2_aBE0.001_be_0.001.stl --csv less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_100.csv --stl-scale 0.001
        """
    )
    
    parser.add_argument('--stl', required=True, type=Path,
                        help='Path to STL file')
    
    parser.add_argument('--csv', required=True, type=Path,
                        help='Path to CSV file with surface points')
    
    parser.add_argument('--tolerance', type=float, default=0.001,
                        help='Distance tolerance for connectivity analysis (meters, default: 0.001)')
    
    parser.add_argument('--skip-connectivity', action='store_true',
                        help='Skip connectivity analysis (faster for large datasets)')
    
    parser.add_argument('--stl-scale', type=float, default=1.0,
                        help='Scale factor to apply to STL coordinates (e.g., 0.001 to convert mm to m)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.stl.exists():
        print(f"❌ Error: STL file not found: {args.stl}")
        sys.exit(1)
    
    if not args.csv.exists():
        print(f"❌ Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    print(f"🚀 Starting STL-CSV matching analysis...")
    
    # Read input files
    stl_vertices, stl_triangles, stl_normals = read_stl_file(args.stl)
    if stl_vertices is None:
        print(f"❌ Failed to read STL file")
        sys.exit(1)
    
    # Apply scale factor to STL coordinates if specified
    if args.stl_scale != 1.0:
        print(f"🔧 Applying scale factor {args.stl_scale} to STL coordinates...")
        stl_vertices = stl_vertices * args.stl_scale
        print(f"   ✅ STL coordinates scaled")
        print(f"   📊 New STL coordinate range:")
        print(f"      X: {stl_vertices[:, 0].min():.6f} to {stl_vertices[:, 0].max():.6f}")
        print(f"      Y: {stl_vertices[:, 1].min():.6f} to {stl_vertices[:, 1].max():.6f}")
        print(f"      Z: {stl_vertices[:, 2].min():.6f} to {stl_vertices[:, 2].max():.6f}")
    
    csv_df = read_csv_file(args.csv)
    if csv_df is None:
        print(f"❌ Failed to read CSV file")
        sys.exit(1)
    
    # Extract CSV coordinates
    csv_coords = csv_df[['X (m)', 'Y (m)', 'Z (m)']].values
    
    # Perform analyses
    matching_results = analyze_coordinate_matching(stl_vertices, csv_coords)
    deformation_results = estimate_time_deformation(stl_vertices, csv_coords)
    
    connectivity_results = None
    if not args.skip_connectivity and stl_triangles is not None:
        connectivity_results = analyze_surface_connectivity(stl_vertices, stl_triangles, csv_coords, args.tolerance)
    
    # Determine scenario
    scenario = determine_scenario_type(matching_results, deformation_results)
    
    # Generate report
    generate_report(args.stl, args.csv, matching_results, connectivity_results, deformation_results, scenario)
    
    print(f"\n🎉 Analysis completed successfully!")

if __name__ == "__main__":
    main() 