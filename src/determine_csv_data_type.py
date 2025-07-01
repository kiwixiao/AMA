#!/usr/bin/env python3
"""
CSV Data Type Analysis Tool

This script analyzes CSV data to determine whether it contains:
1. Cell centers (face centroids)
2. Vertices (mesh nodes)
3. Integration points

Key indicators:
- Face Index column presence indicates face-based data (cell centers)
- Area values indicate face-based quantities
- VdotN (velocity dot normal) indicates face-based flow calculations
- Point density and distribution patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def analyze_csv_structure(csv_path: Path) -> dict:
    """Analyze CSV structure to determine data type."""
    print(f"📊 Analyzing CSV structure: {csv_path}")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    analysis = {
        'total_points': len(df),
        'columns': list(df.columns),
        'has_face_index': 'Face Index' in df.columns,
        'has_area': any('Area' in col for col in df.columns),
        'has_vdotn': 'VdotN' in df.columns,
        'has_pressure': any('Pressure' in col for col in df.columns),
        'has_velocity': any('Velocity' in col for col in df.columns),
    }
    
    # Check Face Index values
    if analysis['has_face_index']:
        face_indices = df['Face Index'].values
        analysis['face_index_stats'] = {
            'min': int(face_indices.min()),
            'max': int(face_indices.max()),
            'unique_count': len(np.unique(face_indices)),
            'sequential': np.array_equal(np.unique(face_indices), np.arange(face_indices.min(), face_indices.max() + 1))
        }
    
    # Analyze area values
    if analysis['has_area']:
        area_col = [col for col in df.columns if 'Area: Magnitude' in col][0]
        areas = df[area_col].values
        analysis['area_stats'] = {
            'min': float(areas.min()),
            'max': float(areas.max()),
            'mean': float(areas.mean()),
            'all_positive': bool(np.all(areas > 0)),
            'typical_scale': float(np.median(areas))
        }
    
    # Analyze coordinate distribution
    coords = df[['X (m)', 'Y (m)', 'Z (m)']].values
    analysis['coordinate_stats'] = {
        'x_range': float(coords[:, 0].max() - coords[:, 0].min()),
        'y_range': float(coords[:, 1].max() - coords[:, 1].min()),
        'z_range': float(coords[:, 2].max() - coords[:, 2].min()),
        'mean_spacing': estimate_point_spacing(coords)
    }
    
    return analysis

def estimate_point_spacing(coords: np.ndarray, sample_size: int = 1000) -> float:
    """Estimate typical spacing between points."""
    from scipy.spatial import cKDTree
    
    # Sample points for efficiency
    if len(coords) > sample_size:
        indices = np.random.choice(len(coords), sample_size, replace=False)
        sample_coords = coords[indices]
    else:
        sample_coords = coords
    
    tree = cKDTree(sample_coords)
    
    # Find nearest neighbor distances
    distances, _ = tree.query(sample_coords, k=2)  # k=2 to exclude self
    nearest_distances = distances[:, 1]  # Second closest (first is self)
    
    return float(np.median(nearest_distances))

def determine_data_type(analysis: dict) -> str:
    """Determine the most likely data type based on analysis."""
    
    print(f"\n🔍 Data Type Analysis:")
    print(f"   Total points: {analysis['total_points']:,}")
    
    # Key indicators
    indicators = []
    
    if analysis['has_face_index']:
        print(f"   ✅ Has Face Index column")
        face_stats = analysis['face_index_stats']
        print(f"      - Face indices: {face_stats['min']} to {face_stats['max']}")
        print(f"      - Unique faces: {face_stats['unique_count']:,}")
        print(f"      - Sequential: {face_stats['sequential']}")
        indicators.append("FACE_BASED")
        
        if face_stats['unique_count'] == analysis['total_points']:
            indicators.append("ONE_POINT_PER_FACE")
    
    if analysis['has_area']:
        print(f"   ✅ Has Area data")
        area_stats = analysis['area_stats']
        print(f"      - Area range: {area_stats['min']:.2e} to {area_stats['max']:.2e} m²")
        print(f"      - Mean area: {area_stats['mean']:.2e} m²")
        print(f"      - All positive: {area_stats['all_positive']}")
        indicators.append("FACE_AREAS")
    
    if analysis['has_vdotn']:
        print(f"   ✅ Has VdotN (velocity dot normal)")
        indicators.append("FACE_NORMAL_VELOCITY")
    
    if analysis['has_pressure'] and analysis['has_velocity']:
        print(f"   ✅ Has pressure and velocity data")
        indicators.append("FLOW_FIELD")
    
    coord_stats = analysis['coordinate_stats']
    print(f"   📏 Coordinate ranges:")
    print(f"      - X: {coord_stats['x_range']*1000:.1f}mm")
    print(f"      - Y: {coord_stats['y_range']*1000:.1f}mm") 
    print(f"      - Z: {coord_stats['z_range']*1000:.1f}mm")
    print(f"      - Typical spacing: {coord_stats['mean_spacing']*1000:.3f}mm")
    
    # Determine data type
    if "FACE_BASED" in indicators and "ONE_POINT_PER_FACE" in indicators:
        if "FACE_AREAS" in indicators and "FACE_NORMAL_VELOCITY" in indicators:
            return "CELL_CENTERS (Face Centroids with Flow Data)"
        else:
            return "FACE_CENTROIDS (Geometric Centers)"
    elif "FACE_BASED" in indicators:
        return "FACE_INTEGRATION_POINTS (Multiple points per face)"
    elif analysis['total_points'] > 50000:  # Heuristic for mesh vertices
        return "MESH_VERTICES (Node-based data)"
    else:
        return "UNKNOWN (Insufficient indicators)"

def compare_with_stl_analysis(csv_analysis: dict, stl_vertices: int, stl_triangles: int) -> None:
    """Compare CSV analysis with STL mesh statistics."""
    
    print(f"\n🔗 STL-CSV Comparison:")
    print(f"   STL vertices: {stl_vertices:,}")
    print(f"   STL triangles: {stl_triangles:,}")
    print(f"   CSV points: {csv_analysis['total_points']:,}")
    
    # Compare counts
    vertex_ratio = csv_analysis['total_points'] / stl_vertices if stl_vertices > 0 else 0
    triangle_ratio = csv_analysis['total_points'] / stl_triangles if stl_triangles > 0 else 0
    
    print(f"   CSV/STL vertex ratio: {vertex_ratio:.2f}")
    print(f"   CSV/STL triangle ratio: {triangle_ratio:.2f}")
    
    # Interpretation
    if 0.8 <= vertex_ratio <= 1.2:
        print(f"   💡 CSV points ≈ STL vertices → Likely VERTEX data")
    elif 0.8 <= triangle_ratio <= 1.2:
        print(f"   💡 CSV points ≈ STL triangles → Likely FACE CENTER data")
    elif csv_analysis['total_points'] > stl_triangles:
        print(f"   💡 CSV points > STL triangles → Likely INTEGRATION POINTS or REFINED MESH")
    else:
        print(f"   💡 Unclear relationship - need deeper analysis")

def main():
    parser = argparse.ArgumentParser(description="Determine if CSV contains cell centers or vertices")
    parser.add_argument('--csv', required=True, type=Path, help='Path to CSV file')
    parser.add_argument('--stl', type=Path, help='Optional STL file for comparison')
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        print(f"❌ CSV file not found: {args.csv}")
        return
    
    # Analyze CSV
    csv_analysis = analyze_csv_structure(args.csv)
    data_type = determine_data_type(csv_analysis)
    
    # Compare with STL if provided
    if args.stl and args.stl.exists():
        try:
            from stl import mesh
            stl_mesh = mesh.Mesh.from_file(str(args.stl))
            vertices = stl_mesh.vectors.reshape(-1, 3)
            unique_vertices = np.unique(vertices, axis=0)
            triangles = len(stl_mesh.vectors)
            
            compare_with_stl_analysis(csv_analysis, len(unique_vertices), triangles)
        except ImportError:
            print(f"   ⚠️  Cannot read STL - numpy-stl not available")
        except Exception as e:
            print(f"   ⚠️  Error reading STL: {e}")
    
    # Final assessment
    print(f"\n" + "="*60)
    print(f"🎯 FINAL ASSESSMENT")
    print(f"="*60)
    print(f"Data Type: {data_type}")
    
    if "CELL_CENTERS" in data_type:
        print(f"\n💡 Implications:")
        print(f"   ✅ Each point represents the center of a mesh face/cell")
        print(f"   ✅ Flow quantities (pressure, velocity) are face-averaged values")
        print(f"   ✅ Area values represent actual face areas")
        print(f"   ✅ VdotN represents mass flux through face")
        print(f"   🔧 For geodesic distances: map to STL triangle centroids")
        
    elif "MESH_VERTICES" in data_type:
        print(f"\n💡 Implications:")
        print(f"   ✅ Each point represents a mesh node/vertex")
        print(f"   ✅ Can directly use STL connectivity")
        print(f"   ✅ Geodesic distances straightforward to implement")
        print(f"   🔧 Flow data interpolated to vertices")
    
    print(f"\n🎉 Analysis completed!")

if __name__ == "__main__":
    main() 