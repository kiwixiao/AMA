#!/usr/bin/env python3
"""
Coordinate Mapper for Painted Regions

This module maps painted STL surface regions to CFD data points in CSV files.
It uses spatial mapping to find corresponding CFD data for painted surface regions.

Usage:
    python -m src.surface_painting.coordinate_mapper --painted-coords region_coords.json --csv-data data.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree
import argparse


class CoordinateMapper:
    """Maps painted surface coordinates to CFD data points."""
    
    def __init__(self, tolerance_levels: List[float] = None):
        """
        Initialize coordinate mapper.
        
        Args:
            tolerance_levels: Distance tolerances in meters for mapping (default: [0.001, 0.002, 0.005])
        """
        if tolerance_levels is None:
            tolerance_levels = [0.001, 0.002, 0.005]  # 1mm, 2mm, 5mm
        
        self.tolerance_levels = tolerance_levels
        self.mapping_results = {}
        
    def load_painted_coordinates(self, coords_file: Path) -> Dict:
        """
        Load painted region coordinates from JSON file.
        
        Args:
            coords_file: Path to painted coordinates JSON file
            
        Returns:
            Dictionary with painted region data
        """
        print(f"Loading painted coordinates from: {coords_file}")
        
        with open(coords_file, 'r') as f:
            coords_data = json.load(f)
        
        print(f"Found {len(coords_data['regions'])} painted regions:")
        for region_name, region_data in coords_data['regions'].items():
            print(f"  {region_name}: {region_data['point_count']:,} points")
        
        return coords_data
    
    def load_cfd_data(self, csv_file: Path) -> pd.DataFrame:
        """
        Load CFD data from CSV file.
        
        Args:
            csv_file: Path to CSV file with CFD data
            
        Returns:
            DataFrame with CFD data
        """
        print(f"Loading CFD data from: {csv_file}")
        
        df = pd.read_csv(csv_file, low_memory=False)
        
        # Check for required columns
        required_cols = ['X (m)', 'Y (m)', 'Z (m)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"Loaded CFD data: {len(df):,} points")
        print(f"Available columns: {list(df.columns)}")
        
        return df
    
    def map_region_to_cfd(self, region_coords: np.ndarray, cfd_coords: np.ndarray, 
                         cfd_data: pd.DataFrame, tolerance: float) -> Dict:
        """
        Map painted region coordinates to CFD data points.
        
        Args:
            region_coords: Painted region coordinates (N, 3)
            cfd_coords: CFD point coordinates (M, 3)
            cfd_data: Full CFD DataFrame
            tolerance: Distance tolerance in meters
            
        Returns:
            Dictionary with mapping results
        """
        # Build KDTree for efficient spatial search
        tree = cKDTree(cfd_coords)
        
        # Find closest CFD points for each painted point
        distances, indices = tree.query(region_coords, distance_upper_bound=tolerance)
        
        # Filter out points beyond tolerance (scipy returns inf for these)
        valid_mask = distances < tolerance
        valid_indices = indices[valid_mask]
        valid_distances = distances[valid_mask]
        
        if len(valid_indices) == 0:
            return {
                'matched_points': 0,
                'matched_indices': [],
                'mean_distance': None,
                'cfd_subset': pd.DataFrame()
            }
        
        # Remove duplicates (multiple painted points might map to same CFD point)
        unique_indices = list(set(valid_indices))
        
        # Extract corresponding CFD data
        cfd_subset = cfd_data.iloc[unique_indices].copy()
        
        return {
            'matched_points': len(unique_indices),
            'matched_indices': unique_indices,
            'mean_distance': float(valid_distances.mean()),
            'max_distance': float(valid_distances.max()),
            'min_distance': float(valid_distances.min()),
            'cfd_subset': cfd_subset
        }
    
    def map_all_regions(self, coords_data: Dict, cfd_data: pd.DataFrame) -> Dict:
        """
        Map all painted regions to CFD data.
        
        Args:
            coords_data: Painted coordinates data
            cfd_data: CFD DataFrame
            
        Returns:
            Dictionary with mapping results for all regions
        """
        print(f"\nMapping painted regions to CFD data...")
        
        # Extract CFD coordinates
        cfd_coords = cfd_data[['X (m)', 'Y (m)', 'Z (m)']].values
        
        results = {}
        
        for region_name, region_data in coords_data['regions'].items():
            print(f"\nProcessing {region_name}...")
            
            # Get painted coordinates for this region
            region_coords = np.array(region_data['coordinates'])
            
            # Try different tolerance levels
            region_results = {}
            best_tolerance = None
            best_match_count = 0
            
            for tolerance in self.tolerance_levels:
                mapping_result = self.map_region_to_cfd(
                    region_coords, cfd_coords, cfd_data, tolerance
                )
                
                match_count = mapping_result['matched_points']
                region_results[f'{tolerance*1000:.1f}mm'] = mapping_result
                
                print(f"  Tolerance {tolerance*1000:.1f}mm: {match_count:,} CFD points matched")
                
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_tolerance = tolerance
            
            # Store results
            results[region_name] = {
                'painted_points': region_data['point_count'],
                'tolerance_results': region_results,
                'best_tolerance': f'{best_tolerance*1000:.1f}mm' if best_tolerance else None,
                'best_match_count': best_match_count,
                'centroid': region_data['centroid'],
                'bounds': region_data['bounds']
            }
            
            print(f"  Best result: {best_match_count:,} matches at {best_tolerance*1000:.1f}mm tolerance")
        
        return results
    
    def compute_regional_statistics(self, mapping_results: Dict) -> Dict:
        """
        Compute statistics for mapped regions.
        
        Args:
            mapping_results: Results from map_all_regions
            
        Returns:
            Dictionary with regional statistics
        """
        print(f"\nComputing regional statistics...")
        
        stats_results = {}
        
        for region_name, region_results in mapping_results.items():
            print(f"\nAnalyzing {region_name}...")
            
            region_stats = {}
            
            # Analyze each tolerance level
            for tolerance_key, tolerance_data in region_results['tolerance_results'].items():
                if tolerance_data['matched_points'] == 0:
                    continue
                
                cfd_subset = tolerance_data['cfd_subset']
                
                # Compute statistics for key quantities
                stats = {
                    'point_count': len(cfd_subset),
                    'mapping_distance': {
                        'mean': tolerance_data['mean_distance'] * 1000,  # Convert to mm
                        'min': tolerance_data['min_distance'] * 1000,
                        'max': tolerance_data['max_distance'] * 1000
                    }
                }
                
                # Analyze CFD quantities if available
                cfd_quantities = {
                    'VdotN': 'v⃗·n⃗ (m/s)',
                    'Total Pressure (Pa)': 'Total Pressure (Pa)',
                    'Static Pressure (Pa)': 'Static Pressure (Pa)',
                    'Velocity: Magnitude (m/s)': 'Velocity Magnitude (m/s)',
                    'Area: Magnitude (m^2)': 'Face Area (m²)'
                }
                
                for col_name, display_name in cfd_quantities.items():
                    if col_name in cfd_subset.columns:
                        values = cfd_subset[col_name].values
                        stats[display_name] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'median': float(np.median(values))
                        }
                
                # Store time information if available
                if 'Time (s)' in cfd_subset.columns:
                    times = cfd_subset['Time (s)'].values
                    stats['Time Range (s)'] = {
                        'min': float(times.min()),
                        'max': float(times.max()),
                        'span': float(times.max() - times.min())
                    }
                
                region_stats[tolerance_key] = stats
                print(f"  {tolerance_key}: {stats['point_count']:,} points")
            
            stats_results[region_name] = region_stats
        
        return stats_results
    
    def export_mapping_results(self, mapping_results: Dict, stats_results: Dict, 
                              output_file: Path = None):
        """
        Export mapping and statistics results to JSON.
        
        Args:
            mapping_results: Results from map_all_regions
            stats_results: Results from compute_regional_statistics
            output_file: Output file path (optional)
        """
        if output_file is None:
            output_file = Path("region_mapping_results.json")
        
        # Prepare export data (excluding large DataFrames)
        export_data = {
            'summary': {
                'total_regions': len(mapping_results),
                'tolerance_levels_mm': [t*1000 for t in self.tolerance_levels]
            },
            'regions': {}
        }
        
        for region_name, region_results in mapping_results.items():
            export_region = {
                'painted_points': region_results['painted_points'],
                'best_tolerance': region_results['best_tolerance'],
                'best_match_count': region_results['best_match_count'],
                'centroid': region_results['centroid'],
                'bounds': region_results['bounds'],
                'tolerance_summary': {}
            }
            
            # Add tolerance summary (without DataFrame)
            for tolerance_key, tolerance_data in region_results['tolerance_results'].items():
                export_region['tolerance_summary'][tolerance_key] = {
                    'matched_points': tolerance_data['matched_points'],
                    'mean_distance_mm': tolerance_data.get('mean_distance', 0) * 1000 if tolerance_data.get('mean_distance') else 0
                }
            
            # Add statistics
            if region_name in stats_results:
                export_region['statistics'] = stats_results[region_name]
            
            export_data['regions'][region_name] = export_region
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Exported mapping results to: {output_file}")
        
        # Print summary
        print(f"\n📊 Mapping Summary:")
        for region_name, region_data in export_data['regions'].items():
            best_count = region_data['best_match_count']
            best_tol = region_data['best_tolerance']
            painted_count = region_data['painted_points']
            print(f"  {region_name}: {painted_count:,} painted → {best_count:,} CFD points (@ {best_tol})")
    
    def export_regional_cfd_data(self, mapping_results: Dict, output_dir: Path = None):
        """
        Export CFD data subsets for each region.
        
        Args:
            mapping_results: Results from map_all_regions
            output_dir: Output directory for CSV files (optional)
        """
        if output_dir is None:
            output_dir = Path("regional_cfd_data")
        
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Exporting regional CFD data to: {output_dir}")
        
        for region_name, region_results in mapping_results.items():
            # Use best tolerance result
            best_tolerance = region_results['best_tolerance']
            if best_tolerance and best_tolerance in region_results['tolerance_results']:
                tolerance_data = region_results['tolerance_results'][best_tolerance]
                cfd_subset = tolerance_data['cfd_subset']
                
                if len(cfd_subset) > 0:
                    output_file = output_dir / f"{region_name}_{best_tolerance}_cfd_data.csv"
                    cfd_subset.to_csv(output_file, index=False)
                    print(f"  {region_name}: {len(cfd_subset):,} points → {output_file}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Map painted STL regions to CFD data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Map painted regions to CFD data
  python -m src.surface_painting.coordinate_mapper --painted-coords out_0100.000_ds_10_l2_aBE0.001_be_0.001_painted_coordinates.json --csv-data less1mmeshOSAMRI007_xyz_tables_with_patches/patched_XYZ_Internal_Table_table_100.csv
  
  # Custom tolerance levels
  python -m src.surface_painting.coordinate_mapper --painted-coords coords.json --csv-data data.csv --tolerances 0.5 1.0 2.0
        """
    )
    
    parser.add_argument('--painted-coords', required=True, type=Path,
                        help='Path to painted coordinates JSON file')
    
    parser.add_argument('--csv-data', required=True, type=Path,
                        help='Path to CFD CSV data file')
    
    parser.add_argument('--tolerances', nargs='+', type=float, default=[1.0, 2.0, 5.0],
                        help='Distance tolerances in millimeters (default: 1.0 2.0 5.0)')
    
    parser.add_argument('--output', type=Path, default=None,
                        help='Output file for mapping results (default: region_mapping_results.json)')
    
    parser.add_argument('--export-cfd', action='store_true',
                        help='Export CFD data subsets for each region')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.painted_coords.exists():
        print(f"Error: Painted coordinates file not found: {args.painted_coords}")
        return
    
    if not args.csv_data.exists():
        print(f"Error: CFD data file not found: {args.csv_data}")
        return
    
    # Convert tolerances to meters
    tolerance_levels = [t / 1000.0 for t in args.tolerances]
    
    try:
        # Initialize mapper
        mapper = CoordinateMapper(tolerance_levels)
        
        # Load data
        coords_data = mapper.load_painted_coordinates(args.painted_coords)
        cfd_data = mapper.load_cfd_data(args.csv_data)
        
        # Perform mapping
        mapping_results = mapper.map_all_regions(coords_data, cfd_data)
        
        # Compute statistics
        stats_results = mapper.compute_regional_statistics(mapping_results)
        
        # Export results
        mapper.export_mapping_results(mapping_results, stats_results, args.output)
        
        # Export CFD data subsets if requested
        if args.export_cfd:
            mapper.export_regional_cfd_data(mapping_results)
        
        print(f"\n🎉 Coordinate mapping completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 