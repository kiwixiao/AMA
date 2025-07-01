#!/usr/bin/env python3
"""
Standalone script to add coordinates to tracking locations JSON files.

This script takes a tracking locations JSON file (with patch numbers and face indices)
and adds the actual XYZ coordinates by looking them up in the corresponding
patched XYZ table files.

Usage:
    python src/add_coordinates_to_tracking.py --subject 2mmeshOSAMRI007 --input 2mmeshOSAMRI007_tracking_locations.json --output 2mmeshOSAMRI007_tracking_locations_with_coords.json

The script will:
1. Load the input tracking locations JSON file
2. Find the first available patched XYZ table for the subject
3. Look up coordinates for each patch/face combination
4. Add coordinates field to each location
5. Save the enhanced JSON file
"""

import json
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

def find_first_patched_xyz_file(subject_name: str) -> Optional[Path]:
    """Find the first available patched XYZ table file for the subject."""
    patched_dir = Path(f"{subject_name}_xyz_tables_with_patches")
    
    if not patched_dir.exists():
        print(f"❌ Error: Patched XYZ directory not found: {patched_dir}")
        return None
    
    # Look for patched CSV files
    csv_files = list(patched_dir.glob("patched_*.csv"))
    if not csv_files:
        print(f"❌ Error: No patched CSV files found in {patched_dir}")
        return None
    
    # Sort to get the first one (by filename)
    csv_files.sort()
    first_file = csv_files[0]
    
    print(f"📁 Using reference file: {first_file}")
    return first_file

def extract_coordinates_from_xyz_table(xyz_file: Path, patch_number: int, face_index: int) -> Optional[Tuple[float, float, float]]:
    """Extract XYZ coordinates for a specific patch and face index."""
    try:
        print(f"   Looking up Patch {patch_number}, Face {face_index}...")
        
        # Read the CSV file with low memory to handle large files
        df = pd.read_csv(xyz_file, low_memory=False)
        
        # Find the matching row
        match = df[(df['Patch Number'] == patch_number) & (df['Face Index'] == face_index)]
        
        if len(match) == 0:
            print(f"   ❌ No match found for Patch {patch_number}, Face {face_index}")
            return None
        
        if len(match) > 1:
            print(f"   ⚠️  Multiple matches found, using first one")
        
        # Extract coordinates
        row = match.iloc[0]
        coords = (float(row['X (m)']), float(row['Y (m)']), float(row['Z (m)']))
        print(f"   ✅ Found coordinates: ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f})")
        
        return coords
        
    except Exception as e:
        print(f"   ❌ Error extracting coordinates: {e}")
        return None

def find_closest_point_by_coordinates(xyz_file: Path, target_coords: Tuple[float, float, float], 
                                     tolerance_levels: List[float] = [0.0005, 0.001, 0.0015]) -> Optional[Tuple[int, int, float]]:
    """
    Find the closest surface point to target coordinates within tolerance.
    
    Args:
        xyz_file: Path to patched XYZ table
        target_coords: Target (x, y, z) coordinates in meters
        tolerance_levels: List of tolerance levels in meters [0.5mm, 1mm, 1.5mm]
        
    Returns:
        Tuple of (patch_number, face_index, distance) if found, None otherwise
    """
    try:
        print(f"   🔍 Searching for closest point to ({target_coords[0]:.6f}, {target_coords[1]:.6f}, {target_coords[2]:.6f})")
        
        # Read the CSV file
        df = pd.read_csv(xyz_file, low_memory=False)
        
        # Calculate distances to all points
        target_x, target_y, target_z = target_coords
        df['distance'] = np.sqrt(
            (df['X (m)'] - target_x)**2 + 
            (df['Y (m)'] - target_y)**2 + 
            (df['Z (m)'] - target_z)**2
        )
        
        # Try progressive tolerance levels
        for tolerance in tolerance_levels:
            within_tolerance = df[df['distance'] <= tolerance]
            
            if len(within_tolerance) > 0:
                # Find the closest point within tolerance
                closest_point = within_tolerance.loc[within_tolerance['distance'].idxmin()]
                
                patch_number = int(closest_point['Patch Number'])
                face_index = int(closest_point['Face Index'])
                distance = float(closest_point['distance'])
                
                print(f"   ✅ Found match within {tolerance*1000:.1f}mm tolerance:")
                print(f"      Patch {patch_number}, Face {face_index}, Distance: {distance*1000:.3f}mm")
                print(f"      Coordinates: ({closest_point['X (m)']:.6f}, {closest_point['Y (m)']:.6f}, {closest_point['Z (m)']:.6f})")
                
                return (patch_number, face_index, distance)
        
        # No match found within any tolerance level
        min_distance = df['distance'].min()
        print(f"   ❌ No match found within tolerance levels {[t*1000 for t in tolerance_levels]}mm")
        print(f"      Closest point distance: {min_distance*1000:.3f}mm")
        
        return None
        
    except Exception as e:
        print(f"   ❌ Error finding closest point: {e}")
        return None

def update_patch_face_indices_from_coordinates(reference_json_file: Path, target_subject: str, 
                                              output_file: Path = None) -> bool:
    """
    Update patch and face indices for target subject using reference coordinates.
    
    Args:
        reference_json_file: Path to JSON file with reference coordinates
        target_subject: Target subject name (e.g., "less1mmeshOSAMRI007")
        output_file: Output file path (default: target_subject_tracking_locations.json)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🎯 Updating patch/face indices for {target_subject} using reference coordinates")
    print(f"📂 Reference file: {reference_json_file}")
    
    # Load reference JSON with coordinates
    try:
        with open(reference_json_file, 'r') as f:
            reference_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading reference file: {e}")
        return False
    
    # Validate that coordinates exist
    if 'coordinate_metadata' not in reference_data:
        print(f"❌ Error: Reference file doesn't contain coordinate metadata")
        return False
    
    reference_subject = reference_data['coordinate_metadata'].get('reference_subject', 'unknown')
    print(f"📍 Reference subject: {reference_subject}")
    
    # Find target XYZ file
    target_xyz_file = find_first_patched_xyz_file(target_subject)
    if target_xyz_file is None:
        return False
    
    # Create output structure based on reference
    output_data = {
        'locations': [],
        'combinations': reference_data.get('combinations', []),
        'coordinate_metadata': {
            'reference_subject': reference_subject,
            'target_subject': target_subject,
            'target_file': str(target_xyz_file.name),
            'tolerance_levels_mm': [0.5, 1.0, 1.5],
            'coordinate_units': 'meters'
        }
    }
    
    # Process each location
    locations_updated = 0
    locations_failed = 0
    tolerance_stats = {0.5: 0, 1.0: 0, 1.5: 0, 'failed': 0}
    
    for location in reference_data.get('locations', []):
        description = location.get('description', 'Unknown')
        ref_coords = location.get('coordinates')
        
        print(f"\n🔍 Processing: {description}")
        
        if ref_coords is None:
            print(f"   ⚠️  Skipping - no reference coordinates found")
            locations_failed += 1
            tolerance_stats['failed'] += 1
            continue
        
        # Find closest point in target mesh
        result = find_closest_point_by_coordinates(target_xyz_file, tuple(ref_coords))
        
        if result is not None:
            patch_number, face_index, distance = result
            
            # Create updated location entry
            updated_location = {
                'patch_number': patch_number,
                'description': description,
                'face_indices': [face_index],
                'coordinates': ref_coords,  # Keep reference coordinates
                'reference_coordinates': ref_coords,
                'target_coordinates': ref_coords,  # Will be updated with actual target coords
                'distance_from_reference_mm': distance * 1000,
                'reference_subject': reference_subject,
                'target_subject': target_subject
            }
            
            # Update tolerance statistics
            distance_mm = distance * 1000
            if distance_mm <= 0.5:
                tolerance_stats[0.5] += 1
            elif distance_mm <= 1.0:
                tolerance_stats[1.0] += 1
            elif distance_mm <= 1.5:
                tolerance_stats[1.5] += 1
            
            output_data['locations'].append(updated_location)
            locations_updated += 1
            
        else:
            locations_failed += 1
            tolerance_stats['failed'] += 1
    
    # Update metadata with results
    output_data['coordinate_metadata'].update({
        'locations_updated': locations_updated,
        'locations_failed': locations_failed,
        'tolerance_statistics': {
            'within_0.5mm': tolerance_stats[0.5],
            'within_1.0mm': tolerance_stats[1.0], 
            'within_1.5mm': tolerance_stats[1.5],
            'failed': tolerance_stats['failed']
        }
    })
    
    # Determine output file
    if output_file is None:
        output_file = Path(f"{target_subject}_tracking_locations.json")
    
    # Save updated JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Updated tracking file saved: {output_file}")
        print(f"📊 Summary:")
        print(f"   • Target subject: {target_subject}")
        print(f"   • Locations updated: {locations_updated}")
        print(f"   • Locations failed: {locations_failed}")
        print(f"   • Within 0.5mm: {tolerance_stats[0.5]}")
        print(f"   • Within 1.0mm: {tolerance_stats[1.0]}")
        print(f"   • Within 1.5mm: {tolerance_stats[1.5]}")
        print(f"   • Failed matches: {tolerance_stats['failed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving output file: {e}")
        return False

def add_coordinates_to_tracking_file(input_file: Path, output_file: Path, subject_name: str) -> bool:
    """
    Add coordinates to tracking locations JSON file.
    
    Args:
        input_file: Path to input tracking locations JSON
        output_file: Path to save enhanced JSON with coordinates
        subject_name: Subject name to find XYZ tables
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🎯 Adding coordinates to tracking file: {input_file}")
    print(f"📂 Subject: {subject_name}")
    
    # Load input JSON
    try:
        with open(input_file, 'r') as f:
            tracking_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return False
    
    # Find reference XYZ file
    xyz_file = find_first_patched_xyz_file(subject_name)
    if xyz_file is None:
        return False
    
    # Process each location
    locations_updated = 0
    locations_failed = 0
    
    for i, location in enumerate(tracking_data.get('locations', [])):
        patch_number = location.get('patch_number')
        face_indices = location.get('face_indices', [])
        description = location.get('description', f'Location {i+1}')
        
        print(f"\n🔍 Processing: {description}")
        
        if patch_number is None or not face_indices:
            print(f"   ⚠️  Skipping - missing patch_number or face_indices")
            locations_failed += 1
            continue
        
        # Use the first face index
        face_index = face_indices[0]
        
        # Extract coordinates
        coords = extract_coordinates_from_xyz_table(xyz_file, patch_number, face_index)
        
        if coords is not None:
            # Add coordinates to the location
            location['coordinates'] = list(coords)
            location['reference_file'] = str(xyz_file.name)
            location['reference_subject'] = subject_name
            locations_updated += 1
        else:
            locations_failed += 1
    
    # Add metadata
    tracking_data['coordinate_metadata'] = {
        'reference_subject': subject_name,
        'reference_file': str(xyz_file.name),
        'locations_updated': locations_updated,
        'locations_failed': locations_failed,
        'coordinate_units': 'meters'
    }
    
    # Save enhanced JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"\n✅ Enhanced tracking file saved: {output_file}")
        print(f"📊 Summary:")
        print(f"   • Locations updated: {locations_updated}")
        print(f"   • Locations failed: {locations_failed}")
        print(f"   • Reference file: {xyz_file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving output file: {e}")
        return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Add XYZ coordinates to tracking locations JSON file or update patch/face indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add coordinates to 2mmesh tracking file
  python src/add_coordinates_to_tracking.py --subject 2mmeshOSAMRI007 --input 2mmeshOSAMRI007_tracking_locations.json
  
  # Update patch/face indices for less1mmesh using 2mmesh coordinates
  python src/add_coordinates_to_tracking.py --update-target less1mmeshOSAMRI007 --reference 2mmeshOSAMRI007_tracking_locations_with_coords.json
  
  # Update patch/face indices for 23mmesh using 2mmesh coordinates  
  python src/add_coordinates_to_tracking.py --update-target 23mmeshOSAMRI007 --reference 2mmeshOSAMRI007_tracking_locations_with_coords.json
        """
    )
    
    # Create mutually exclusive groups for different modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # Mode 1: Add coordinates to existing tracking file
    mode_group.add_argument('--subject',
                           help='Subject name to add coordinates for (e.g., 2mmeshOSAMRI007)')
    
    # Mode 2: Update patch/face indices using reference coordinates
    mode_group.add_argument('--update-target',
                           help='Target subject to update patch/face indices for (e.g., less1mmeshOSAMRI007)')
    
    # Common arguments
    parser.add_argument('--input', type=Path,
                        help='Input tracking locations JSON file (for --subject mode)')
    
    parser.add_argument('--reference', type=Path,
                        help='Reference JSON file with coordinates (for --update-target mode)')
    
    parser.add_argument('--output', type=Path,
                        help='Output file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Mode 1: Add coordinates
    if args.subject:
        if not args.input:
            print(f"❌ Error: --input required when using --subject")
            sys.exit(1)
        
        if not args.input.exists():
            print(f"❌ Error: Input file not found: {args.input}")
            sys.exit(1)
        
        # Determine output file
        if args.output is None:
            stem = args.input.stem
            suffix = args.input.suffix
            args.output = args.input.parent / f"{stem}_with_coords{suffix}"
        
        # Run coordinate extraction
        success = add_coordinates_to_tracking_file(args.input, args.output, args.subject)
    
    # Mode 2: Update patch/face indices
    elif args.update_target:
        if not args.reference:
            print(f"❌ Error: --reference required when using --update-target")
            sys.exit(1)
        
        if not args.reference.exists():
            print(f"❌ Error: Reference file not found: {args.reference}")
            sys.exit(1)
        
        # Determine output file
        if args.output is None:
            args.output = Path(f"{args.update_target}_tracking_locations.json")
        
        # Run patch/face index update
        success = update_patch_face_indices_from_coordinates(args.reference, args.update_target, args.output)
    
    if success:
        print(f"\n🎉 Success!")
        sys.exit(0)
    else:
        print(f"\n💥 Failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 