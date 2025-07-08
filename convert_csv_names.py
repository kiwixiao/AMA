#!/usr/bin/env python3
"""
Standalone script to convert CSV filenames from scientific notation to numerical pattern.

Usage: python convert_csv_names.py /path/to/xyz_tables_directory

This script:
1. Finds all CSV files with scientific notation timesteps
2. Sorts them chronologically by actual timestep value  
3. Renames them to table_1, table_2, table_3, etc.
4. Preserves the rest of the filename structure
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple
import argparse
import datetime


def extract_timestep_from_filename(file_path: Path) -> float:
    """
    Extract timestep from XYZ table filename, handling scientific notation.
    
    Args:
        file_path: Path to the XYZ table file
        
    Returns:
        Timestep as float
    """
    stem = file_path.stem
    
    # Pattern 1: Scientific notation (e.g., table_2.300000e+00)
    sci_pattern = r'table_([+-]?\d*\.?\d+[eE][+-]?\d+)'
    match = re.search(sci_pattern, stem)
    if match:
        return float(match.group(1))
    
    # Pattern 2: Decimal format (e.g., table_1.500000)
    decimal_pattern = r'table_(\d+\.\d+)'
    match = re.search(decimal_pattern, stem)
    if match:
        return float(match.group(1))
    
    # Pattern 3: Integer format (e.g., table_123)
    int_pattern = r'table_(\d+)'
    match = re.search(int_pattern, stem)
    if match:
        return float(match.group(1))
    
    raise ValueError(f"Could not extract timestep from filename: {file_path.name}")


def find_csv_files_with_timesteps(directory: Path) -> List[Path]:
    """Find all CSV files that appear to have timesteps in their names."""
    csv_files = []
    
    for file_path in directory.glob('*.csv'):
        # Check if it looks like an XYZ table file
        if 'XYZ_Internal_Table_table_' in file_path.name:
            csv_files.append(file_path)
    
    return csv_files


def convert_filenames(directory_path: str, dry_run: bool = False, output_file: str = None) -> None:
    """
    Convert CSV filenames from scientific notation to numerical pattern.
    
    Args:
        directory_path: Path to directory containing CSV files
        dry_run: If True, show what would be renamed without actually renaming
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ Error: Directory {directory_path} does not exist")
        return
    
    if not directory.is_dir():
        print(f"❌ Error: {directory_path} is not a directory")
        return
    
    print(f"🔍 Scanning directory: {directory}")
    
    # Find all CSV files with timesteps
    csv_files = find_csv_files_with_timesteps(directory)
    
    if not csv_files:
        print("❌ No CSV files with timestep patterns found")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Extract timesteps and sort chronologically
    timestep_file_pairs = []
    failed_files = []
    
    for file_path in csv_files:
        try:
            timestep = extract_timestep_from_filename(file_path)
            timestep_file_pairs.append((timestep, file_path))
        except ValueError as e:
            print(f"⚠️  Warning: {e}")
            failed_files.append(file_path)
    
    if failed_files:
        print(f"⚠️  Could not parse {len(failed_files)} files:")
        for file_path in failed_files:
            print(f"   - {file_path.name}")
    
    if not timestep_file_pairs:
        print("❌ No files with valid timestep patterns found")
        return
    
    # Sort by timestep value (chronological order)
    timestep_file_pairs.sort(key=lambda x: x[0])
    
    print(f"\n📊 Timestep range: {timestep_file_pairs[0][0]:.6f} to {timestep_file_pairs[-1][0]:.6f}")
    
    # Generate new filenames - preserving original timestep meaning
    rename_operations = []
    
    for timestep, old_file in timestep_file_pairs:
        # Extract the prefix and suffix parts
        old_name = old_file.name
        
        # Replace the timestep part with integer representation
        if 'XYZ_Internal_Table_table_' in old_name:
            # Find where the timestep part starts and ends
            prefix = old_name.split('table_')[0] + 'table_'
            suffix = '.csv'
            
            # Convert timestep to integer, preserving original meaning
            # If timestep < 10, it's likely in seconds, convert to milliseconds
            # If timestep >= 10, it's likely already in milliseconds
            if timestep < 10:
                timestep_int = int(timestep * 1000)  # Convert seconds to milliseconds
            else:
                timestep_int = int(timestep)  # Already in milliseconds
            
            new_name = f"{prefix}{timestep_int}{suffix}"
            new_file = old_file.parent / new_name
            
            rename_operations.append((old_file, new_file, timestep))
    
    # Show the conversion plan
    print(f"\n📋 Conversion plan ({len(rename_operations)} files):")
    print("Old filename → New filename (preserving timestep meaning)")
    print("-" * 80)
    
    for old_file, new_file, timestep in rename_operations[:10]:  # Show first 10
        # Extract the new timestep integer from the new filename
        new_timestep = int(new_file.stem.split('table_')[1])
        print(f"{old_file.name}")
        print(f"  → {new_file.name} (timestep: {timestep:.6f} → {new_timestep}ms)")
    
    if len(rename_operations) > 10:
        print(f"  ... and {len(rename_operations) - 10} more files")
    
    # Check for conflicts
    conflicts = []
    for old_file, new_file, timestep in rename_operations:
        if new_file.exists() and new_file != old_file:
            conflicts.append(new_file)
    
    if conflicts:
        print(f"\n❌ Error: {len(conflicts)} filename conflicts detected:")
        for conflict in conflicts:
            print(f"   - {conflict.name} already exists")
        print("Please resolve conflicts before proceeding.")
        return
    
    # Perform the renaming
    if dry_run:
        # Create output file name if not provided
        if output_file is None:
            folder_name = os.path.basename(directory_path.rstrip('/'))
            output_file = f"{folder_name}_conversion_plan.txt"
        
        print(f"\n🔍 DRY RUN: No files were actually renamed")
        print(f"💾 Saving detailed conversion plan to: {output_file}")
        
        # Write detailed conversion plan to file
        with open(output_file, 'w') as f:
            f.write("CSV FILENAME CONVERSION PLAN\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Directory: {directory_path}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files to convert: {len(rename_operations)}\n")
            f.write(f"Timestep range: {timestep_file_pairs[0][0]:.6f} to {timestep_file_pairs[-1][0]:.6f}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            f.write("ORIGINAL FILENAME                                          →  NEW FILENAME\n")
            f.write("(timestep value)                                              (preserving meaning)\n")
            f.write("-" * 80 + "\n")
            
            for i, (old_file, new_file, timestep) in enumerate(rename_operations, 1):
                # Extract the new timestep integer from the new filename
                new_timestep = int(new_file.stem.split('table_')[1])
                f.write(f"{i:4d}: {old_file.name:<50} →  {new_file.name}\n")
                f.write(f"      (timestep: {timestep:.6f} → {new_timestep}ms)\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"SUMMARY: {len(rename_operations)} files will be renamed preserving timestep meaning\n")
            f.write("Each new filename directly represents the original timestep value in milliseconds\n")
            f.write("Run without --dry-run to perform the actual conversion\n")
        
        print(f"✅ Conversion plan saved to: {output_file}")
        print(f"📊 Plan includes {len(rename_operations)} files")
        print("Run without --dry-run to perform the actual renaming")
    else:
        print(f"\n🔄 Renaming {len(rename_operations)} files...")
        
        success_count = 0
        for old_file, new_file, timestep in rename_operations:
            try:
                old_file.rename(new_file)
                success_count += 1
            except Exception as e:
                print(f"❌ Error renaming {old_file.name}: {e}")
        
        print(f"✅ Successfully renamed {success_count}/{len(rename_operations)} files")
        
        if success_count == len(rename_operations):
            print("\n🎉 All files converted successfully!")
            print("You can now run your main pipeline with simple numerical timesteps")
        else:
            print(f"\n⚠️  {len(rename_operations) - success_count} files failed to rename")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV filenames from scientific notation to numerical pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_csv_names.py /path/to/23mmeshOSAMRI007_xyz_tables
  python convert_csv_names.py /path/to/xyz_directory --dry-run
  python convert_csv_names.py ./23mmeshOSAMRI007_xyz_tables --dry-run --output-file my_plan.txt
  
This script will convert filenames like:
  XYZ_Internal_Table_table_1.000000e-03.csv → XYZ_Internal_Table_table_1.csv (1ms)
  XYZ_Internal_Table_table_2.000000e-03.csv → XYZ_Internal_Table_table_2.csv (2ms)
  XYZ_Internal_Table_table_1.000000e-02.csv → XYZ_Internal_Table_table_10.csv (10ms)
  
Files preserve their original timestep meaning in the new numerical names.
The --dry-run option saves a detailed conversion plan to a text file for review.
        """
    )
    
    parser.add_argument('directory', 
                       help='Path to directory containing CSV files to rename')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be renamed without actually renaming files')
    parser.add_argument('--output-file', type=str,
                       help='Custom output file for dry-run conversion plan (default: auto-generated)')
    
    args = parser.parse_args()
    
    try:
        convert_filenames(args.directory, dry_run=args.dry_run, output_file=args.output_file)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 