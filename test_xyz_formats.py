#!/usr/bin/env python3
"""
Test script to demonstrate XYZ file format handling for different mesh variants.

This script shows how the pipeline handles different naming conventions and time intervals.
"""

from src.utils.file_processing import (
    extract_timestep_from_filename,
    detect_time_unit,
    get_xyz_file_info,
    find_closest_xyz_file
)
from pathlib import Path

def test_timestep_extraction():
    """Test timestep extraction from different filename formats."""
    print("🧪 Testing timestep extraction from different filename formats...")
    
    # Test cases for different naming conventions
    test_cases = [
        # Integer format (milliseconds)
        ("XYZ_Internal_Table_table_2387.csv", 2387.0),
        ("XYZ_Internal_Table_table_100.csv", 100.0),
        ("XYZ_Internal_Table_table_1.csv", 1.0),
        
        # Scientific notation format (seconds)
        ("XYZ_Internal_Table_table_2.300000e+00.csv", 2.3),
        ("XYZ_Internal_Table_table_1.500000e+00.csv", 1.5),
        ("XYZ_Internal_Table_table_7.500000e-01.csv", 0.75),
        ("XYZ_Internal_Table_table_2.500000e-01.csv", 0.25),
        
        # Decimal format
        ("XYZ_Internal_Table_table_1.500000.csv", 1.5),
        ("XYZ_Internal_Table_table_2.250000.csv", 2.25),
    ]
    
    for filename, expected in test_cases:
        try:
            result = extract_timestep_from_filename(Path(filename))
            status = "✅" if abs(result - expected) < 1e-6 else "❌"
            print(f"  {status} {filename} → {result} (expected: {expected})")
        except Exception as e:
            print(f"  ❌ {filename} → ERROR: {e}")

def analyze_existing_subjects():
    """Analyze existing subjects in the directory."""
    print("\n🔍 Analyzing existing mesh variants...")
    
    # List of subjects to analyze
    subjects = ["2mmeshOSAMRI007", "23mmeshOSAMRI007", "less1mmesh_OSAMRI007"]
    
    for subject in subjects:
        print(f"\n📋 Subject: {subject}")
        file_info = get_xyz_file_info(subject)
        
        if file_info['files_found'] == 0:
            print(f"  ❌ No XYZ files found")
            continue
        
        print(f"  📁 Directory: {file_info['directory']}")
        print(f"  📊 Files found: {file_info['files_found']}")
        print(f"  📊 Files parsed: {file_info['files_parsed']}")
        print(f"  🕐 Time unit: {file_info['time_unit']}")
        print(f"  📝 Naming convention: {file_info['naming_convention']}")
        print(f"  ⏱️  Time range: {file_info['time_range']}")
        print(f"  📏 Intervals: {file_info['intervals']}")
        
        if file_info['failed_files']:
            print(f"  ⚠️  Failed to parse: {file_info['failed_files']}")
        
        # Show sample timesteps
        if file_info['timesteps']:
            sample_timesteps = file_info['timesteps'][:5]
            if len(file_info['timesteps']) > 5:
                sample_timesteps.append("...")
                sample_timesteps.extend(file_info['timesteps'][-2:])
            print(f"  🔢 Sample timesteps: {sample_timesteps}")

def test_file_finding():
    """Test finding closest XYZ files for visualization."""
    print("\n🎯 Testing file finding for visualization...")
    
    subjects = ["2mmeshOSAMRI007", "23mmeshOSAMRI007"]
    target_timesteps = [50, 100, 1000, 2000]  # milliseconds
    
    for subject in subjects:
        print(f"\n📋 Subject: {subject}")
        for target in target_timesteps:
            closest_file = find_closest_xyz_file(subject, target)
            if closest_file:
                try:
                    actual_timestep = extract_timestep_from_filename(closest_file)
                    print(f"  🎯 Target {target}ms → {closest_file.name} (actual: {actual_timestep})")
                except:
                    print(f"  🎯 Target {target}ms → {closest_file.name} (parsing error)")
            else:
                print(f"  ❌ Target {target}ms → No file found")

def demonstrate_usage_scenarios():
    """Demonstrate usage scenarios for the pipeline."""
    print("\n🎯 Usage Scenarios with Different Mesh Variants:")
    
    print("\n1. 📊 Integer timestep format (2mmeshOSAMRI007):")
    print("   Files: table_2387.csv, table_2386.csv, ...")
    print("   Time unit: Likely milliseconds")
    print("   Usage: python src/main.py --highlight-patches --subject 2mmeshOSAMRI007 --patch-timestep 100")
    print("   → System finds closest file to 100ms")
    
    print("\n2. 🔬 Scientific notation format (23mmeshOSAMRI007):")
    print("   Files: table_2.300000e+00.csv, table_2.250000e+00.csv, ...")
    print("   Time unit: Likely seconds")
    print("   Usage: python src/main.py --raw-surface --subject 23mmeshOSAMRI007 --surface-timestep 1000")
    print("   → System converts 1000ms to 1.0s and finds closest file")
    
    print("\n3. 🔄 Automatic handling:")
    print("   • Pipeline detects naming convention automatically")
    print("   • Converts between time units as needed")
    print("   • Finds closest available timestep")
    print("   • Works with visualization modes seamlessly")

if __name__ == "__main__":
    print("🚀 XYZ File Format Handling Test Suite")
    print("=" * 60)
    
    test_timestep_extraction()
    analyze_existing_subjects()
    test_file_finding()
    demonstrate_usage_scenarios()
    
    print("\n✅ Test suite completed!")
    print("\n💡 Key Benefits:")
    print("   • Handles multiple naming conventions automatically")
    print("   • Detects time units (seconds vs milliseconds)")
    print("   • Finds closest files for visualization")
    print("   • Works with non-uniform time intervals")
    print("   • Seamless integration with existing pipeline") 