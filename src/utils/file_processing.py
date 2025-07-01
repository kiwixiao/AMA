import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
import re

def extract_base_subject(subject_name: str) -> str:
    """
    Extract the base subject name from a mesh variant subject name.
    
    Examples:
        '2mmeshOSAMRI007' -> 'OSAMRI007'
        'less1mmesh_OSAMRI007' -> 'OSAMRI007'
        'fineMeshOSAMRI007' -> 'OSAMRI007'
        'OSAMRI007' -> 'OSAMRI007' (unchanged)
    
    Args:
        subject_name: Full subject name (potentially with mesh prefix)
        
    Returns:
        Base subject name without mesh variant prefix
    """
    # Common mesh variant prefixes
    mesh_prefixes = [
        r'^\d+mmesh',           # e.g., '2mmesh', '1mmesh'
        r'^less\d+mmesh_?',     # e.g., 'less1mmesh_', 'less2mmesh'
        r'^fine[Mm]esh_?',      # e.g., 'fineMesh_', 'finemesh'
        r'^coarse[Mm]esh_?',    # e.g., 'coarseMesh_', 'coarsemesh'
        r'^high[Rr]es_?',       # e.g., 'highRes_', 'highres'
        r'^low[Rr]es_?',        # e.g., 'lowRes_', 'lowres'
    ]
    
    for prefix_pattern in mesh_prefixes:
        match = re.match(prefix_pattern, subject_name)
        if match:
            # Remove the matched prefix
            base_name = subject_name[match.end():]
            # Remove leading underscore if present
            if base_name.startswith('_'):
                base_name = base_name[1:]
            return base_name
    
    # No mesh prefix found, return original name
    return subject_name

def find_flow_profile_file(subject_name: str) -> Optional[Path]:
    """
    Find the appropriate flow profile file for a subject, with smart fallback.
    
    Search order:
    1. Exact match: {subject_name}FlowProfile.csv
    2. Base subject: {base_subject}FlowProfile.csv
    3. Smoothed versions of the above
    
    Args:
        subject_name: Subject name (potentially with mesh variant)
        
    Returns:
        Path to flow profile file, or None if not found
    """
    base_subject = extract_base_subject(subject_name)
    
    # Define search candidates in priority order
    candidates = [
        f"{subject_name}FlowProfile.csv",
        f"{subject_name}FlowProfile_smoothed.csv",
        f"{base_subject}FlowProfile.csv",
        f"{base_subject}FlowProfile_smoothed.csv",
    ]
    
    print(f"🔍 Searching for flow profile for subject: {subject_name}")
    if base_subject != subject_name:
        print(f"   Base subject detected: {base_subject}")
    
    for candidate in candidates:
        file_path = Path(candidate)
        if file_path.exists():
            if candidate.startswith(subject_name):
                print(f"✅ Found exact match: {candidate}")
            else:
                print(f"✅ Found base subject match: {candidate}")
                print(f"   Using {base_subject} flow profile for {subject_name}")
            return file_path
    
    print(f"❌ No flow profile found for {subject_name}")
    print(f"   Searched for: {candidates}")
    return None

def find_tracking_locations_file(subject_name: str) -> Optional[Path]:
    """
    Find the appropriate tracking locations file for a subject, with smart fallback.
    
    Search order:
    1. Exact match: {subject_name}_tracking_locations.json
    2. Base subject: {base_subject}_tracking_locations.json
    3. Generic: tracking_locations.json
    
    Args:
        subject_name: Subject name (potentially with mesh variant)
        
    Returns:
        Path to tracking locations file, or None if not found
    """
    base_subject = extract_base_subject(subject_name)
    
    # Define search candidates in priority order
    candidates = [
        f"{subject_name}_tracking_locations.json",
        f"{base_subject}_tracking_locations.json",
        "tracking_locations.json",
    ]
    
    print(f"🔍 Searching for tracking locations for subject: {subject_name}")
    if base_subject != subject_name:
        print(f"   Base subject detected: {base_subject}")
    
    for candidate in candidates:
        file_path = Path(candidate)
        if file_path.exists():
            if candidate.startswith(subject_name):
                print(f"✅ Found exact match: {candidate}")
            elif candidate.startswith(base_subject):
                print(f"✅ Found base subject match: {candidate}")
                print(f"   Using {base_subject} tracking locations for {subject_name}")
            else:
                print(f"✅ Found generic match: {candidate}")
                print(f"   Using generic tracking locations for {subject_name}")
            return file_path
    
    print(f"❌ No tracking locations found for {subject_name}")
    print(f"   Searched for: {candidates}")
    return None

def create_variant_tracking_locations(subject_name: str, force_create: bool = False) -> bool:
    """
    Create a subject-specific tracking locations file if it doesn't exist.
    
    This copies from the base subject or generic file and creates a variant-specific copy.
    Useful for mesh variants that should have their own tracking locations file.
    
    Args:
        subject_name: Subject name (potentially with mesh variant)
        force_create: If True, create even if exact match already exists
        
    Returns:
        True if file was created or already exists, False if failed
    """
    target_file = Path(f"{subject_name}_tracking_locations.json")
    
    # Check if exact match already exists
    if target_file.exists() and not force_create:
        print(f"✅ Tracking locations file already exists: {target_file}")
        return True
    
    # Find source file to copy from
    source_file = find_tracking_locations_file(subject_name)
    if source_file is None:
        print(f"❌ Cannot create {target_file} - no source file found")
        return False
    
    # Don't copy if source and target are the same
    if source_file.resolve() == target_file.resolve():
        print(f"✅ Tracking locations file already exists: {target_file}")
        return True
    
    try:
        # Copy the file
        import shutil
        shutil.copy2(source_file, target_file)
        print(f"✅ Created tracking locations file: {target_file}")
        print(f"   Copied from: {source_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create {target_file}: {e}")
        return False

def validate_subject_files(subject_name: str, required_files: List[str] = None) -> Dict[str, bool]:
    """
    Validate that all required files exist for a subject.
    
    Args:
        subject_name: Subject name to validate
        required_files: List of required file patterns. If None, uses default set.
        
    Returns:
        Dictionary mapping file type to existence status
    """
    if required_files is None:
        required_files = [
            "flow_profile",
            "flow_profile_smoothed", 
            "tracking_locations",
            "xyz_tables"
        ]
    
    results = {}
    
    print(f"🔍 Validating files for subject: {subject_name}")
    
    # Check flow profiles
    if "flow_profile" in required_files:
        flow_file = find_flow_profile_file(subject_name)
        results["flow_profile"] = flow_file is not None
        if not results["flow_profile"]:
            print(f"❌ Missing flow profile")
    
    if "flow_profile_smoothed" in required_files:
        base_subject = extract_base_subject(subject_name)
        smoothed_candidates = [
            f"{subject_name}FlowProfile_smoothed.csv",
            f"{base_subject}FlowProfile_smoothed.csv"
        ]
        found_smoothed = any(Path(f).exists() for f in smoothed_candidates)
        results["flow_profile_smoothed"] = found_smoothed
        if not found_smoothed:
            print(f"❌ Missing smoothed flow profile")
    
    # Check tracking locations
    if "tracking_locations" in required_files:
        tracking_file = find_tracking_locations_file(subject_name)
        results["tracking_locations"] = tracking_file is not None
        if not results["tracking_locations"]:
            print(f"❌ Missing tracking locations")
    
    # Check XYZ tables directory
    if "xyz_tables" in required_files:
        xyz_dirs = [
            f"{subject_name}_xyz_tables",
            f"{subject_name}_xyz_tables_with_patches"
        ]
        found_xyz = any(Path(d).exists() and Path(d).is_dir() for d in xyz_dirs)
        results["xyz_tables"] = found_xyz
        if not found_xyz:
            print(f"❌ Missing XYZ tables directory")
    
    # Summary
    all_found = all(results.values())
    if all_found:
        print(f"✅ All required files found for {subject_name}")
    else:
        missing = [k for k, v in results.items() if not v]
        print(f"❌ Missing files for {subject_name}: {missing}")
    
    return results

def load_tracking_locations(subject_name: str = None, config_file: str = None) -> Dict:
    """
    Load tracking locations from configuration file with smart subject resolution.
    
    Args:
        subject_name: Subject name (e.g., "OSAMRI007" or "2mmeshOSAMRI007"). 
                     If provided, will use smart resolution to find appropriate file.
        config_file: Explicit config file path. If provided, overrides subject_name logic
        
    Returns:
        Dictionary with 'locations' and 'combinations' keys
    """
    # Determine config file to use
    if config_file is not None:
        # Explicit file path provided
        file_to_use = config_file
        file_path = Path(file_to_use)
    elif subject_name is not None:
        # Check for exact subject-specific file first
        exact_file = Path(f"{subject_name}_tracking_locations.json")
        base_subject = extract_base_subject(subject_name)
        
        if exact_file.exists():
            file_path = exact_file
            file_to_use = str(file_path)
            print(f"Loaded tracking locations from: {file_to_use}")
        else:
            # Use smart resolution to find appropriate file
            file_path = find_tracking_locations_file(subject_name)
            if file_path is None:
                print(f"❌ No tracking locations file found for {subject_name}")
                return {'locations': [], 'combinations': []}
            file_to_use = str(file_path)
            
            # Warn if using inherited file for mesh variant
            if base_subject != subject_name and not file_to_use.startswith(subject_name):
                print(f"⚠️  WARNING: Using inherited tracking locations from {base_subject}")
                print(f"⚠️  Patch numbers may be incorrect for mesh variant {subject_name}")
                print(f"⚠️  Consider updating: python src/main.py --raw-surface --subject {subject_name}")
    else:
        # Default to generic file
        file_to_use = "tracking_locations.json"
        file_path = Path(file_to_use)
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        # Return the full config (with both 'locations' and 'combinations' if present)
        if 'locations' not in config:
            print(f"Missing 'locations' section in {file_to_use}!")
            return {'locations': [], 'combinations': []}
        
        # Add empty combinations if not present
        if 'combinations' not in config:
            config['combinations'] = []
            
        return config
    except FileNotFoundError:
        print(f"Configuration file {file_to_use} not found!")
        return {'locations': [], 'combinations': []}
    except json.JSONDecodeError:
        print(f"Error parsing {file_to_use}. Make sure it's valid JSON!")
        return {'locations': [], 'combinations': []}
    except KeyError:
        print(f"Invalid configuration format in {file_to_use}!")
        return {'locations': [], 'combinations': []}

def find_breathing_cycle_bounds(subject_name: str) -> Tuple[float, float]:
    """
    Find the time bounds of a single breathing cycle from the flow profile.
    
    Args:
        subject_name: Name of the subject (e.g., 'OSAMRI007')
        
    Returns:
        Tuple of (first_crossing_time, last_crossing_time) in milliseconds
    """
    print(f"\nAnalyzing flow profile for subject {subject_name}...")
    
    # Find the flow profile file (handles base subject lookup)
    flow_profile_path = find_flow_profile_file(subject_name)
    if flow_profile_path is None:
        print(f"❌ Could not find flow profile for subject {subject_name}")
        return None, None
    
    print(f"📊 Using flow profile: {flow_profile_path}")
    
    # Read flow profile data
    flow_df = pd.read_csv(flow_profile_path)
    print("\nFlow profile data structure:")
    print(flow_df.head())
    print("\nFlow profile time range (in seconds):")
    print(f"Start: {flow_df.iloc[0, 0]:.3f} s")
    print(f"End: {flow_df.iloc[-1, 0]:.3f} s")
    
    # Convert time from seconds to milliseconds
    time = flow_df.iloc[:, 0].values * 1000
    flow = flow_df.iloc[:, 1].values
    
    # Find zero crossings
    zero_crossings = np.where(np.diff(np.signbit(flow)))[0]
    
    if len(zero_crossings) < 2:
        print("Warning: Could not find enough zero crossings in flow profile!")
        return None, None
    
    first_crossing_time = time[zero_crossings[0]]
    last_crossing_time = time[zero_crossings[-1]]
    
    print(f"\nFound breathing cycle bounds:")
    print(f"First zero crossing at t = {first_crossing_time:.2f} ms")
    print(f"Last zero crossing at t = {last_crossing_time:.2f} ms")
    
    # Plot the flow profile with bounds
    plot_flow_profile(time, flow, first_crossing_time, last_crossing_time, subject_name)
    
    return first_crossing_time, last_crossing_time

def filter_xyz_files_by_time(xyz_files: List[Path], start_time: float, end_time: float) -> List[Path]:
    """Filter XYZ table files to only include those within the specified time range."""
    
    filtered_files = []
    
    print("\nMatching CFD data files with breathing cycle:")
    print(f"Breathing cycle time range: {start_time:.1f} ms to {end_time:.1f} ms")
    print("\nAnalyzing files:")
    
    # Detect time unit from the files
    time_unit = detect_time_unit(xyz_files)
    print(f"Detected time unit: {time_unit}")
    
    for file_path in xyz_files:
        try:
            timestep = extract_timestep_from_filename(file_path)
            
            # Convert timestep to milliseconds for comparison
            if time_unit == 's':
                time_ms = timestep * 1000.0
            elif time_unit == 'ms':
                time_ms = timestep
            else:
                # Make best guess based on magnitude
                if timestep < 10:
                    time_ms = timestep * 1000.0  # Assume seconds
                else:
                    time_ms = timestep  # Assume milliseconds
            
            if start_time <= time_ms <= end_time:
                filtered_files.append(file_path)
                print(f"✓ {file_path.name} (t = {time_ms:.1f} ms) - Within breathing cycle")
            else:
                print(f"✗ {file_path.name} (t = {time_ms:.1f} ms) - Outside breathing cycle")
        except ValueError as e:
            print(f"⚠ Skipping {file_path.name} - {e}")
            continue
    
    if not filtered_files:
        print("Error: No valid files found within breathing cycle!")
        return filtered_files
    
    print(f"\nSummary:")
    print(f"Total files: {len(xyz_files)}")
    print(f"Files within breathing cycle: {len(filtered_files)}")
    
    # Calculate time range using the robust extraction function
    time_values = []
    for f in filtered_files:
        try:
            timestep = extract_timestep_from_filename(f)
            if time_unit == 's':
                time_ms = timestep * 1000.0
            elif time_unit == 'ms':
                time_ms = timestep
            else:
                if timestep < 10:
                    time_ms = timestep * 1000.0
                else:
                    time_ms = timestep
            time_values.append(time_ms)
        except ValueError:
            continue
    
    if time_values:
        print(f"Time range of included data: {min(time_values):.1f} ms to {max(time_values):.1f} ms")
    
    return filtered_files

def plot_flow_profile(time: np.ndarray, flow: np.ndarray, 
                     first_crossing_time: float, last_crossing_time: float,
                     subject_name: str):
    """Plot flow profile with breathing cycle highlighted."""
    plt.figure(figsize=(15, 8))
    
    # Plot the full flow profile
    plt.plot(time, flow, 'b-', label='Flow Rate', linewidth=2, alpha=0.6)
    
    # Highlight the clean breathing cycle region
    cycle_mask = (time >= first_crossing_time) & (time <= last_crossing_time)
    plt.plot(time[cycle_mask], flow[cycle_mask], 'b-', linewidth=3, label='Clean Breathing Cycle')
    
    # Add vertical lines at zero crossings
    plt.axvline(x=first_crossing_time, color='r', linestyle='--', linewidth=2, 
                label=f'First Zero Crossing ({first_crossing_time:.1f} ms)')
    plt.axvline(x=last_crossing_time, color='g', linestyle='--', linewidth=2,
                label=f'Last Zero Crossing ({last_crossing_time:.1f} ms)')
    
    # Fill the region of clean breathing cycle
    plt.fill_between(time[cycle_mask], flow[cycle_mask], alpha=0.2, color='blue',
                    label='Clean Cycle Region')
    
    # Add horizontal line at zero flow
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Customize the plot
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Flow Rate', fontsize=14)
    plt.title(f'Flow Profile with Clean Breathing Cycle Highlighted\nSubject: {subject_name}', fontsize=17)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for the time range
    plt.text(0.02, 0.98, 
             f'Clean Breathing Cycle:\n'
             f'Start: {first_crossing_time:.1f} ms\n'
             f'End: {last_crossing_time:.1f} ms\n'
             f'Duration: {last_crossing_time - first_crossing_time:.1f} ms',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top',
             fontsize=10)
    
    # Adjust legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Save with extra space for legend
    plt.savefig(f'{subject_name}_flow_profile_bounds.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def add_patch_numbers_to_table(input_file: Path, output_file: Path) -> int:
    """
    Add patch numbers to surface table based on Face Index resets.
    Returns the total number of patches found.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save processed CSV file
        
    Returns:
        Total number of patches found
    """
    print(f"Processing {input_file.name}")
    
    # Process in chunks to handle large files
    chunk_size = 1000
    first_chunk = True
    current_patch = 1
    prev_face_idx = -1
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Calculate patch numbers for this chunk
        patch_numbers = []
        
        for _, row in chunk.iterrows():
            face_idx = row['Face Index']
            # Start new patch when Face Index resets to 0 (after being > 0)
            if face_idx == 0 and prev_face_idx > 0:
                current_patch += 1
            patch_numbers.append(current_patch)
            prev_face_idx = face_idx
        
        # Add Patch Number column right after Face Index
        chunk.insert(chunk.columns.get_loc('Face Index') + 1, 'Patch Number', patch_numbers)
        
        # Write to output file (append mode after first chunk)
        chunk.to_csv(output_file, 
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False)
        first_chunk = False
    
    return current_patch

def preprocess_all_tables(xyz_files: List[Path], subject_name: str) -> List[Path]:
    """
    Add patch numbers to all surface tables.
    Returns list of paths to processed files.
    
    Args:
        xyz_files: List of paths to XYZ table files
        subject_name: Name of the subject
        
    Returns:
        List of paths to processed files with patch numbers
    """
    # If files are already patched (have patched_ prefix), just return them
    if any(f.name.startswith('patched_') for f in xyz_files):
        print("Using existing patched tables")
        return xyz_files
    
    # Create output directory for new patched files    
    output_dir = Path(f'{subject_name}_xyz_tables_with_patches')
    output_dir.mkdir(exist_ok=True)
    print(f"\nPreprocessing tables to add patch numbers...")
    print(f"Output directory: {output_dir}")
    
    processed_files = []
    
    # Process first file to get reference patch count
    first_file = xyz_files[0]
    first_output = output_dir / f"patched_{first_file.name}"
    total_patches = add_patch_numbers_to_table(first_file, first_output)
    processed_files.append(first_output)
    print(f"Found {total_patches} patches in first table")
    
    # Process remaining files
    for file_path in xyz_files[1:]:
        output_file = output_dir / f"patched_{file_path.name}"
        add_patch_numbers_to_table(file_path, output_file)
        processed_files.append(output_file)
    
    return processed_files

def extract_timestep_from_filename(file_path: Path) -> float:
    """
    Extract timestep from XYZ table filename, handling various naming conventions.
    
    Supports:
    - Integer format: XYZ_Internal_Table_table_123.csv → 123.0
    - Scientific notation: XYZ_Internal_Table_table_2.300000e+00.csv → 2.3
    - Decimal format: XYZ_Internal_Table_table_1.500000.csv → 1.5
    
    Args:
        file_path: Path to the XYZ table file
        
    Returns:
        Timestep as float (in seconds or milliseconds depending on source)
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
    
    # Fallback: try splitting by '_' and find numeric parts
    parts = stem.split('_')
    for part in reversed(parts):  # Start from the end
        try:
            # Try as integer first
            if part.isdigit():
                return float(part)
            # Try as float
            return float(part)
        except ValueError:
            continue
    
    raise ValueError(f"Could not extract timestep from filename: {file_path.name}")

def detect_time_unit(xyz_files: List[Path]) -> str:
    """
    Detect the time unit used in XYZ table filenames.
    
    Args:
        xyz_files: List of XYZ table file paths
        
    Returns:
        'ms' for milliseconds, 's' for seconds, or 'unknown'
    """
    if not xyz_files:
        return 'unknown'
    
    try:
        # Extract timesteps from a sample of files
        sample_size = min(10, len(xyz_files))
        timesteps = []
        for file_path in xyz_files[:sample_size]:
            try:
                timestep = extract_timestep_from_filename(file_path)
                timesteps.append(timestep)
            except ValueError:
                continue
        
        if not timesteps:
            return 'unknown'
        
        max_timestep = max(timesteps)
        min_timestep = min(timesteps)
        
        # Heuristic: if max timestep > 100, likely milliseconds
        # if max timestep < 10, likely seconds
        if max_timestep > 100:
            return 'ms'
        elif max_timestep < 10:
            return 's'
        else:
            # Ambiguous range, check intervals
            if len(timesteps) > 1:
                timesteps.sort()
                intervals = [timesteps[i+1] - timesteps[i] for i in range(len(timesteps)-1)]
                avg_interval = np.mean(intervals)
                
                # If average interval is around 1, likely milliseconds
                # If average interval is around 0.001, likely seconds
                if avg_interval >= 0.1:
                    return 'ms'
                else:
                    return 's'
            
        return 'unknown'
        
    except Exception:
        return 'unknown'

def get_xyz_file_info(subject_name: str) -> Dict:
    """
    Analyze XYZ table files for a subject and return comprehensive information.
    
    Args:
        subject_name: Subject name (e.g., "23mmeshOSAMRI007")
        
    Returns:
        Dictionary with file analysis information
    """
    # Find XYZ tables directory
    xyz_dirs = [
        f"{subject_name}_xyz_tables_with_patches",
        f"{subject_name}_xyz_tables"
    ]
    
    xyz_dir = None
    for dir_name in xyz_dirs:
        if Path(dir_name).exists():
            xyz_dir = Path(dir_name)
            break
    
    if xyz_dir is None:
        return {
            'directory': None,
            'files_found': 0,
            'time_unit': 'unknown',
            'timesteps': [],
            'naming_convention': 'unknown',
            'time_range': None,
            'intervals': 'unknown'
        }
    
    # Get all CSV files
    csv_files = list(xyz_dir.glob('*.csv'))
    
    if not csv_files:
        return {
            'directory': str(xyz_dir),
            'files_found': 0,
            'time_unit': 'unknown',
            'timesteps': [],
            'naming_convention': 'unknown',
            'time_range': None,
            'intervals': 'unknown'
        }
    
    # Extract timesteps
    timesteps = []
    failed_files = []
    
    for file_path in csv_files:
        try:
            timestep = extract_timestep_from_filename(file_path)
            timesteps.append(timestep)
        except ValueError:
            failed_files.append(file_path.name)
    
    # Detect naming convention
    sample_file = csv_files[0].stem
    if 'e+' in sample_file or 'e-' in sample_file:
        naming_convention = 'scientific_notation'
    elif '.' in sample_file and any(char.isdigit() for char in sample_file.split('.')[-1]):
        naming_convention = 'decimal'
    else:
        naming_convention = 'integer'
    
    # Detect time unit
    time_unit = detect_time_unit(csv_files)
    
    # Analyze intervals
    if len(timesteps) > 1:
        timesteps.sort()
        intervals = [timesteps[i+1] - timesteps[i] for i in range(len(timesteps)-1)]
        
        # Check if intervals are uniform
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)
        
        if interval_std < 0.01 * interval_mean:  # Less than 1% variation
            intervals_type = f"uniform_{interval_mean:.3f}"
        else:
            intervals_type = f"variable_{interval_mean:.3f}±{interval_std:.3f}"
    else:
        intervals_type = "single_file"
    
    return {
        'directory': str(xyz_dir),
        'files_found': len(csv_files),
        'files_parsed': len(timesteps),
        'failed_files': failed_files,
        'time_unit': time_unit,
        'timesteps': sorted(timesteps),  # Natural sorting by timestep value
        'naming_convention': naming_convention,
        'time_range': (min(timesteps), max(timesteps)) if timesteps else None,
        'intervals': intervals_type
    }

def standardize_timestep_for_pipeline(timestep: float, time_unit: str) -> int:
    """
    Convert timestep to pipeline-standard format (milliseconds as integer).
    
    Args:
        timestep: Original timestep value
        time_unit: 'ms', 's', or 'unknown'
        
    Returns:
        Timestep in milliseconds as integer
    """
    if time_unit == 's':
        # Convert seconds to milliseconds
        return int(timestep * 1000)
    elif time_unit == 'ms':
        # Already in milliseconds
        return int(timestep)
    else:
        # Unknown unit, make best guess based on magnitude
        if timestep < 10:
            # Likely seconds
            return int(timestep * 1000)
        else:
            # Likely milliseconds
            return int(timestep)

def find_closest_xyz_file(subject_name: str, target_timestep: int) -> Optional[Path]:
    """
    Find the XYZ file closest to the target timestep for a given subject.
    
    Args:
        subject_name: Subject name (e.g., "23mmeshOSAMRI007")
        target_timestep: Target timestep in milliseconds
        
    Returns:
        Path to the closest XYZ file, or None if not found
    """
    file_info = get_xyz_file_info(subject_name)
    
    if file_info['files_found'] == 0:
        return None
    
    xyz_dir = Path(file_info['directory'])
    timesteps = file_info['timesteps']
    time_unit = file_info['time_unit']
    
    # Convert target timestep to the file's time unit
    if time_unit == 's':
        target_in_file_units = target_timestep / 1000.0
    elif time_unit == 'ms':
        target_in_file_units = float(target_timestep)
    else:
        # Unknown unit, make best guess
        if max(timesteps) < 10:
            target_in_file_units = target_timestep / 1000.0
        else:
            target_in_file_units = float(target_timestep)
    
    # Find closest timestep
    closest_timestep = min(timesteps, key=lambda x: abs(x - target_in_file_units))
    
    # Find the corresponding file
    csv_files = list(xyz_dir.glob('*.csv'))
    for file_path in csv_files:
        try:
            file_timestep = extract_timestep_from_filename(file_path)
            if abs(file_timestep - closest_timestep) < 1e-6:  # Account for floating point precision
                return file_path
        except ValueError:
            continue
    
    return None

def get_xyz_files_in_chronological_order(subject_name: str) -> List[Path]:
    """
    Get XYZ table files in natural chronological order based on timestep values.
    
    This ensures proper time sequence regardless of filename format:
    - Scientific notation: 2.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.25e+00, ...
    - Integer format: 1, 2, 3, 4, 5, ...
    - Decimal format: 0.25, 0.5, 0.75, 1.0, 1.25, ...
    
    Args:
        subject_name: Subject name (e.g., "23mmeshOSAMRI007")
        
    Returns:
        List of Path objects sorted by actual timestep value
    """
    # Find XYZ tables directory
    xyz_dirs = [
        f"{subject_name}_xyz_tables_with_patches",
        f"{subject_name}_xyz_tables"
    ]
    
    xyz_dir = None
    for dir_name in xyz_dirs:
        if Path(dir_name).exists():
            xyz_dir = Path(dir_name)
            break
    
    if xyz_dir is None:
        return []
    
    # Get all CSV files
    csv_files = list(xyz_dir.glob('*XYZ_Internal_Table_table_*.csv'))
    
    if not csv_files:
        return []
    
    # Extract timesteps and create (timestep, file_path) pairs
    timestep_file_pairs = []
    for file_path in csv_files:
        try:
            timestep = extract_timestep_from_filename(file_path)
            timestep_file_pairs.append((timestep, file_path))
        except ValueError:
            # Skip files that can't be parsed
            continue
    
    # Sort by timestep value (natural chronological order)
    timestep_file_pairs.sort(key=lambda x: x[0])
    
    # Return just the file paths in chronological order
    return [file_path for timestep, file_path in timestep_file_pairs]

def get_timestep_range_info(subject_name: str) -> Dict:
    """
    Get comprehensive timestep range information for a subject.
    
    Args:
        subject_name: Subject name
        
    Returns:
        Dictionary with timestep range and file information
    """
    files_in_order = get_xyz_files_in_chronological_order(subject_name)
    
    if not files_in_order:
        return {
            'files_found': 0,
            'timestep_range': None,
            'first_file': None,
            'last_file': None,
            'total_duration': None,
            'time_unit': 'unknown'
        }
    
    # Extract timesteps
    timesteps = []
    for file_path in files_in_order:
        try:
            timestep = extract_timestep_from_filename(file_path)
            timesteps.append(timestep)
        except ValueError:
            continue
    
    if not timesteps:
        return {
            'files_found': len(files_in_order),
            'timestep_range': None,
            'first_file': None,
            'last_file': None,
            'total_duration': None,
            'time_unit': 'unknown'
        }
    
    # Detect time unit
    time_unit = detect_time_unit(files_in_order)
    
    # Calculate range information
    first_timestep = timesteps[0]
    last_timestep = timesteps[-1]
    total_duration = last_timestep - first_timestep
    
    return {
        'files_found': len(files_in_order),
        'timestep_range': (first_timestep, last_timestep),
        'first_file': files_in_order[0],
        'last_file': files_in_order[-1],
        'total_duration': total_duration,
        'time_unit': time_unit,
        'chronological_files': files_in_order
    } 