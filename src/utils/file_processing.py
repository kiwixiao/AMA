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
    Find the appropriate tracking locations file for a subject.

    PRODUCTION MODE: Only search in results folder for self-contained subject data.

    Search order:
    1. {subject_name}_results/{subject_name}_tracking_locations.json

    Args:
        subject_name: Subject name (potentially with mesh variant)

    Returns:
        Path to tracking locations file, or None if not found
    """
    # PRODUCTION MODE: Only check results folder
    results_file = Path(f"{subject_name}_results/{subject_name}_tracking_locations.json")

    print(f"🔍 Searching for tracking locations for subject: {subject_name}")

    if results_file.exists():
        print(f"✅ Found in results folder: {results_file}")
        return results_file

    print(f"❌ No tracking locations found for {subject_name}")
    print(f"   Expected location: {results_file}")
    print(f"   Run --patch-selection first to create template JSON")
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
    
    # Note: Smoothed flow profile is optional and generated automatically if needed
    
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
        # PRODUCTION MODE: Only check results folder for self-contained subject data
        results_file = Path(f"{subject_name}_results/{subject_name}_tracking_locations.json")
        base_subject = extract_base_subject(subject_name)

        if results_file.exists():
            # Load from results folder (production-ready, self-contained)
            file_path = results_file
            file_to_use = str(file_path)
            print(f"✅ Loaded tracking locations from: {file_to_use}")
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


def create_template_tracking_locations(subject_name: str, output_dir: Path = None,
                                       remesh_info: dict = None) -> Path:
    """
    Create a template tracking locations JSON file for fresh cases.

    This template provides placeholder values that the user must update
    after using the interactive HTML visualization to identify patch/face indices.

    Args:
        subject_name: Subject name (e.g., "OSAMRI007")
        output_dir: Directory to save the template (default: {subject}_results/)
        remesh_info: Optional dictionary with remesh configuration from Phase 1

    Returns:
        Path to the created template file
    """
    if output_dir is None:
        output_dir = Path(f"{subject_name}_results")

    output_dir.mkdir(parents=True, exist_ok=True)
    template_file = output_dir / f"{subject_name}_tracking_locations.json"

    # Check if file already exists
    if template_file.exists():
        print(f"📋 Template tracking locations file already exists: {template_file}")
        return template_file

    # Determine if remesh is enabled
    has_remesh = remesh_info and remesh_info.get('has_remesh', False)

    # Create location template - always include all fields for consistency
    def create_location_template(num):
        location = {
            "description": f"Location {num} - UPDATE THIS DESCRIPTION",
            "patch_number": 0,
            "face_indices": [0],
            "coordinates": [0.0, 0.0, 0.0],
            "post_remesh_list": []  # Empty if no remesh, populated in Phase 2 if remesh
        }
        return location

    # Create template structure with placeholders
    template = {
        "locations": [
            create_location_template(1),
            create_location_template(2),
            create_location_template(3)
        ],
        "combinations": [],
        "_instructions": {
            "step1": "Open the interactive HTML visualization in your browser",
            "step2": "Hover over points to see Patch Number and Face Index",
            "step3": "Update each location's patch_number, face_indices, and coordinates",
            "step4": "Update the description to something meaningful (e.g., 'Posterior soft palate')",
            "step5": "Add or remove locations as needed",
            "step6": "Run the pipeline again with --plotting to generate analysis"
        }
    }

    # Add remesh_info if available
    if has_remesh:
        template["remesh_info"] = {
            "has_remesh": True,
            "remesh_before_file": remesh_info.get('remesh_before_file'),
            "remesh_after_file": remesh_info.get('remesh_after_file'),
            "remesh_timestep_ms": remesh_info.get('remesh_timestep_ms'),
            "_note": "post_remesh fields will be auto-populated when you run --plotting"
        }
        # Include remesh_events list if available (for multiple remesh events)
        if remesh_info.get('remesh_events'):
            template["remesh_info"]["remesh_events"] = remesh_info['remesh_events']
        template["_instructions"]["step7"] = "For remesh cases: post_remesh mappings are auto-calculated from coordinates"
    else:
        template["remesh_info"] = {
            "has_remesh": False
        }

    # Write template to file
    with open(template_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"📋 Created template tracking locations file: {template_file}")
    print("   ⚠️  UPDATE THIS FILE with correct patch/face values from interactive HTML")
    if has_remesh:
        print("   🔄 Remesh enabled: post_remesh fields will be auto-calculated in Phase 2")

    return template_file


def find_closest_point_after_remesh(target_coords: Tuple[float, float, float],
                                    csv_file_path: Path,
                                    max_distance: float = 0.005) -> Optional[Dict]:
    """
    Find the closest point in a CSV file to the target coordinates.

    This is used for remesh handling - when the CFD mesh changes during simulation,
    the patch_number and face_index change but the physical location stays the same.
    We use coordinates to find the corresponding point in the new mesh.

    Args:
        target_coords: (x, y, z) coordinates to match
        csv_file_path: Path to the CSV file to search
        max_distance: Maximum allowed distance (meters) for a valid match (default: 5mm)

    Returns:
        Dictionary with matched point info, or None if no match found
        {
            'patch_number': int,
            'face_index': int,
            'coordinates': [x, y, z],
            'distance': float  # distance from target
        }
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"❌ Error reading CSV file {csv_file_path}: {e}")
        return None

    # Check required columns
    required_cols = ['X (m)', 'Y (m)', 'Z (m)', 'Face Index']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ CSV file missing required columns: {required_cols}")
        return None

    # Calculate Patch Number if not present
    if 'Patch Number' not in df.columns:
        # Compute patch numbers based on Face Index resets
        patch_numbers = []
        current_patch = 1
        prev_face_idx = -1
        for face_idx in df['Face Index']:
            if face_idx == 0 and prev_face_idx > 0:
                current_patch += 1
            patch_numbers.append(current_patch)
            prev_face_idx = face_idx
        df['Patch Number'] = patch_numbers

    # Calculate distances from target coordinates
    target_x, target_y, target_z = target_coords
    distances = np.sqrt(
        (df['X (m)'] - target_x)**2 +
        (df['Y (m)'] - target_y)**2 +
        (df['Z (m)'] - target_z)**2
    )

    # Find minimum distance
    min_idx = distances.idxmin()
    min_distance = distances[min_idx]

    if min_distance > max_distance:
        print(f"⚠️  Closest point is {min_distance*1000:.2f}mm away (max: {max_distance*1000:.2f}mm)")
        return None

    # Return matched point info
    matched_row = df.loc[min_idx]
    return {
        'patch_number': int(matched_row['Patch Number']),
        'face_index': int(matched_row['Face Index']),
        'coordinates': [
            float(matched_row['X (m)']),
            float(matched_row['Y (m)']),
            float(matched_row['Z (m)'])
        ],
        'distance': float(min_distance)
    }


def update_tracking_locations_for_remesh(tracking_config: Dict,
                                         before_csv: Path,
                                         after_csv: Path,
                                         max_distance: float = 0.005) -> Dict:
    """
    Update tracking locations for a remeshed CFD simulation.

    This function:
    1. Gets coordinates from the tracking location (or validates against before-remesh data)
    2. Finds the closest points in the first frame after remesh
    3. PRESERVES original patch_number and face_indices
    4. ADDS post_remesh field with new patch/face for after-remesh timesteps

    The original patch/face is used for pre-remesh timesteps.
    The post_remesh patch/face is used for post-remesh timesteps.

    Args:
        tracking_config: Current tracking locations config
        before_csv: Path to CSV file just before remesh
        after_csv: Path to CSV file just after remesh
        max_distance: Maximum allowed distance for coordinate matching

    Returns:
        Updated tracking config with post_remesh mappings (original preserved)
    """
    import pandas as pd

    print(f"\n🔄 Calculating post-remesh mappings...")
    print(f"   Before remesh: {before_csv}")
    print(f"   After remesh: {after_csv}")
    print(f"   NOTE: Original patch/face values will be PRESERVED")

    # Load before-remesh data to validate coordinates
    try:
        before_df = pd.read_csv(before_csv)
    except Exception as e:
        print(f"❌ Error reading before-remesh CSV: {e}")
        return tracking_config

    # Calculate Patch Number for before_df if needed
    if 'Patch Number' not in before_df.columns:
        patch_numbers = []
        current_patch = 1
        prev_face_idx = -1
        for face_idx in before_df['Face Index']:
            if face_idx == 0 and prev_face_idx > 0:
                current_patch += 1
            patch_numbers.append(current_patch)
            prev_face_idx = face_idx
        before_df['Patch Number'] = patch_numbers

    updated_config = tracking_config.copy()

    # Update remesh_info with file paths (preserve existing info if present)
    if 'remesh_info' not in updated_config:
        updated_config['remesh_info'] = {}
    updated_config['remesh_info']['has_remesh'] = True
    updated_config['remesh_info']['before_file'] = str(Path(before_csv).name)
    updated_config['remesh_info']['after_file'] = str(Path(after_csv).name)
    updated_config['remesh_info']['mappings'] = []

    success_count = 0
    for i, location in enumerate(updated_config['locations']):
        patch_num = location['patch_number']
        face_idx = location['face_indices'][0]
        description = location.get('description', f'Location {i+1}')

        print(f"\n   Processing: {description}")
        print(f"   Original (pre-remesh): Patch {patch_num}, Face {face_idx}")

        # Get coordinates - prefer stored coordinates, validate against before-remesh data
        coords = None
        if 'coordinates' in location and location['coordinates'] != [0.0, 0.0, 0.0]:
            coords = tuple(location['coordinates'])
            print(f"   Using stored coordinates: ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f})")
        else:
            # Try to get from before-remesh data
            mask = (before_df['Patch Number'] == patch_num) & (before_df['Face Index'] == face_idx)
            matched_rows = before_df[mask]
            if len(matched_rows) > 0:
                row = matched_rows.iloc[0]
                coords = (row['X (m)'], row['Y (m)'], row['Z (m)'])
                # Update stored coordinates
                updated_config['locations'][i]['coordinates'] = list(coords)
                print(f"   Extracted coordinates: ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f})")
            else:
                print(f"   ❌ No coordinates available, skipping")
                updated_config['locations'][i]['post_remesh'] = None
                continue

        # Find closest point in after-remesh data
        match = find_closest_point_after_remesh(coords, after_csv, max_distance)

        if match is None:
            print(f"   ❌ No matching point found after remesh (within {max_distance*1000:.1f}mm)")
            updated_config['locations'][i]['post_remesh'] = None
            continue

        print(f"   Post-remesh: Patch {match['patch_number']}, Face {match['face_index']}")
        print(f"   Match distance: {match['distance']*1000:.3f}mm")

        # ADD post_remesh field (DO NOT overwrite original patch_number/face_indices)
        updated_config['locations'][i]['post_remesh'] = {
            'patch_number': match['patch_number'],
            'face_index': match['face_index'],
            'coordinates': match['coordinates'],
            'distance_mm': match['distance'] * 1000
        }

        # Store mapping info for reference
        updated_config['remesh_info']['mappings'].append({
            'description': description,
            'pre_remesh': {'patch_number': patch_num, 'face_index': face_idx},
            'post_remesh': {'patch_number': match['patch_number'], 'face_index': match['face_index']},
            'distance_mm': match['distance'] * 1000
        })
        success_count += 1

    print(f"\n✅ Remesh mapping complete:")
    print(f"   {success_count}/{len(updated_config['locations'])} locations mapped")
    print(f"   Original patch/face values PRESERVED for pre-remesh timesteps")
    print(f"   post_remesh values added for post-remesh timesteps")
    return updated_config


def update_tracking_locations_for_multiple_remesh(tracking_config: Dict,
                                                   remesh_events: list,
                                                   xyz_tables_dir: Path,
                                                   max_distance: float = 0.005) -> Dict:
    """
    Update tracking locations for multiple remesh events.

    For each remesh event, calculates the post-remesh mapping using the coordinates
    from the before-remesh timestep. Builds a post_remesh_list with one mapping
    per remesh event.

    Args:
        tracking_config: Current tracking locations config
        remesh_events: List of remesh events, each with {before_file, after_file, timestep_ms}
        xyz_tables_dir: Path to directory containing CSV files
        max_distance: Maximum allowed distance for coordinate matching

    Returns:
        Updated tracking config with post_remesh_list for each location
    """
    import pandas as pd

    if not remesh_events:
        return tracking_config

    print(f"\n🔄 Calculating post-remesh mappings for {len(remesh_events)} remesh event(s)...")

    updated_config = tracking_config.copy()

    # Initialize post_remesh_list for each location
    for i in range(len(updated_config['locations'])):
        updated_config['locations'][i]['post_remesh_list'] = []

    # Update remesh_info
    if 'remesh_info' not in updated_config:
        updated_config['remesh_info'] = {}
    updated_config['remesh_info']['has_remesh'] = True
    updated_config['remesh_info']['remesh_events'] = remesh_events
    updated_config['remesh_info']['mappings'] = []

    # Process each remesh event
    for event_idx, event in enumerate(remesh_events):
        before_file = event['before_file']
        after_file = event['after_file']
        timestep_ms = event['timestep_ms']

        print(f"\n   {'─'*40}")
        print(f"   Remesh Event #{event_idx + 1}: boundary at {timestep_ms:.1f}ms")
        print(f"   Before: {before_file}")
        print(f"   After:  {after_file}")

        # Find the CSV files
        before_csv = xyz_tables_dir / before_file
        after_csv = xyz_tables_dir / after_file

        if not before_csv.exists():
            print(f"   ❌ Before file not found: {before_csv}")
            continue
        if not after_csv.exists():
            print(f"   ❌ After file not found: {after_csv}")
            continue

        # For first event, use original coordinates
        # For subsequent events, use the previous post_remesh coordinates
        for i, location in enumerate(updated_config['locations']):
            description = location.get('description', f'Location {i+1}')

            if event_idx == 0:
                # First remesh: use original patch/face to get coordinates
                patch_num = location['patch_number']
                face_idx = location['face_indices'][0]
                source = "original"

                # Get coordinates
                if 'coordinates' in location and location['coordinates'] != [0.0, 0.0, 0.0]:
                    coords = tuple(location['coordinates'])
                else:
                    # Need to extract from before-remesh CSV
                    try:
                        before_df = pd.read_csv(before_csv)
                        if 'Patch Number' not in before_df.columns:
                            patch_numbers = []
                            current_patch = 1
                            prev_face_idx = -1
                            for fidx in before_df['Face Index']:
                                if fidx == 0 and prev_face_idx > 0:
                                    current_patch += 1
                                patch_numbers.append(current_patch)
                                prev_face_idx = fidx
                            before_df['Patch Number'] = patch_numbers

                        mask = (before_df['Patch Number'] == patch_num) & (before_df['Face Index'] == face_idx)
                        matched_rows = before_df[mask]
                        if len(matched_rows) > 0:
                            row = matched_rows.iloc[0]
                            coords = (row['X (m)'], row['Y (m)'], row['Z (m)'])
                            updated_config['locations'][i]['coordinates'] = list(coords)
                        else:
                            print(f"      {description}: ❌ No coordinates, skipping")
                            continue
                    except Exception as e:
                        print(f"      {description}: ❌ Error reading coordinates: {e}")
                        continue
            else:
                # Subsequent remesh: use previous post_remesh coordinates
                prev_mapping = updated_config['locations'][i]['post_remesh_list'][-1] if updated_config['locations'][i]['post_remesh_list'] else None
                if prev_mapping and 'coordinates' in prev_mapping:
                    coords = tuple(prev_mapping['coordinates'])
                    source = f"post_remesh_{event_idx}"
                else:
                    print(f"      {description}: ❌ No previous mapping, skipping")
                    continue

            # Find closest point in after-remesh data
            match = find_closest_point_after_remesh(coords, after_csv, max_distance)

            if match is None:
                print(f"      {description}: ❌ No match within {max_distance*1000:.1f}mm")
                updated_config['locations'][i]['post_remesh_list'].append(None)
                continue

            mapping = {
                'event_index': event_idx,
                'patch_number': match['patch_number'],
                'face_index': match['face_index'],
                'coordinates': match['coordinates'],
                'distance_mm': match['distance'] * 1000
            }
            updated_config['locations'][i]['post_remesh_list'].append(mapping)

            print(f"      {description}: Patch {match['patch_number']}, Face {match['face_index']} ({match['distance']*1000:.2f}mm)")

    # Count successes
    success_count = sum(1 for loc in updated_config['locations']
                        if loc.get('post_remesh_list') and all(m is not None for m in loc['post_remesh_list']))

    print(f"\n✅ Multi-remesh mapping complete:")
    print(f"   {success_count}/{len(updated_config['locations'])} locations fully mapped")
    print(f"   {len(remesh_events)} remesh event(s) processed")
    return updated_config


def ask_remesh_questions_interactive(xyz_files: list) -> dict:
    """
    Interactively ask user about remesh configuration during Phase 1.

    This function:
    1. Shows available CSV files
    2. Asks if there's a remesh in this case
    3. If yes, asks for before/after CSV filenames (with TAB completion)
    4. Validates the files exist
    5. Calculates remesh timestep boundary

    Args:
        xyz_files: List of CSV file paths (sorted chronologically)

    Returns:
        Dictionary with remesh info:
        {
            'has_remesh': bool,
            'remesh_before_file': str or None,
            'remesh_after_file': str or None,
            'remesh_timestep_ms': float or None,
            'pre_remesh_files': list,  # Files before remesh
            'post_remesh_files': list  # Files after remesh
        }
    """
    import re

    def extract_timestep(filepath):
        """Extract timestep in ms from CSV filename."""
        filename = Path(filepath).name
        # Try scientific notation first (e.g., 2.300000e+00)
        match = re.search(r'table_([0-9.]+e[+-]?[0-9]+)\.csv', filename, re.IGNORECASE)
        if match:
            return float(match.group(1)) * 1000  # seconds to ms
        # Try integer format (e.g., 2387)
        match = re.search(r'table_(\d+)\.csv', filename)
        if match:
            return float(match.group(1))  # already in ms
        return None

    # Build a fast lookup: timestep number -> full filename
    # e.g., "1000" -> "XYZ_Internal_Table_table_1000.csv"
    timestep_to_filename = {}
    for f in xyz_files:
        name = Path(f).name
        # Extract just the number part
        match = re.search(r'table_(\d+)\.csv', name)
        if match:
            timestep_to_filename[match.group(1)] = name

    # Cache for completions
    completion_cache = {'text': None, 'matches': []}

    def setup_tab_completion():
        """Setup readline tab completion - supports both paths and timestep numbers."""
        try:
            import readline
            import glob as glob_module

            def completer(text, state):
                # Use cache if same text
                if completion_cache['text'] != text:
                    completion_cache['text'] = text
                    matches = []

                    # Path completion: if contains '/' or starts with '.' or letter
                    if '/' in text or text.startswith('.') or (text and text[0].isalpha()):
                        # Use glob for path completion
                        pattern = text + '*'
                        glob_matches = glob_module.glob(pattern)
                        # Add trailing slash for directories
                        matches = [m + '/' if Path(m).is_dir() else m for m in glob_matches]
                        matches = sorted(matches)[:20]  # Limit results

                    # Numeric completion: timestep numbers
                    elif text.isdigit() or text == '':
                        matches = [ts for ts in timestep_to_filename.keys() if ts.startswith(text)]
                        matches = sorted(matches, key=int)[:20]

                    completion_cache['matches'] = matches

                matches = completion_cache['matches']
                if state < len(matches):
                    return matches[state]
                return None

            readline.set_completer(completer)
            readline.set_completer_delims(' \t\n')
            readline.parse_and_bind('tab: complete')
            return True
        except ImportError:
            return False

    def restore_tab_completion():
        """Restore default tab completion."""
        try:
            import readline
            readline.set_completer(None)
        except ImportError:
            pass

    result = {
        'has_remesh': False,
        'remesh_events': [],  # List of {before_file, after_file, timestep_ms}
        # Backward compatibility (populated from first remesh event):
        'remesh_before_file': None,
        'remesh_after_file': None,
        'remesh_timestep_ms': None,
        'pre_remesh_files': xyz_files,
        'post_remesh_files': []
    }

    print("\n" + "="*60)
    print("🔄 REMESH CONFIGURATION")
    print("="*60)

    # Show file range
    if xyz_files:
        first_ts = extract_timestep(xyz_files[0])
        last_ts = extract_timestep(xyz_files[-1])
        print(f"📊 Found {len(xyz_files)} CSV files")
        print(f"   Time range: {first_ts:.1f}ms - {last_ts:.1f}ms")
        print(f"   First file: {Path(xyz_files[0]).name}")
        print(f"   Last file: {Path(xyz_files[-1]).name}")

    print("\n❓ Does this CFD simulation have mesh remeshing during the run?")
    print("   (Remesh = mesh topology changes, causing patch/face indices to change)")

    while True:
        response = input("\n   Has remesh? [y/n]: ").strip().lower()
        if response in ['y', 'yes']:
            result['has_remesh'] = True
            break
        elif response in ['n', 'no']:
            result['has_remesh'] = False
            print("✅ No remesh - will use consistent patch/face indices throughout")
            return result
        else:
            print("   Please enter 'y' or 'n'")

    # Ask for before/after files with tab completion
    print("\n📋 Please specify the remesh boundary files:")
    print("   💡 TIP: Use TAB for auto-completion")

    # Enable tab completion
    tab_enabled = setup_tab_completion()
    if tab_enabled:
        print("   ✅ Tab completion enabled:")
        print("      - Type number + TAB (e.g., '100' → timesteps)")
        print("      - Type path + TAB (e.g., '2mm' → folders/files)")

    def resolve_file_input(user_input):
        """Resolve user input to a matching file. Accepts timestep number, filename, or full path."""
        # First, try exact timestep match (e.g., "1000" -> "XYZ_Internal_Table_table_1000.csv")
        if user_input in timestep_to_filename:
            return timestep_to_filename[user_input]

        # Check if it's a full path that exists
        input_path = Path(user_input)
        if input_path.exists() and input_path.is_file():
            return input_path.name

        # Check if just the filename part matches
        if '/' in user_input:
            # User gave a path, extract filename
            filename = input_path.name
            if filename in timestep_to_filename.values():
                return filename
            # Try to match the filename
            matches = [f for f in xyz_files if filename in str(f)]
            if len(matches) == 1:
                return Path(matches[0]).name
            elif len(matches) > 1:
                return matches

        # Try matching against filenames
        matches = [f for f in xyz_files if user_input in str(f)]
        if len(matches) == 1:
            return Path(matches[0]).name
        elif len(matches) > 1:
            return matches  # Multiple matches
        return None  # No match

    # Loop to collect multiple remesh events
    remesh_event_num = 1
    while True:
        print(f"\n{'─'*40}")
        print(f"📍 Remesh Event #{remesh_event_num}")
        print(f"{'─'*40}")

        # Get before file
        before_file = None
        while True:
            before_input = input("\n   CSV file BEFORE remesh (last file with old mesh): ").strip()
            resolved = resolve_file_input(before_input)

            if isinstance(resolved, str):
                before_file = resolved
                before_ts = extract_timestep([f for f in xyz_files if resolved in str(f)][0])
                print(f"   ✅ Found: {before_file} (t={before_ts:.1f}ms)")
                break
            elif isinstance(resolved, list):
                print(f"   ⚠️  Multiple matches found. Please be more specific:")
                for m in resolved[:5]:
                    print(f"      - {Path(m).name}")
            else:
                print(f"   ❌ File not found. Enter timestep number (e.g., 1000) or filename.")
                print(f"      Available range: {min(timestep_to_filename.keys(), key=int)} - {max(timestep_to_filename.keys(), key=int)}")

        # Get after file
        after_file = None
        while True:
            after_input = input("\n   CSV file AFTER remesh (first file with new mesh): ").strip()
            resolved = resolve_file_input(after_input)

            if isinstance(resolved, str):
                after_file = resolved
                after_ts = extract_timestep([f for f in xyz_files if resolved in str(f)][0])
                print(f"   ✅ Found: {after_file} (t={after_ts:.1f}ms)")
                break
            elif isinstance(resolved, list):
                print(f"   ⚠️  Multiple matches found. Please be more specific:")
                for m in resolved[:5]:
                    print(f"      - {Path(m).name}")
            else:
                print(f"   ❌ File not found. Enter timestep number (e.g., 1001) or filename.")

        # Calculate remesh timestep (midpoint between before and after)
        before_ts = extract_timestep([f for f in xyz_files if before_file in str(f)][0])
        after_ts = extract_timestep([f for f in xyz_files if after_file in str(f)][0])
        remesh_timestep = (before_ts + after_ts) / 2

        # Add this remesh event
        remesh_event = {
            'before_file': before_file,
            'after_file': after_file,
            'timestep_ms': remesh_timestep
        }
        result['remesh_events'].append(remesh_event)

        print(f"\n   ✅ Remesh #{remesh_event_num} recorded: boundary at {remesh_timestep:.1f}ms")

        # Ask if there are more remesh events
        print(f"\n❓ Is there another remesh event?")
        while True:
            response = input("   Another remesh? [y/n]: ").strip().lower()
            if response in ['y', 'yes']:
                remesh_event_num += 1
                break  # Continue outer loop
            elif response in ['n', 'no']:
                break  # Exit both loops
            else:
                print("   Please enter 'y' or 'n'")

        if response in ['n', 'no']:
            break  # Exit outer loop

    # Restore default tab completion
    restore_tab_completion()

    # Sort remesh events by timestep
    result['remesh_events'].sort(key=lambda e: e['timestep_ms'])

    # Backward compatibility: populate single-remesh fields from first event
    if result['remesh_events']:
        first_event = result['remesh_events'][0]
        result['remesh_before_file'] = first_event['before_file']
        result['remesh_after_file'] = first_event['after_file']
        result['remesh_timestep_ms'] = first_event['timestep_ms']

    # Split files into chunks based on all remesh events
    # For backward compatibility, pre_remesh_files = before first remesh
    # post_remesh_files = after last remesh
    if result['remesh_events']:
        first_boundary = result['remesh_events'][0]['timestep_ms']
        last_boundary = result['remesh_events'][-1]['timestep_ms']

        result['pre_remesh_files'] = []
        result['post_remesh_files'] = []
        for f in xyz_files:
            ts = extract_timestep(f)
            if ts is not None:
                if ts < first_boundary:
                    result['pre_remesh_files'].append(f)
                elif ts >= last_boundary:
                    result['post_remesh_files'].append(f)

    print(f"\n{'='*60}")
    print(f"✅ Remesh configuration complete:")
    print(f"   Total remesh events: {len(result['remesh_events'])}")
    for i, event in enumerate(result['remesh_events'], 1):
        print(f"   #{i}: boundary at {event['timestep_ms']:.1f}ms")
        print(f"       before: {event['before_file']}")
        print(f"       after:  {event['after_file']}")
    print(f"{'='*60}")

    return result


def find_breathing_cycle_bounds(subject_name: str, flow_profile_path: str = None) -> Tuple[float, float]:
    """
    Find the time bounds of a single breathing cycle from the flow profile.

    Args:
        subject_name: Name of the subject (e.g., 'OSAMRI007')
        flow_profile_path: Optional explicit path to flow profile CSV file.
                          If provided, uses this path instead of auto-detection.

    Returns:
        Tuple of (first_crossing_time, last_crossing_time) in milliseconds
    """
    print(f"\nAnalyzing flow profile for subject {subject_name}...")

    # Use explicit path if provided, otherwise auto-detect
    if flow_profile_path is not None:
        flow_profile_path = Path(flow_profile_path)
        if not flow_profile_path.exists():
            print(f"❌ Flow profile file not found: {flow_profile_path}")
            return None, None
    else:
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

def preprocess_all_tables_parallel(xyz_files: List[Path], subject_name: str) -> List[Path]:
    """
    Add patch numbers to all surface tables using parallel processing.
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
    
    # Import parallel processing module
    try:
        from .parallel_csv_processing import process_csv_files_parallel, get_file_processing_stats
    except ImportError:
        print("⚠️  Parallel processing module not available, falling back to sequential processing")
        return preprocess_all_tables(xyz_files, subject_name)
    
    # Get file statistics
    stats = get_file_processing_stats(xyz_files)
    print(f"\n📊 CSV Processing Statistics:")
    print(f"   • Files to process: {stats['total_files']}")
    print(f"   • Total data size: {stats['total_size_gb']:.1f} GB")
    print(f"   • Average file size: {stats['avg_file_size_mb']:.1f} MB")
    print(f"   • Estimated sequential time: {stats['estimated_time_sequential_min']:.1f} minutes")
    
    # Process files in parallel
    processed_files = process_csv_files_parallel(xyz_files, subject_name)
    
    if not processed_files:
        print("❌ Parallel processing failed, falling back to sequential processing")
        return preprocess_all_tables(xyz_files, subject_name)
    
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


def detect_timestep_from_csv(xyz_file: Path) -> Dict:
    """
    Read actual timestep from CSV 'Time Step (s)' column.

    Only needs to read first CSV file (all tables have same timestep info).
    This is more accurate than guessing from filename patterns.

    Args:
        xyz_file: Path to any XYZ table CSV file

    Returns:
        {
            'time_seconds': float,     # e.g., 0.001
            'time_ms': float,          # e.g., 1.0
            'time_column_name': str,   # Column name found
            'source_file': Path
        }

    Raises:
        ValueError: If no time column found or file cannot be read
    """
    # Read just the first few rows for efficiency
    try:
        df = pd.read_csv(xyz_file, nrows=5)
    except Exception as e:
        raise ValueError(f"Could not read CSV file {xyz_file}: {e}")

    # Look for time column (various possible names)
    time_column_patterns = [
        'Time Step (s)',
        'Time Step(s)',
        'TimeStep (s)',
        'TimeStep(s)',
        'Time (s)',
        'Time(s)',
        'time_step',
        'timestep',
    ]

    time_column = None
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in time_column_patterns:
            if pattern.lower() in col_lower or col_lower in pattern.lower():
                time_column = col
                break
        if time_column:
            break

    if time_column is None:
        # Try regex match for any column containing "time" and "step"
        for col in df.columns:
            if 'time' in col.lower() and ('step' in col.lower() or '(s)' in col.lower()):
                time_column = col
                break

    if time_column is None:
        raise ValueError(f"Could not find time column in {xyz_file}. Available columns: {list(df.columns)}")

    # Get the time value (should be same for all rows in a file)
    time_seconds = float(df[time_column].iloc[0])
    time_ms = time_seconds * 1000.0

    return {
        'time_seconds': time_seconds,
        'time_ms': time_ms,
        'time_column_name': time_column,
        'source_file': xyz_file
    }


def detect_remesh_from_file_sizes(xyz_files: List[Path],
                                   size_change_threshold: float = 0.02) -> Dict:
    """
    Detect remesh events by scanning file sizes in chronological order.

    A 2% or greater file size change indicates a remesh.
    Small variations (<2%) are ignored as they may be due to floating point
    precision differences in data values, not mesh changes.

    Args:
        xyz_files: List of XYZ table files (should be in chronological order)
        size_change_threshold: Minimum relative size change to trigger detection (default 0.02 = 2%)

    Returns:
        {
            'has_remesh': bool,
            'remesh_events': [
                {
                    'before_file': Path,
                    'after_file': Path,
                    'before_size': int,
                    'after_size': int,
                    'size_change_percent': float,
                    'timestep_boundary': float  # timestep of after_file
                }
            ],
            'max_size_variation_percent': float,
            'threshold_percent': float
        }
    """
    if len(xyz_files) < 2:
        return {
            'has_remesh': False,
            'remesh_events': [],
            'max_size_variation_percent': 0.0,
            'threshold_percent': size_change_threshold * 100
        }

    remesh_events = []
    max_variation = 0.0

    # Get file sizes
    file_sizes = []
    for f in xyz_files:
        try:
            size = f.stat().st_size
            file_sizes.append((f, size))
        except OSError:
            continue

    if len(file_sizes) < 2:
        return {
            'has_remesh': False,
            'remesh_events': [],
            'max_size_variation_percent': 0.0,
            'threshold_percent': size_change_threshold * 100
        }

    # Compare consecutive files
    for i in range(len(file_sizes) - 1):
        before_file, before_size = file_sizes[i]
        after_file, after_size = file_sizes[i + 1]

        if before_size == 0:
            continue

        # Calculate relative change
        size_change = abs(after_size - before_size) / before_size
        size_change_percent = size_change * 100

        max_variation = max(max_variation, size_change_percent)

        # Check if this exceeds threshold
        if size_change >= size_change_threshold:
            # Extract timestep from after_file for boundary
            try:
                timestep = extract_timestep_from_filename(after_file)
            except ValueError:
                timestep = float(i + 1)  # Fallback to index

            remesh_events.append({
                'before_file': before_file,
                'after_file': after_file,
                'before_size': before_size,
                'after_size': after_size,
                'size_change_percent': size_change_percent,
                'timestep_boundary': timestep
            })

    return {
        'has_remesh': len(remesh_events) > 0,
        'remesh_events': remesh_events,
        'max_size_variation_percent': max_variation,
        'threshold_percent': size_change_threshold * 100
    }


def detect_breathing_cycle_enhanced(
    flow_profile_path: Optional[Path],
    xyz_files: List[Path],
    timestep_info: Optional[Dict] = None,
    is_complete_cycle: Optional[bool] = None,
    manual_times: Optional[Dict] = None
) -> Dict:
    """
    Enhanced breathing cycle detection with four modes.

    Mode M: Manual Input (CLI args or interactive)
        - User specifies all 3 key times directly in seconds
        - Highest priority - skips all auto-detection

    Mode A: Complete Single Cycle (user confirms 'a')
        - First time point = beginning of inhale
        - Last time point = end of exhale
        - Zero crossing in middle 80% = inhale-to-exhale transition

    Mode B: Multi-Cycle Detection (user confirms 'b')
        - Use existing algorithm: find all zero crossings
        - First crossing = beginning of inhale
        - Subsequent crossings = transitions

    Mode C: No Flow Profile
        - Calculate time history from: timestep × number of files
        - Assume all files are within one breathing cycle
        - Inhale-to-exhale = middle of total time

    Args:
        flow_profile_path: Path to flow profile CSV (None for Mode C)
        xyz_files: List of XYZ table files
        timestep_info: Output from detect_timestep_from_csv() (required for Mode C)
        is_complete_cycle: True=Mode A, False=Mode B, None=ask user (only if flow_profile exists)
        manual_times: Dict with 'start_s', 'transition_s', 'end_s' for manual mode (all in seconds)

    Returns:
        {
            'mode': 'complete_cycle' | 'multi_cycle' | 'no_flow_profile' | 'manual' | 'manual_cli',
            'start_time_ms': float,
            'end_time_ms': float,
            'inhale_exhale_transition_ms': float,
            'total_duration_ms': float,
            'flow_profile_used': bool,
            'zero_crossings_ms': List[float],  # All zero crossings found
            'user_input': Dict  # Original user input (for manual modes)
        }
    """
    result = {
        'mode': None,
        'start_time_ms': None,
        'end_time_ms': None,
        'inhale_exhale_transition_ms': None,
        'total_duration_ms': None,
        'flow_profile_used': False,
        'zero_crossings_ms': [],
        'user_input': None
    }

    # Mode M: Manual Input (highest priority - from CLI args)
    if manual_times is not None:
        start_s = manual_times.get('start_s')
        transition_s = manual_times.get('transition_s')
        end_s = manual_times.get('end_s')

        if start_s is not None and transition_s is not None and end_s is not None:
            # Validate order
            if not (start_s < transition_s < end_s):
                print(f"❌ ERROR: Times must be in order: start < transition < end")
                print(f"   Got: start={start_s}s, transition={transition_s}s, end={end_s}s")
                raise ValueError("Manual times are not in chronological order")

            # Validate positive
            if start_s < 0 or transition_s < 0 or end_s < 0:
                print(f"❌ ERROR: All times must be positive")
                raise ValueError("Manual times must be positive")

            result['mode'] = 'manual_cli'
            result['start_time_ms'] = start_s * 1000.0
            result['end_time_ms'] = end_s * 1000.0
            result['inhale_exhale_transition_ms'] = transition_s * 1000.0
            result['total_duration_ms'] = (end_s - start_s) * 1000.0
            result['flow_profile_used'] = flow_profile_path is not None and Path(flow_profile_path).exists()
            result['user_input'] = {
                'start_s': start_s,
                'transition_s': transition_s,
                'end_s': end_s
            }

            print(f"\n✓ Breathing cycle mode: Manual (CLI arguments)")
            print(f"  Start of inhale: {result['start_time_ms']:.1f} ms ({start_s:.3f} s)")
            print(f"  Inhale-exhale transition: {result['inhale_exhale_transition_ms']:.1f} ms ({transition_s:.3f} s)")
            print(f"  End of exhale: {result['end_time_ms']:.1f} ms ({end_s:.3f} s)")
            return result

    # Mode C: No Flow Profile
    if flow_profile_path is None or not Path(flow_profile_path).exists():
        result['mode'] = 'no_flow_profile'
        result['flow_profile_used'] = False

        if not xyz_files:
            print("❌ No XYZ files provided for breathing cycle detection")
            return result

        # Get timestep info if not provided
        if timestep_info is None:
            try:
                timestep_info = detect_timestep_from_csv(xyz_files[0])
            except ValueError as e:
                print(f"⚠ Could not detect timestep: {e}")
                print("  Using default 1ms timestep")
                timestep_info = {'time_ms': 1.0, 'time_seconds': 0.001}

        # Calculate time range from files
        timestep_ms = timestep_info['time_ms']
        n_files = len(xyz_files)

        # Get actual timesteps from first and last files
        try:
            first_timestep = extract_timestep_from_filename(xyz_files[0])
            last_timestep = extract_timestep_from_filename(xyz_files[-1])

            # Determine if timesteps are in seconds or milliseconds
            time_unit = detect_time_unit(xyz_files)
            if time_unit == 's':
                start_time_ms = first_timestep * 1000
                end_time_ms = last_timestep * 1000
            else:
                start_time_ms = first_timestep
                end_time_ms = last_timestep
        except ValueError:
            # Fallback: calculate from count and timestep
            start_time_ms = 0.0
            end_time_ms = timestep_ms * (n_files - 1)

        total_duration_ms = end_time_ms - start_time_ms
        inhale_exhale_ms = start_time_ms + (total_duration_ms / 2.0)

        result['start_time_ms'] = start_time_ms
        result['end_time_ms'] = end_time_ms
        result['inhale_exhale_transition_ms'] = inhale_exhale_ms
        result['total_duration_ms'] = total_duration_ms

        print(f"\n📊 Breathing Cycle Detection (No Flow Profile)")
        print(f"   Mode: Assumed single cycle from CSV files")
        print(f"   Timestep: {timestep_ms:.3f} ms (from CSV data)")
        print(f"   Files: {n_files}")
        print(f"   Start time: {start_time_ms:.1f} ms")
        print(f"   End time: {end_time_ms:.1f} ms")
        print(f"   Inhale-exhale transition: {inhale_exhale_ms:.1f} ms (midpoint)")

        return result

    # Load flow profile
    flow_profile_path = Path(flow_profile_path)
    print(f"\n📊 Loading flow profile: {flow_profile_path}")

    try:
        flow_df = pd.read_csv(flow_profile_path)
    except Exception as e:
        print(f"❌ Could not read flow profile: {e}")
        # Fall back to Mode C
        return detect_breathing_cycle_enhanced(None, xyz_files, timestep_info, None)

    # Extract time and flow data
    time_s = flow_df.iloc[:, 0].values  # Time in seconds
    flow = flow_df.iloc[:, 1].values    # Flow rate
    time_ms = time_s * 1000.0           # Convert to milliseconds

    result['flow_profile_used'] = True

    # Find all zero crossings
    zero_crossing_indices = np.where(np.diff(np.signbit(flow)))[0]
    zero_crossings_ms = [time_ms[i] for i in zero_crossing_indices]
    result['zero_crossings_ms'] = zero_crossings_ms

    print(f"   Time range: {time_ms[0]:.1f} ms to {time_ms[-1]:.1f} ms")
    print(f"   Zero crossings found: {len(zero_crossings_ms)}")

    # Ask user about breathing cycle mode if not specified
    manual_mode_selected = False
    if is_complete_cycle is None:
        print(f"\n{'='*60}")
        print(f"Flow profile: {flow_profile_path.name}")
        print(f"\nHow should we detect breathing cycle boundaries?")
        print(f"  [a] Auto-Complete - First point = start of inhale, last = end of exhale")
        print(f"  [b] Auto-Detect   - Use zero-crossing detection for multiple cycles")
        print(f"  [m] Manual        - Specify the time points manually (in seconds)")
        print(f"{'='*60}")

        while True:
            try:
                response = input("> ").strip().lower()
                if response in ['a', 'y', 'yes']:
                    is_complete_cycle = True
                    break
                elif response in ['b', 'n', 'no']:
                    is_complete_cycle = False
                    break
                elif response in ['m', 'manual']:
                    manual_mode_selected = True
                    break
                else:
                    print("Please enter 'a', 'b', or 'm'")
            except (EOFError, KeyboardInterrupt):
                print("\n  Defaulting to automatic zero-crossing detection (Mode B)")
                is_complete_cycle = False
                break

    # Mode M: Manual Input (interactive)
    if manual_mode_selected:
        print(f"\n📝 Enter time points in SECONDS:")
        print(f"   (Flow profile time range: {time_ms[0]/1000:.3f}s to {time_ms[-1]/1000:.3f}s)")

        while True:
            try:
                start_s = float(input("  Begin of inhale (s): ").strip())
                transition_s = float(input("  Inhale-to-exhale transition (s): ").strip())
                end_s = float(input("  End of exhale (s): ").strip())

                # Validate order
                if not (start_s < transition_s < end_s):
                    print(f"❌ Times must be in order: start < transition < end")
                    print(f"   Got: {start_s}s < {transition_s}s < {end_s}s - try again")
                    continue

                # Validate positive
                if start_s < 0 or transition_s < 0 or end_s < 0:
                    print(f"❌ All times must be positive - try again")
                    continue

                # Warn if outside flow profile range (but allow it)
                if start_s < time_ms[0]/1000 or end_s > time_ms[-1]/1000:
                    print(f"⚠ Warning: Some times are outside flow profile range")

                result['mode'] = 'manual'
                result['start_time_ms'] = start_s * 1000.0
                result['end_time_ms'] = end_s * 1000.0
                result['inhale_exhale_transition_ms'] = transition_s * 1000.0
                result['total_duration_ms'] = (end_s - start_s) * 1000.0
                result['flow_profile_used'] = True
                result['user_input'] = {
                    'start_s': start_s,
                    'transition_s': transition_s,
                    'end_s': end_s
                }

                print(f"\n✓ Breathing cycle mode: Manual (interactive)")
                print(f"  Start of inhale: {result['start_time_ms']:.1f} ms ({start_s:.3f} s)")
                print(f"  Inhale-exhale transition: {result['inhale_exhale_transition_ms']:.1f} ms ({transition_s:.3f} s)")
                print(f"  End of exhale: {result['end_time_ms']:.1f} ms ({end_s:.3f} s)")
                return result

            except ValueError:
                print("❌ Invalid input - please enter numeric values")
                continue
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled manual input - falling back to Mode B")
                is_complete_cycle = False
                break

    # Mode A: Complete Single Cycle
    if is_complete_cycle:
        result['mode'] = 'complete_cycle'
        result['start_time_ms'] = time_ms[0]
        result['end_time_ms'] = time_ms[-1]
        result['total_duration_ms'] = time_ms[-1] - time_ms[0]

        # Find zero crossing in middle 80% of the data
        n_points = len(time_ms)
        middle_start = int(n_points * 0.1)
        middle_end = int(n_points * 0.9)

        # Find zero crossings within middle 80%
        middle_crossings = [
            zc for zc in zero_crossings_ms
            if time_ms[middle_start] <= zc <= time_ms[middle_end]
        ]

        if middle_crossings:
            # Use the first zero crossing in the middle region
            result['inhale_exhale_transition_ms'] = middle_crossings[0]
        elif zero_crossings_ms:
            # Fallback: use first zero crossing overall
            result['inhale_exhale_transition_ms'] = zero_crossings_ms[0]
        else:
            # No zero crossings: use midpoint
            result['inhale_exhale_transition_ms'] = (time_ms[0] + time_ms[-1]) / 2.0

        print(f"\n✓ Breathing cycle mode: Complete single cycle")
        print(f"  Start of inhale: {result['start_time_ms']:.1f} ms")
        print(f"  Inhale-exhale transition: {result['inhale_exhale_transition_ms']:.1f} ms")
        print(f"  End of exhale: {result['end_time_ms']:.1f} ms")

    # Mode B: Multi-Cycle Detection
    else:
        result['mode'] = 'multi_cycle'

        if len(zero_crossings_ms) >= 2:
            result['start_time_ms'] = zero_crossings_ms[0]
            result['end_time_ms'] = zero_crossings_ms[-1]

            # Second zero crossing is inhale-to-exhale
            if len(zero_crossings_ms) >= 2:
                result['inhale_exhale_transition_ms'] = zero_crossings_ms[1]
            else:
                result['inhale_exhale_transition_ms'] = (zero_crossings_ms[0] + zero_crossings_ms[-1]) / 2.0

            result['total_duration_ms'] = result['end_time_ms'] - result['start_time_ms']

            print(f"\n✓ Breathing cycle mode: Multi-cycle detection")
            print(f"  First zero crossing (start of inhale): {result['start_time_ms']:.1f} ms")
            print(f"  Second zero crossing (inhale-exhale): {result['inhale_exhale_transition_ms']:.1f} ms")
            print(f"  Last zero crossing (end of exhale): {result['end_time_ms']:.1f} ms")
        else:
            # Not enough zero crossings - use full range with midpoint
            result['start_time_ms'] = time_ms[0]
            result['end_time_ms'] = time_ms[-1]
            result['inhale_exhale_transition_ms'] = (time_ms[0] + time_ms[-1]) / 2.0
            result['total_duration_ms'] = time_ms[-1] - time_ms[0]

            print(f"\n⚠ Not enough zero crossings found ({len(zero_crossings_ms)})")
            print(f"  Using full time range with midpoint transition")
            print(f"  Start: {result['start_time_ms']:.1f} ms")
            print(f"  Transition: {result['inhale_exhale_transition_ms']:.1f} ms")
            print(f"  End: {result['end_time_ms']:.1f} ms")

    return result


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


def create_metadata_json(subject_name: str, output_dir: Path,
                         timestep_info: Dict = None,
                         breathing_cycle: Dict = None,
                         remesh_info: Dict = None,
                         light_hdf5_timestep_ms: float = None) -> Path:
    """
    Create the system-generated metadata JSON file.

    This file contains all auto-detected configuration that shouldn't be
    edited by users. It's separate from picked_points.json which users edit.

    Args:
        subject_name: Subject identifier
        output_dir: Directory to save the metadata file
        timestep_info: Dict from detect_timestep_from_csv()
        breathing_cycle: Dict from detect_breathing_cycle_enhanced()
        remesh_info: Dict from detect_remesh_from_file_sizes()
        light_hdf5_timestep_ms: Timestep used for lightweight HDF5 extraction

    Returns:
        Path to the created metadata JSON file
    """
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "subject": subject_name,
        "created": datetime.now().isoformat(),
        "phase1_complete": True,

        "timestep_info": timestep_info or {
            "time_ms": 1.0,
            "time_column_name": "unknown",
            "source_file": "unknown"
        },

        "breathing_cycle": breathing_cycle or {
            "mode": "unknown",
            "start_time_ms": 0.0,
            "end_time_ms": 0.0,
            "inhale_exhale_transition_ms": 0.0,
            "flow_profile_used": False
        },

        "remesh_info": remesh_info or {
            "has_remesh": False,
            "remesh_events": [],
            "max_size_variation_percent": 0.0,
            "detection_threshold_percent": 2.0
        },

        "light_hdf5_info": {
            "timestep_ms": light_hdf5_timestep_ms,
            "filename": f"{subject_name}_cfd_data_light.h5" if light_hdf5_timestep_ms else None
        },

        # This will be populated by Phase 2 when user provides picked points
        "post_remesh_mappings": []
    }

    output_path = output_dir / f"{subject_name}_metadata.json"

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✅ Created metadata JSON: {output_path}")
    return output_path


def create_picked_points_template(subject_name: str, output_dir: Path,
                                  timestep_ms: float = None,
                                  light_hdf5_filename: str = None) -> Path:
    """
    Create an empty picked points template JSON for user to fill in.

    This file is portable and designed to be:
    1. Copied to local machine with lightweight HDF5
    2. Filled in using point picker GUI
    3. Copied back to cluster for Phase 2 processing

    Args:
        subject_name: Subject identifier
        output_dir: Directory to save the template
        timestep_ms: Timestep of the lightweight HDF5 file
        light_hdf5_filename: Name of the lightweight HDF5 file

    Returns:
        Path to the created template JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = {
        "subject": subject_name,
        "created_from": light_hdf5_filename or f"{subject_name}_cfd_data_light.h5",
        "timestep_ms": timestep_ms,
        "locations": [
            {
                "id": 1,
                "description": "Example: Posterior soft palate - UPDATE THIS",
                "patch_number": 0,
                "face_indices": [0],
                "coordinates": [0.0, 0.0, 0.0]
            }
        ],
        "_instructions": {
            "step1": "Copy this file and the lightweight HDF5 to your local machine",
            "step2": "Run: python src/main.py --subject {subject} --point-picker --light-h5",
            "step3": "Pick points interactively using the GUI",
            "step4": "Copy this file back to the cluster results folder",
            "step5": "Run Phase 2: python src/main.py --subject {subject} --plotting"
        }
    }

    output_path = output_dir / f"{subject_name}_picked_points.json"

    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"✅ Created picked points template: {output_path}")
    return output_path


def load_picked_points(subject_name: str, results_dir: Path = None) -> Optional[Dict]:
    """
    Load the picked points JSON file.

    Args:
        subject_name: Subject identifier
        results_dir: Results directory (defaults to {subject}_results/)

    Returns:
        Dict with picked points or None if not found
    """
    if results_dir is None:
        results_dir = Path(f"{subject_name}_results")

    picked_points_path = results_dir / f"{subject_name}_picked_points.json"

    if not picked_points_path.exists():
        return None

    with open(picked_points_path, 'r') as f:
        return json.load(f)


def load_metadata(subject_name: str, results_dir: Path = None) -> Optional[Dict]:
    """
    Load the metadata JSON file.

    Args:
        subject_name: Subject identifier
        results_dir: Results directory (defaults to {subject}_results/)

    Returns:
        Dict with metadata or None if not found
    """
    if results_dir is None:
        results_dir = Path(f"{subject_name}_results")

    metadata_path = results_dir / f"{subject_name}_metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_and_merge_configs(subject_name: str, results_dir: Path = None) -> Dict:
    """
    Load and merge picked_points.json and metadata.json for Phase 2 processing.

    This function:
    1. Loads both JSON files from the results directory
    2. Merges them into the format expected by the tracking system
    3. Falls back to legacy tracking_locations.json if new files don't exist

    Args:
        subject_name: Subject identifier
        results_dir: Results directory (defaults to {subject}_results/)

    Returns:
        Merged configuration dict ready for tracking
    """
    if results_dir is None:
        results_dir = Path(f"{subject_name}_results")
    else:
        results_dir = Path(results_dir)

    # Try new split file format first
    picked_points = load_picked_points(subject_name, results_dir)
    metadata = load_metadata(subject_name, results_dir)

    if picked_points is not None and metadata is not None:
        print(f"✅ Loading split JSON format (picked_points + metadata)")

        # Filter out template/example locations
        valid_locations = [
            loc for loc in picked_points.get("locations", [])
            if loc.get("patch_number", 0) != 0 or
               loc.get("coordinates", [0, 0, 0]) != [0.0, 0.0, 0.0]
        ]

        if not valid_locations:
            print("⚠️  No valid picked points found (only template entries)")
            print("   Please run the point picker to select anatomical landmarks")

        # Merge into tracking config format
        merged_config = {
            "subject": subject_name,
            "source": "split_json",
            "locations": valid_locations,
            "combinations": [],  # Can be added later if needed

            # Include metadata for reference
            "timestep_info": metadata.get("timestep_info", {}),
            "breathing_cycle": metadata.get("breathing_cycle", {}),
            "remesh_info": metadata.get("remesh_info", {}),
            "post_remesh_mappings": metadata.get("post_remesh_mappings", [])
        }

        return merged_config

    # Fall back to legacy tracking_locations.json
    legacy_path = results_dir / f"{subject_name}_tracking_locations.json"
    if legacy_path.exists():
        print(f"📂 Using legacy tracking_locations.json format")
        with open(legacy_path, 'r') as f:
            legacy_config = json.load(f)
            legacy_config["source"] = "legacy_tracking_locations"
            return legacy_config

    # Also check root directory for backward compatibility
    root_legacy_path = Path(f"{subject_name}_tracking_locations.json")
    if root_legacy_path.exists():
        print(f"📂 Using legacy tracking_locations.json from root directory")
        with open(root_legacy_path, 'r') as f:
            legacy_config = json.load(f)
            legacy_config["source"] = "legacy_root"
            return legacy_config

    # No config found
    print(f"❌ No tracking configuration found for {subject_name}")
    print(f"   Expected files:")
    print(f"   - {results_dir}/{subject_name}_picked_points.json")
    print(f"   - {results_dir}/{subject_name}_metadata.json")
    print(f"   Or legacy: {results_dir}/{subject_name}_tracking_locations.json")

    return {
        "subject": subject_name,
        "source": "none",
        "locations": [],
        "combinations": [],
        "error": "No tracking configuration found"
    }


def save_post_remesh_mappings(subject_name: str, mappings: List[Dict],
                               results_dir: Path = None) -> Path:
    """
    Save post-remesh mappings to the metadata.json file.

    This is called during Phase 2 after coordinate matching is performed
    for subjects with remesh events.

    Args:
        subject_name: Subject identifier
        mappings: List of remesh mapping dicts
        results_dir: Results directory

    Returns:
        Path to the updated metadata JSON file
    """
    if results_dir is None:
        results_dir = Path(f"{subject_name}_results")
    else:
        results_dir = Path(results_dir)

    metadata_path = results_dir / f"{subject_name}_metadata.json"

    if not metadata_path.exists():
        print(f"⚠️  Metadata file not found: {metadata_path}")
        print(f"   Creating new metadata file with remesh mappings")
        metadata = {
            "subject": subject_name,
            "created": None,
            "phase1_complete": False,
            "post_remesh_mappings": mappings
        }
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata["post_remesh_mappings"] = mappings

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✅ Saved post-remesh mappings to: {metadata_path}")
    return metadata_path