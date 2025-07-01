#!/usr/bin/env python3
"""
Painted Region Analysis Pipeline

This standalone tool applies the same CFD analysis workflow used for tracked points
to painted surface regions. It generates time-series analysis, correlations, and 
reports for painted regions without modifying the existing pipeline.

Usage:
    python -m src.surface_painting.painted_region_pipeline --painted-coords coords.json --subject less1mmeshOSAMRI007
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from scipy.signal import savgol_filter, find_peaks
from scipy.spatial import cKDTree

# Import existing utilities (without modifying them)
sys.path.append('.')
from src.utils.file_processing import get_xyz_files_in_chronological_order, extract_timestep_from_filename


def apply_smoothing(data, window_size=20):
    """Apply smoothing function (matching existing pipeline logic)."""
    if len(data) < window_size:
        return data.copy()
    smoothed = data.copy()
    half_window = window_size // 2
    for i in range(half_window, len(data) - half_window):
        smoothed[i] = np.mean(data[i-half_window:i+half_window+1])
    return smoothed


class PaintedRegionPipeline:
    """Standalone pipeline for analyzing painted regions using existing CFD analysis methods."""
    
    def __init__(self, subject: str, painted_coords_file: Path):
        """
        Initialize the painted region pipeline.
        
        Args:
            subject: Subject identifier (e.g., 'less1mmeshOSAMRI007')
            painted_coords_file: Path to painted coordinates JSON file
        """
        self.subject = subject
        self.painted_coords_file = painted_coords_file
        
        # Analysis settings (matching existing pipeline)
        self.LABEL_SIZE = 14
        self.TITLE_SIZE = 17
        
        # Data storage
        self.painted_regions = {}
        self.regional_data = {}
        self.regional_timeseries = {}
        
        print(f"🎨 Initializing Painted Region Pipeline for {subject}")
        
    def load_painted_coordinates(self) -> Dict:
        """Load painted region coordinates."""
        print(f"📍 Loading painted coordinates from: {self.painted_coords_file}")
        
        with open(self.painted_coords_file, 'r') as f:
            coords_data = json.load(f)
        
        print(f"Found {len(coords_data['regions'])} painted regions:")
        for region_name, region_data in coords_data['regions'].items():
            print(f"  {region_name}: {region_data['point_count']:,} points")
            
        return coords_data
    
    def find_csv_files(self) -> List[Path]:
        """Find all CSV files for the subject using existing file processing logic."""
        # Use existing file processing utilities
        csv_files = get_xyz_files_in_chronological_order(self.subject)
        
        if not csv_files:
            raise FileNotFoundError(f"No XYZ CSV files found for subject: {self.subject}")
        
        print(f"📁 Found {len(csv_files)} CSV files for analysis")
        
        return csv_files
    
    def map_regions_to_cfd_data(self, coords_data: Dict, csv_files: List[Path]) -> Dict:
        """Map painted regions to CFD data across all timesteps."""
        print(f"🗺️  Mapping painted regions to CFD data across {len(csv_files)} timesteps...")
        
        regional_data = {}
        
        # Initialize regional data structure
        for region_name in coords_data['regions'].keys():
            regional_data[region_name] = {
                'timesteps': [],
                'data_points': [],
                'mean_values': {}
            }
        
        # Process each timestep
        for i, csv_file in enumerate(csv_files):
            if i % 10 == 0:
                print(f"  Processing timestep {i+1}/{len(csv_files)}: {csv_file.name}")
            
            try:
                # Load CFD data for this timestep
                df = pd.read_csv(csv_file, low_memory=False)
                
                # Extract timestep
                timestep = extract_timestep_from_filename(csv_file)
                
                # Map each region
                for region_name, region_info in coords_data['regions'].items():
                    mapped_data = self.map_single_region_to_timestep(
                        region_info, df, timestep
                    )
                    
                    if mapped_data is not None:
                        regional_data[region_name]['timesteps'].append(timestep)
                        regional_data[region_name]['data_points'].append(mapped_data)
                        
            except Exception as e:
                print(f"    Warning: Error processing {csv_file.name}: {e}")
                continue
        
        # Compute mean time series for each region
        for region_name in regional_data.keys():
            regional_data[region_name] = self.compute_regional_timeseries(
                regional_data[region_name]
            )
        
        return regional_data
    
    def map_single_region_to_timestep(self, region_info: Dict, df: pd.DataFrame, 
                                    timestep: float, tolerance: float = 0.002) -> Optional[Dict]:
        """Map a single region to CFD data for one timestep."""
        # Get painted coordinates
        painted_coords = np.array(region_info['coordinates'])
        
        # Get CFD coordinates
        if not all(col in df.columns for col in ['X (m)', 'Y (m)', 'Z (m)']):
            return None
            
        cfd_coords = df[['X (m)', 'Y (m)', 'Z (m)']].values
        
        # Find matching points using spatial search
        tree = cKDTree(cfd_coords)
        distances, indices = tree.query(painted_coords, distance_upper_bound=tolerance)
        
        # Filter valid matches
        valid_mask = distances < tolerance
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) == 0:
            return None
        
        # Remove duplicates
        unique_indices = list(set(valid_indices))
        
        # Extract CFD data for matched points
        matched_data = df.iloc[unique_indices]
        
        # Compute mean values for key quantities
        mean_data = {
            'timestep': timestep,
            'matched_points': len(unique_indices),
            'total_painted_points': len(painted_coords)
        }
        
        # Calculate means for CFD quantities (matching existing pipeline)
        cfd_quantities = {
            'VdotN': 'VdotN',
            'Total Pressure (Pa)': 'Total Pressure (Pa)',
            'Static Pressure (Pa)': 'Static Pressure (Pa)',
            'Velocity: Magnitude (m/s)': 'Velocity: Magnitude (m/s)',
            'Area: Magnitude (m^2)': 'Area: Magnitude (m^2)'
        }
        
        for col_name, key in cfd_quantities.items():
            if col_name in matched_data.columns:
                values = matched_data[col_name].values
                mean_data[key] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': len(values)
                }
        
        return mean_data
    
    def compute_regional_timeseries(self, region_data: Dict) -> Dict:
        """Compute time series data for a region (matching existing pipeline approach)."""
        if not region_data['data_points']:
            return region_data
        
        # Sort by timestep
        timesteps = np.array(region_data['timesteps'])
        data_points = region_data['data_points']
        
        sorted_indices = np.argsort(timesteps)
        sorted_timesteps = timesteps[sorted_indices]
        sorted_data = [data_points[i] for i in sorted_indices]
        
        # Extract time series for each quantity
        quantities = ['VdotN', 'Total Pressure (Pa)', 'Static Pressure (Pa)', 
                     'Velocity: Magnitude (m/s)', 'Area: Magnitude (m^2)']
        
        timeseries = {
            'timesteps': sorted_timesteps.tolist(),
            'quantities': {}
        }
        
        for quantity in quantities:
            if quantity in sorted_data[0]:
                values = [dp[quantity]['mean'] for dp in sorted_data if quantity in dp]
                
                if len(values) > 0:
                    values = np.array(values)
                    
                    # Apply smoothing (using existing pipeline method)
                    smooth_values = apply_smoothing(values)
                    
                    # Compute acceleration for velocity
                    if quantity == 'VdotN':
                        dt = np.mean(np.diff(sorted_timesteps))
                        acceleration = np.gradient(smooth_values, dt)
                        timeseries['quantities']['Acceleration'] = {
                            'values': acceleration.tolist(),
                            'smooth_values': acceleration.tolist(),
                            'units': 'm/s²'
                        }
                    
                    timeseries['quantities'][quantity] = {
                        'values': values.tolist(),
                        'smooth_values': smooth_values.tolist(),
                        'units': self.get_units(quantity)
                    }
        
        region_data['timeseries'] = timeseries
        return region_data
    
    def get_units(self, quantity: str) -> str:
        """Get units for quantities."""
        unit_map = {
            'VdotN': 'm/s',
            'Total Pressure (Pa)': 'Pa',
            'Static Pressure (Pa)': 'Pa',
            'Velocity: Magnitude (m/s)': 'm/s',
            'Area: Magnitude (m^2)': 'm²'
        }
        return unit_map.get(quantity, '')
    
    def find_zero_crossings(self, times: np.ndarray, values: np.ndarray) -> List[float]:
        """Find zero crossings (matching existing pipeline logic)."""
        crossings = []
        for i in range(len(values) - 1):
            if values[i] * values[i + 1] < 0:  # Sign change
                # Linear interpolation
                t_cross = times[i] - values[i] * (times[i + 1] - times[i]) / (values[i + 1] - values[i])
                crossings.append(float(t_cross))
        return crossings
    
    def generate_painted_region_report(self, output_file: Path = None):
        """Generate comprehensive report for painted regions (matching existing 3x3 format)."""
        if output_file is None:
            output_file = Path(f"{self.subject}_painted_regions_analysis.pdf")
        
        print(f"📊 Generating painted region analysis report: {output_file}")
        
        with PdfPages(output_file) as pdf:
            # Create 3x3 panel for each region (matching existing pipeline)
            for region_name, region_data in self.regional_data.items():
                if 'timeseries' not in region_data:
                    continue
                    
                self.create_region_analysis_page(region_name, region_data, pdf)
        
        print(f"✅ Painted region analysis report saved: {output_file}")
    
    def create_region_analysis_page(self, region_name: str, region_data: Dict, pdf):
        """Create analysis page for a region (3x3 format matching existing pipeline)."""
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle(f'Painted Region Analysis: {region_name} ({self.subject})', 
                    fontsize=20, fontweight='bold')
        
        timeseries = region_data['timeseries']
        times = np.array(timeseries['timesteps'])
        
        # Get data
        velocity_data = timeseries['quantities'].get('VdotN', {})
        pressure_data = timeseries['quantities'].get('Total Pressure (Pa)', {})
        accel_data = timeseries['quantities'].get('Acceleration', {})
        
        if not velocity_data or not pressure_data:
            plt.figtext(0.5, 0.5, f'Insufficient data for {region_name}', 
                       ha='center', va='center', fontsize=16)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)
            return
        
        velocity = np.array(velocity_data['smooth_values']) * 1000  # Convert to mm/s
        pressure = np.array(pressure_data['smooth_values'])
        
        # Row 1: Time series plots
        # Velocity vs Time
        ax = axes[0, 0]
        ax.plot(times, velocity, 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Regional Velocity', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add zero crossings
        zero_crossings = self.find_zero_crossings(times, velocity)
        for zc in zero_crossings:
            ax.axvline(x=zc, color='red', linestyle='--', alpha=0.7)
        
        # Pressure vs Time
        ax = axes[0, 1]
        ax.plot(times, pressure, 'g-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Total Pressure (Pa)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Regional Pressure', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Acceleration vs Time (if available)
        ax = axes[0, 2]
        if accel_data:
            acceleration = np.array(accel_data['smooth_values']) * 1000  # Convert to mm/s²
            ax.plot(times, acceleration, 'r-', linewidth=2, alpha=0.8)
            ax.set_ylabel('a⃗·n⃗ (mm/s²)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Regional Acceleration', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Row 2: Phase plots
        # Pressure vs Velocity
        ax = axes[1, 0]
        ax.scatter(velocity, pressure, c=times, cmap='viridis', alpha=0.6, s=20)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Total Pressure (Pa)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Pressure vs Velocity', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Acceleration vs Velocity (if available)
        ax = axes[1, 1]
        if accel_data:
            ax.scatter(acceleration, velocity, c=times, cmap='viridis', alpha=0.6, s=20)
            ax.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('v⃗·n⃗ (mm/s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Velocity vs Acceleration', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Pressure vs Acceleration (if available)
        ax = axes[1, 2]
        if accel_data:
            ax.scatter(acceleration, pressure, c=times, cmap='viridis', alpha=0.6, s=20)
            ax.set_xlabel('a⃗·n⃗ (mm/s²)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('Total Pressure (Pa)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Pressure vs Acceleration', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Row 3: Analysis plots
        # Velocity histogram
        ax = axes[2, 0]
        ax.hist(velocity, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('v⃗·n⃗ (mm/s)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Velocity Distribution', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Pressure histogram
        ax = axes[2, 1]
        ax.hist(pressure, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Total Pressure (Pa)', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=self.LABEL_SIZE, fontweight='bold')
        ax.set_title('Pressure Distribution', fontsize=self.TITLE_SIZE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[2, 2]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""
Region Summary:
• Total Painted Points: {region_data['data_points'][0]['total_painted_points']:,}
• Avg Matched Points: {np.mean([dp['matched_points'] for dp in region_data['data_points']]):.0f}
• Time Points: {len(times):,}
• Time Span: {times[-1] - times[0]:.3f}s

Velocity Statistics:
• Mean: {velocity.mean():.1f} mm/s
• Std: {velocity.std():.1f} mm/s
• Range: [{velocity.min():.1f}, {velocity.max():.1f}] mm/s
• Zero Crossings: {len(zero_crossings)}

Pressure Statistics:
• Mean: {pressure.mean():.1f} Pa
• Std: {pressure.std():.1f} Pa
• Range: [{pressure.min():.1f}, {pressure.max():.1f}] Pa
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def run_analysis(self):
        """Run the complete painted region analysis pipeline."""
        try:
            # Load painted coordinates
            coords_data = self.load_painted_coordinates()
            
            # Find CSV files
            csv_files = self.find_csv_files()
            
            # Map regions to CFD data
            self.regional_data = self.map_regions_to_cfd_data(coords_data, csv_files)
            
            # Generate report
            self.generate_painted_region_report()
            
            print(f"🎉 Painted region analysis completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in painted region analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Painted Region Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze painted regions for a subject
  python -m src.surface_painting.painted_region_pipeline --painted-coords out_0100.000_ds_10_l2_aBE0.001_be_0.001_painted_coordinates.json --subject less1mmeshOSAMRI007
  
  # Custom output file
  python -m src.surface_painting.painted_region_pipeline --painted-coords coords.json --subject 2mmeshOSAMRI007 --output custom_analysis.pdf
        """
    )
    
    parser.add_argument('--painted-coords', required=True, type=Path,
                        help='Path to painted coordinates JSON file')
    
    parser.add_argument('--subject', required=True, type=str,
                        help='Subject identifier (e.g., less1mmeshOSAMRI007)')
    
    parser.add_argument('--output', type=Path, default=None,
                        help='Output PDF file (default: {subject}_painted_regions_analysis.pdf)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.painted_coords.exists():
        print(f"Error: Painted coordinates file not found: {args.painted_coords}")
        return
    
    try:
        # Initialize and run pipeline
        pipeline = PaintedRegionPipeline(args.subject, args.painted_coords)
        
        if args.output:
            pipeline.output_file = args.output
            
        pipeline.run_analysis()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 