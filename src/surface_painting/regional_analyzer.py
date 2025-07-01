#!/usr/bin/env python3
"""
Regional Analyzer for Painted CFD Regions

This module provides advanced analysis capabilities for painted surface regions
mapped to CFD data. It generates comprehensive reports with statistics,
visualizations, and time-series analysis.

Usage:
    python -m src.surface_painting.regional_analyzer --mapping-results results.json --region-data-dir regional_cfd_data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy.signal import savgol_filter
import seaborn as sns


class RegionalAnalyzer:
    """Advanced analyzer for painted surface regions."""
    
    def __init__(self):
        """Initialize the regional analyzer."""
        # Plot settings
        plt.style.use('default')
        self.colors = {
            'region_1': '#FF3333',  # Red
            'region_2': '#33FF33',  # Green
            'region_3': '#3333FF',  # Blue
            'region_4': '#FFFF33',  # Yellow
            'region_5': '#FF33FF',  # Magenta
        }
        
        # Font sizes (matching existing pipeline)
        self.LABEL_SIZE = 14
        self.TITLE_SIZE = 17
        
    def load_mapping_results(self, results_file: Path) -> Dict:
        """
        Load mapping results from JSON file.
        
        Args:
            results_file: Path to mapping results JSON
            
        Returns:
            Dictionary with mapping results
        """
        print(f"Loading mapping results from: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"Found {results['summary']['total_regions']} regions")
        return results
    
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
    
    def generate_regional_comparison_report(self, mapping_results: Dict, output_file: Path = None):
        """
        Generate comprehensive regional comparison report.
        
        Args:
            mapping_results: Mapping results
            output_file: Output PDF file path
        """
        if output_file is None:
            output_file = Path("regional_analysis_report.pdf")
        
        print(f"\nGenerating regional comparison report: {output_file}")
        
        with PdfPages(output_file) as pdf:
            # Page 1: Regional Overview
            self.create_overview_page(mapping_results, pdf)
        
        print(f"✅ Regional analysis report saved: {output_file}")
    
    def create_overview_page(self, mapping_results: Dict, pdf):
        """Create regional overview page."""
        fig = plt.figure(figsize=(16, 12))
        
        # Main title
        fig.suptitle('Painted Region Analysis Overview', fontsize=20, fontweight='bold', y=0.95)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
        
        # Overview statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # Prepare table data
        table_data = [['Region', 'Painted Points', 'CFD Points', 'Best Tolerance']]
        
        for region_name, region_info in mapping_results['regions'].items():
            painted_points = region_info['painted_points']
            cfd_points = region_info['best_match_count']
            tolerance = region_info['best_tolerance']
            
            table_data.append([
                region_name, 
                f"{painted_points:,}",
                f"{cfd_points:,}", 
                tolerance or "N/A"
            ])
        
        # Create table
        table = ax1.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Regional Mapping Summary', fontsize=self.TITLE_SIZE, fontweight='bold', pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Advanced analysis for painted surface regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate regional analysis report
  python -m src.surface_painting.regional_analyzer --mapping-results region_mapping_results.json
        """
    )
    
    parser.add_argument('--mapping-results', required=True, type=Path,
                        help='Path to mapping results JSON file')
    
    parser.add_argument('--output', type=Path, default=None,
                        help='Output PDF file (default: regional_analysis_report.pdf)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.mapping_results.exists():
        print(f"Error: Mapping results file not found: {args.mapping_results}")
        return
    
    try:
        # Initialize analyzer
        analyzer = RegionalAnalyzer()
        
        # Load data
        mapping_results = analyzer.load_mapping_results(args.mapping_results)
        
        # Generate report
        analyzer.generate_regional_comparison_report(mapping_results, args.output)
        
        print(f"\n🎉 Regional analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 