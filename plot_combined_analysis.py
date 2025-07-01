import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Read the CSV files
tp_df = pd.read_csv('AirwaySurfaceTotalPressureForce.csv')
dvdt_df = pd.read_csv('dvdt.csv')

# Skip first 10 points for all dataframes
start_index = 10
tp_df = tp_df.iloc[start_index:]
dvdt_df = dvdt_df.iloc[start_index:]

def create_dvdt_vs_tp_plot(region, quadrant=None):
    """Create a scatter plot of dvdt vs TP"""
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Construct column names based on region and quadrant
    if quadrant:
        tp_col = f'TP_{region}{quadrant} Monitor: TP_{region}{quadrant} Monitor (m^2)'
        dvdt_col = f'dvdt_{region}{quadrant} Monitor: dvdt_{region}{quadrant} Monitor (m^2)'
        title_suffix = f' - {quadrant}'
        units = 'm²'
    else:
        tp_col = f'TP_{region} Monitor: TP_{region} Monitor (N)'
        dvdt_col = f'dvdt_{region} Monitor: dvdt_{region} Monitor (m^2)'
        title_suffix = ''
        units = 'N'
    
    # Get and scale the data
    tp_data = tp_df[tp_col]
    dvdt_data = dvdt_df[dvdt_col]
    
    tp_range = max(abs(tp_data.max()), abs(tp_data.min()))
    dvdt_range = max(abs(dvdt_data.max()), abs(dvdt_data.min()))
    
    tp_scale = 10 ** (np.floor(np.log10(tp_range)))
    dvdt_scale = 10 ** (np.floor(np.log10(dvdt_range)))
    
    tp_data_scaled = tp_data / tp_scale
    dvdt_data_scaled = dvdt_data / dvdt_scale
    
    # Calculate limits
    max_abs_tp = max(abs(tp_data_scaled.max()), abs(tp_data_scaled.min()))
    max_abs_dvdt = max(abs(dvdt_data_scaled.max()), abs(dvdt_data_scaled.min()))
    max_range = max(max_abs_tp, max_abs_dvdt) * 1.1
    
    # Create scatter plot
    scatter = ax.scatter(dvdt_data_scaled, tp_data_scaled, alpha=0.5, 
                        c=range(len(tp_data)), cmap='viridis', label='Data points')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time progression', fontsize=10)
    
    # Add regression line
    z = np.polyfit(dvdt_data_scaled, tp_data_scaled, 1)
    p = np.poly1d(z)
    x_range = np.linspace(-max_range, max_range, 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, 
            label=f'Linear fit (slope: {z[0]:.2f})')
    
    # Calculate correlation
    corr = np.corrcoef(dvdt_data_scaled, tp_data_scaled)[0,1]
    
    # Add zero lines and grid
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set limits and labels
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_xlabel(f'Surface Acceleration (m/s²) [×{dvdt_scale:.0e}]', fontsize=12)
    ax.set_ylabel(f'Total Pressure Force ({units}) [×{tp_scale:.0e}]', fontsize=12)
    
    # Add title and legend
    title = f'{region}{title_suffix}\nTotal Pressure Force vs Acceleration'
    subtitle = f'Correlation coefficient: {corr:.3f}'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, pad=20)
    ax.legend(fontsize=10)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    return fig

def create_diff_vs_tp_plot(region, quadrant=None):
    """Create a scatter plot of TP vs (TP-dvdt)"""
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Construct column names based on region and quadrant
    if quadrant:
        tp_col = f'TP_{region}{quadrant} Monitor: TP_{region}{quadrant} Monitor (m^2)'
        dvdt_col = f'dvdt_{region}{quadrant} Monitor: dvdt_{region}{quadrant} Monitor (m^2)'
        title_suffix = f' - {quadrant}'
        units = 'm²'
    else:
        tp_col = f'TP_{region} Monitor: TP_{region} Monitor (N)'
        dvdt_col = f'dvdt_{region} Monitor: dvdt_{region} Monitor (m^2)'
        title_suffix = ''
        units = 'N'
    
    # Get and scale the data
    tp_data = tp_df[tp_col]
    dvdt_data = dvdt_df[dvdt_col]
    
    tp_range = max(abs(tp_data.max()), abs(tp_data.min()))
    dvdt_range = max(abs(dvdt_data.max()), abs(dvdt_data.min()))
    
    tp_scale = 10 ** (np.floor(np.log10(tp_range)))
    dvdt_scale = 10 ** (np.floor(np.log10(dvdt_range)))
    
    tp_data_scaled = tp_data / tp_scale
    dvdt_data_scaled = dvdt_data / dvdt_scale
    
    # Calculate the difference
    diff_data = tp_data_scaled - dvdt_data_scaled
    
    # Calculate limits
    max_abs_tp = max(abs(tp_data_scaled.max()), abs(tp_data_scaled.min()))
    max_abs_diff = max(abs(diff_data.max()), abs(diff_data.min()))
    max_range = max(max_abs_tp, max_abs_diff) * 1.1
    
    # Create scatter plot with TP on y-axis and diff on x-axis
    scatter = ax.scatter(diff_data, tp_data_scaled, alpha=0.5, 
                        c=range(len(tp_data)), cmap='viridis', label='Data points')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time progression', fontsize=10)
    
    # Add regression line
    z = np.polyfit(diff_data, tp_data_scaled, 1)
    p = np.poly1d(z)
    x_range = np.linspace(-max_range, max_range, 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, 
            label=f'Linear fit (slope: {z[0]:.2f})')
    
    # Calculate correlation
    corr = np.corrcoef(diff_data, tp_data_scaled)[0,1]
    
    # Add zero lines and grid
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set limits and labels
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_xlabel('TP-dv/dt (normalized)', fontsize=12)
    ax.set_ylabel(f'Total Pressure Force ({units}) [×{tp_scale:.0e}]', fontsize=12)
    
    # Add title and legend
    title = f'{region}{title_suffix}\nTotal Pressure Force vs TP-dv/dt'
    subtitle = f'Correlation coefficient: {corr:.3f}'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, pad=20)
    ax.legend(fontsize=10)
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    return fig

# Define regions and quadrants
regions = ['Nasopharynx', 'Oropharynx', 'Larynx', 'Trachea']
quadrants = ['Anterior', 'Posterior', 'Left', 'Right']

# Create PDF with both types of plots
with PdfPages('combined_analysis.pdf') as pdf:
    # First create main region plots
    for region in regions:
        # Create and save dvdt vs TP plot
        fig = create_dvdt_vs_tp_plot(region)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Create and save TP-dvdt vs TP plot
        fig = create_diff_vs_tp_plot(region)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Then create quadrant plots
        for quadrant in quadrants:
            # Create and save dvdt vs TP plot
            fig = create_dvdt_vs_tp_plot(region, quadrant)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Create and save TP-dvdt vs TP plot
            fig = create_diff_vs_tp_plot(region, quadrant)
            pdf.savefig(fig)
            plt.close(fig)

print("All plots have been saved to 'combined_analysis.pdf'") 