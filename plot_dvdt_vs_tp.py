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

def create_scatter_plot(region, quadrant=None):
    """Create a scatter plot of (TP-dvdt) vs TP for a specific region and quadrant"""
    # Create figure
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
    
    # Get the data
    tp_data = tp_df[tp_col]
    dvdt_data = dvdt_df[dvdt_col]
    
    # Calculate scale factors to make ranges similar
    tp_range = max(abs(tp_data.max()), abs(tp_data.min()))
    dvdt_range = max(abs(dvdt_data.max()), abs(dvdt_data.min()))
    
    # Scale factor to make ranges similar (aiming for order of 1)
    tp_scale = 10 ** (np.floor(np.log10(tp_range)))
    dvdt_scale = 10 ** (np.floor(np.log10(dvdt_range)))
    
    # Scale the data
    tp_data_scaled = tp_data / tp_scale
    dvdt_data_scaled = dvdt_data / dvdt_scale
    
    # Calculate the difference (TP-dvdt)
    diff_data = tp_data_scaled - dvdt_data_scaled
    
    # Calculate appropriate axis limits for scaled data
    max_abs_tp = max(abs(tp_data_scaled.max()), abs(tp_data_scaled.min()))
    max_abs_diff = max(abs(diff_data.max()), abs(diff_data.min()))
    max_range = max(max_abs_tp, max_abs_diff) * 1.1  # Add 10% margin
    
    # Create scatter plot with scaled data
    scatter = ax.scatter(tp_data_scaled, diff_data, alpha=0.5, c=range(len(tp_data)), 
                        cmap='viridis', label='Data points')
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time progression', fontsize=10)
    
    # Calculate and plot linear regression
    z = np.polyfit(tp_data_scaled, diff_data, 1)
    p = np.poly1d(z)
    x_range = np.linspace(-max_range, max_range, 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, 
            label=f'Linear fit (slope: {z[0]:.2f})')
    
    # Calculate correlation coefficient
    corr = np.corrcoef(tp_data_scaled, diff_data)[0,1]
    
    # Add zero lines
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    # Set equal limits for both axes
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    
    # Label axes with scale information
    ax.set_xlabel(f'Total Pressure Force ({units}) [×{tp_scale:.0e}]', fontsize=14)
    ax.set_ylabel(f'TP-dv/dt (normalized)', fontsize=14)
    
    # Add title
    title = f'{region}{title_suffix}\nTP-dv/dt vs Total Pressure Force'
    subtitle = f'Correlation coefficient: {corr:.3f}'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, pad=20)
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Make axes equal to show true relationship
    ax.set_aspect('equal', adjustable='box')
    
    # Format axis ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Define regions and quadrants
regions = ['Nasopharynx', 'Oropharynx', 'Larynx', 'Trachea']
quadrants = ['Anterior', 'Posterior', 'Left', 'Right']

# Create PDF with all scatter plots
with PdfPages('dvdt_vs_tp_analysis.pdf') as pdf:
    # First create main region plots
    for region in regions:
        fig = create_scatter_plot(region)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Then create quadrant plots for each region
        for quadrant in quadrants:
            fig = create_scatter_plot(region, quadrant)
            pdf.savefig(fig)
            plt.close(fig)

print("All scatter plots have been saved to 'dvdt_vs_tp_analysis.pdf'") 