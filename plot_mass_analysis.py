import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
tp_df = pd.read_csv('AirwaySurfaceTotalPressureForce.csv')
dvdt_df = pd.read_csv('dvdt.csv')

# Skip first 10 points for all dataframes
start_index = 10
tp_df = tp_df.iloc[start_index:]
dvdt_df = dvdt_df.iloc[start_index:]

# Set up the region and columns
region = 'Nasopharynx'
quadrant = 'Anterior'
tp_col = f'TP_{region}{quadrant} Monitor: TP_{region}{quadrant} Monitor (m^2)'
dvdt_col = f'dvdt_{region}{quadrant} Monitor: dvdt_{region}{quadrant} Monitor (m^2)'

# Get and scale the data
tp_data = tp_df[tp_col]
dvdt_data = dvdt_df[dvdt_col]

tp_range = max(abs(tp_data.max()), abs(tp_data.min()))
dvdt_range = max(abs(dvdt_data.max()), abs(dvdt_data.min()))

tp_scale = 10 ** (np.floor(np.log10(tp_range)))
dvdt_scale = 10 ** (np.floor(np.log10(dvdt_range)))

tp_data_scaled = tp_data / tp_scale
dvdt_data_scaled = dvdt_data / dvdt_scale

# Create figure with subplots for different mass values
masses = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Calculate overall max range for consistent scaling across subplots
max_ranges = []
for m in masses:
    diff_data = tp_data_scaled - m * dvdt_data_scaled
    max_abs_tp = max(abs(tp_data_scaled.max()), abs(tp_data_scaled.min()))
    max_abs_diff = max(abs(diff_data.max()), abs(diff_data.min()))
    max_ranges.append(max(max_abs_tp, max_abs_diff))

overall_max_range = max(max_ranges) * 1.1

# Create plots for each mass value
for i, m in enumerate(masses):
    ax = axes[i]
    
    # Calculate the difference with mass factor
    diff_data = tp_data_scaled - m * dvdt_data_scaled
    
    # Create scatter plot
    scatter = ax.scatter(diff_data, tp_data_scaled, alpha=0.5, 
                        c=range(len(tp_data)), cmap='viridis', s=20)
    
    # Add regression line
    z = np.polyfit(diff_data, tp_data_scaled, 1)
    p = np.poly1d(z)
    x_range = np.linspace(-overall_max_range, overall_max_range, 100)
    ax.plot(x_range, p(x_range), "r--", alpha=0.8, 
            label=f'slope: {z[0]:.2f}')
    
    # Calculate correlation
    corr = np.corrcoef(diff_data, tp_data_scaled)[0,1]
    
    # Add zero lines and grid
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Set consistent limits across all subplots
    ax.set_xlim(-overall_max_range, overall_max_range)
    ax.set_ylim(-overall_max_range, overall_max_range)
    
    # Add title and legend
    ax.set_title(f'm = {m:.1f}\nCorr: {corr:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

# Add common labels
fig.text(0.5, 0.02, 'TP-m·dv/dt (normalized)', ha='center', fontsize=12)
fig.text(0.02, 0.5, f'Total Pressure Force (m²) [×{tp_scale:.0e}]', 
         va='center', rotation='vertical', fontsize=12)

# Add main title
plt.suptitle(f'{region} {quadrant}\nEffect of Mass on TP vs (TP-m·dv/dt) Relationship', 
             fontsize=14, y=0.95)

# Adjust layout
plt.tight_layout(rect=[0.03, 0.03, 1, 0.92])

# Save the plot
plt.savefig('mass_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Mass analysis plot has been saved as 'mass_analysis.png'") 