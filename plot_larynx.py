import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
tp_df = pd.read_csv('AirwaySurfaceTotalPressureForce.csv')
dvdt_df = pd.read_csv('dvdt.csv')
fdotv_df = pd.read_csv('fdotv.csv')

# Skip first 10 points for all dataframes
start_index = 10
tp_df = tp_df.iloc[start_index:]
dvdt_df = dvdt_df.iloc[start_index:]
fdotv_df = fdotv_df.iloc[start_index:]

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(15, 8))

# Plot TP on primary y-axis (left)
color1 = '#1f77b4'  # blue
ln1 = ax1.plot(tp_df['Time'], tp_df['TP_Larynx Monitor: TP_Larynx Monitor (N)'],
         color=color1, label='Larynx TP (N)', linewidth=2)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Force (N)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)

# Create first secondary y-axis (right)
ax2 = ax1.twinx()
color2 = '#ff7f0e'  # orange
ln2 = ax2.plot(dvdt_df['Time'], dvdt_df['dvdt_Larynx Monitor: dvdt_Larynx Monitor (m^2)'],
         color=color2, label='Larynx dvdt (m/s)', linewidth=2)
ax2.set_ylabel('Velocity (m/s)', color=color2, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color2)

# Create second secondary y-axis (far right)
ax3 = ax1.twinx()
# Offset the second secondary axis
ax3.spines['right'].set_position(('outward', 60))
color3 = '#2ca02c'  # green
ln3 = ax3.plot(fdotv_df['Time'], fdotv_df['fdotv_Larynx_Monitor_pts: fdotv_Larynx_Monitor_pts (m^2)'],
         color=color3, label='Larynx fdotv (W)', linewidth=2)
ax3.set_ylabel('Power (W)', color=color3, fontsize=12)
ax3.tick_params(axis='y', labelcolor=color3)

# Calculate the ranges and limits for each axis to align zeros
def get_aligned_limits(data):
    data_min = np.min(data)
    data_max = np.max(data)
    # Calculate how far zero is from each end
    dist_to_min = abs(0 - data_min)
    dist_to_max = abs(data_max - 0)
    # Use the larger distance for both sides to center zero
    max_dist = max(dist_to_min, dist_to_max) * 1.1
    return -max_dist, max_dist

# Set aligned limits for each axis
tp_limits = get_aligned_limits(tp_df['TP_Larynx Monitor: TP_Larynx Monitor (N)'])
dvdt_limits = get_aligned_limits(dvdt_df['dvdt_Larynx Monitor: dvdt_Larynx Monitor (m^2)'])
fdotv_limits = get_aligned_limits(fdotv_df['fdotv_Larynx_Monitor_pts: fdotv_Larynx_Monitor_pts (m^2)'])

ax1.set_ylim(tp_limits)
ax2.set_ylim(dvdt_limits)
ax3.set_ylim(fdotv_limits)

# Add horizontal line at y=0 for reference
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Format axis ticks
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))

# Add title
plt.title('Larynx Measurements Over Time (Excluding First 10 Points)\nZero-Aligned Y-Axes', fontsize=14, pad=20)

# Add legend with larger font
lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right', fontsize=12)

# Add grid
ax1.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('larynx_measurements.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot has been saved as 'larynx_measurements.png'")

# Print the value ranges for each measurement (excluding first 10 points)
print("\nValue ranges (excluding first 10 points):")
print(f"TP (N): {tp_df['TP_Larynx Monitor: TP_Larynx Monitor (N)'].min():.2e} to {tp_df['TP_Larynx Monitor: TP_Larynx Monitor (N)'].max():.2e}")
print(f"dvdt (m/s): {dvdt_df['dvdt_Larynx Monitor: dvdt_Larynx Monitor (m^2)'].min():.2e} to {dvdt_df['dvdt_Larynx Monitor: dvdt_Larynx Monitor (m^2)'].max():.2e}")
print(f"fdotv (W): {fdotv_df['fdotv_Larynx_Monitor_pts: fdotv_Larynx_Monitor_pts (m^2)'].min():.2e} to {fdotv_df['fdotv_Larynx_Monitor_pts: fdotv_Larynx_Monitor_pts (m^2)'].max():.2e}") 
