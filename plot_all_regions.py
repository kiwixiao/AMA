import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

# Read the CSV files
tp_df = pd.read_csv('AirwaySurfaceTotalPressureForce.csv')
dvdt_df = pd.read_csv('dvdt.csv')
fdotv_df = pd.read_csv('fdotv.csv')
flow_df = pd.read_csv('FlowProfile.csv')

# Skip first 10 points for all dataframes
start_index = 10
tp_df = tp_df.iloc[start_index:]
dvdt_df = dvdt_df.iloc[start_index:]
fdotv_df = fdotv_df.iloc[start_index:]

def get_aligned_limits(data):
    data_min = np.min(data)
    data_max = np.max(data)
    dist_to_min = abs(0 - data_min)
    dist_to_max = abs(data_max - 0)
    max_dist = max(dist_to_min, dist_to_max) * 1.1
    return -max_dist, max_dist

def analyze_dynamics(tp_data, dvdt_data, fdotv_data, flow_time, flow_data):
    """Analyze the airway dynamics and return a description"""
    # Interpolate flow data to match the length of other data
    time = np.linspace(0, 1, len(tp_data))  # Normalize time to [0,1]
    flow_time_norm = (flow_time - flow_time.min()) / (flow_time.max() - flow_time.min())
    flow_interp = interp1d(flow_time_norm, flow_data, bounds_error=False, fill_value="extrapolate")
    flow_resampled = flow_interp(time)
    
    # Calculate key metrics
    force_mean = np.mean(tp_data)
    force_std = np.std(tp_data)
    accel_mean = np.mean(dvdt_data)
    power_mean = np.mean(fdotv_data)
    
    # Analyze force-velocity alignment
    alignment_ratio = np.sum(fdotv_data > 0) / len(fdotv_data)
    
    # Analyze phase relationships
    inhale_mask = flow_resampled > 0
    exhale_mask = flow_resampled < 0
    
    force_inhale = np.mean(tp_data[inhale_mask])
    force_exhale = np.mean(tp_data[exhale_mask])
    
    # Generate analysis text
    analysis = []
    analysis.append("Airway Dynamic Analysis:")
    
    # Force characteristics
    if abs(force_mean) > force_std:
        analysis.append(f"• Predominant {'positive' if force_mean > 0 else 'negative'} pressure force")
    else:
        analysis.append("• Highly variable pressure force")
    
    # Force-acceleration patterns
    if alignment_ratio > 0.6:
        analysis.append("• Force mostly assists surface motion")
    elif alignment_ratio < 0.4:
        analysis.append("• Force mostly opposes surface motion")
    else:
        analysis.append("• Mixed force-motion interaction")
    
    # Phase behavior
    if force_inhale > force_exhale:
        analysis.append("• Stronger forces during inhalation")
    else:
        analysis.append("• Stronger forces during exhalation")
    
    # Stability assessment based on acceleration
    if np.std(dvdt_data) > abs(np.mean(dvdt_data)) * 2:
        analysis.append("• Highly variable surface acceleration")
    else:
        analysis.append("• Relatively stable surface acceleration")
    
    # Add phase-specific analysis
    inhale_power = np.mean(fdotv_data[inhale_mask])
    exhale_power = np.mean(fdotv_data[exhale_mask])
    if inhale_power > 0 and exhale_power > 0:
        analysis.append("• Stable airway behavior in both phases")
    elif inhale_power < 0 and exhale_power < 0:
        analysis.append("• Potential collapse risk in both phases")
    elif inhale_power < 0:
        analysis.append("• Higher collapse risk during inhalation")
    else:
        analysis.append("• Higher collapse risk during exhalation")
    
    return "\n".join(analysis)

def create_plot(region, quadrant=None):
    """Create a plot for a specific region and quadrant"""
    # Create figure with single plot (no text section)
    fig = plt.figure(figsize=(11, 6))  # Adjusted for better use of page width
    
    # Create main plot
    ax1 = plt.gca()
    
    # Adjust margins to ensure all axes are visible
    plt.subplots_adjust(right=0.85, left=0.15)  # Keep left-right margins for axis labels
    
    # Construct column names based on region and quadrant
    if quadrant:
        tp_col = f'TP_{region}{quadrant} Monitor: TP_{region}{quadrant} Monitor (m^2)'
        dvdt_col = f'dvdt_{region}{quadrant} Monitor: dvdt_{region}{quadrant} Monitor (m^2)'
        fdotv_col = f'fdotv_{region}{quadrant}_Monitor_pts: fdotv_{region}{quadrant}_Monitor_pts (m^2)'
        title_suffix = f' - {quadrant}'
    else:
        tp_col = f'TP_{region} Monitor: TP_{region} Monitor (N)'
        dvdt_col = f'dvdt_{region} Monitor: dvdt_{region} Monitor (m^2)'
        fdotv_col = f'fdotv_{region}_Monitor_pts: fdotv_{region}_Monitor_pts (m^2)'
        title_suffix = ''

    # Plot TP on the left y-axis
    color1 = '#1f77b4'  # blue
    ln1 = ax1.plot(tp_df['Time'], tp_df[tp_col], color=color1, 
                   label=f'{region}{title_suffix} TP' + (' (N)' if not quadrant else ' (m²)'), linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Force (N)' if not quadrant else 'Area (m²)', color=color1, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Calculate the difference between TP and dvdt (normalized)
    tp_normalized = (tp_df[tp_col] - tp_df[tp_col].mean()) / tp_df[tp_col].std()
    dvdt_normalized = (dvdt_df[dvdt_col] - dvdt_df[dvdt_col].mean()) / dvdt_df[dvdt_col].std()
    diff = tp_normalized - dvdt_normalized
    
    # Plot the difference on a new left axis
    ax_diff = ax1.twinx()
    ax_diff.spines['right'].set_visible(False)  # Hide right spine
    ax_diff.spines['left'].set_position(('outward', 60))  # Position on left
    ax_diff.yaxis.set_label_position('left')
    ax_diff.yaxis.tick_left()
    color_diff = '#9467bd'  # purple
    ln_diff = ax_diff.plot(tp_df['Time'], diff, color=color_diff,
                          label='TP-dv/dt (normalized)', linewidth=2, linestyle='--')
    ax_diff.set_ylabel('TP-dv/dt\n(normalized)', color=color_diff, fontsize=12)
    ax_diff.tick_params(axis='y', labelcolor=color_diff)
    
    # Set aligned limits for difference plot
    diff_limits = get_aligned_limits(diff)
    ax_diff.set_ylim(diff_limits)

    # Plot dvdt (acceleration) - first right axis
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 0))  # No offset for first right axis
    color2 = '#ff7f0e'  # orange
    ln2 = ax2.plot(dvdt_df['Time'], dvdt_df[dvdt_col], color=color2,
                   label=f'{region}{title_suffix} dv/dt (m/s²)', linewidth=2)
    ax2.set_ylabel('Acceleration (m/s²)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Plot fdotv - second right axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 40))  # Reduced spacing
    color3 = '#2ca02c'  # green
    ln3 = ax3.plot(fdotv_df['Time'], fdotv_df[fdotv_col], color=color3,
                   label=f'{region}{title_suffix} fdotv (W)', linewidth=2)
    ax3.set_ylabel('Power (W)', color=color3, fontsize=12)
    ax3.tick_params(axis='y', labelcolor=color3)

    # Plot flow profile - third right axis
    ax_flow = ax1.twinx()
    ax_flow.spines['right'].set_position(('outward', 80))  # Reduced spacing
    flow_limits = get_aligned_limits(flow_df['OSAMRI003FlowProfile: Massflowrate'])
    ax_flow.set_ylim(flow_limits)
    
    ln_flow = ax_flow.plot(flow_df['OSAMRI003FlowProfile: time (s)'], 
                          flow_df['OSAMRI003FlowProfile: Massflowrate'],
                          color='gray', alpha=0.3, linestyle=':', linewidth=2,
                          label='Flow Rate')
    ax_flow.set_ylabel('Flow (kg/s)', color='gray', fontsize=12, labelpad=5)
    ax_flow.tick_params(axis='y', labelcolor='gray', pad=2)

    # Add phase indicators
    zero_crossings = np.where(np.diff(np.signbit(flow_df['OSAMRI003FlowProfile: Massflowrate'])))[0]
    for x in zero_crossings:
        ax1.axvline(x=flow_df['OSAMRI003FlowProfile: time (s)'].iloc[x], 
                    color='gray', linestyle=':', alpha=0.5)
    
    # Fill inhale/exhale regions
    ax_flow.fill_between(flow_df['OSAMRI003FlowProfile: time (s)'],
                        flow_df['OSAMRI003FlowProfile: Massflowrate'],
                        0, where=flow_df['OSAMRI003FlowProfile: Massflowrate'] > 0,
                        color='lightblue', alpha=0.1, label='Inhale')
    ax_flow.fill_between(flow_df['OSAMRI003FlowProfile: time (s)'],
                        flow_df['OSAMRI003FlowProfile: Massflowrate'],
                        0, where=flow_df['OSAMRI003FlowProfile: Massflowrate'] < 0,
                        color='lightpink', alpha=0.1, label='Exhale')

    # Set aligned limits
    tp_limits = get_aligned_limits(tp_df[tp_col])
    dvdt_limits = get_aligned_limits(dvdt_df[dvdt_col])
    fdotv_limits = get_aligned_limits(fdotv_df[fdotv_col])

    ax1.set_ylim(tp_limits)
    ax2.set_ylim(dvdt_limits)
    ax3.set_ylim(fdotv_limits)

    # Add horizontal line at y=0
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Format axis ticks with fewer decimal places
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    ax_flow.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))

    # Adjust tick parameters for better spacing
    for ax in [ax1, ax2, ax3, ax_flow]:
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='y', pad=0)  # Reduce padding between ticks and labels

    # Add title
    title = f'{region}{title_suffix} Measurements Over Time\n(Excluding First 10 Points, Zero-Aligned Y-Axes)'
    ax1.set_title(title, fontsize=12, pad=20)

    # Add legend with better positioning
    lns = ln1 + ln2 + ln3 + ln_flow + ln_diff
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 1))

    # Add grid
    ax1.grid(True, alpha=0.3)
    
    return fig

# Define regions and quadrants
regions = ['Nasopharynx', 'Oropharynx', 'Larynx', 'Trachea']
quadrants = ['Anterior', 'Posterior', 'Left', 'Right']

# Create PDF with all plots
with PdfPages('airway_measurements.pdf') as pdf:
    # First create main region plots
    for region in regions:
        fig = create_plot(region)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Then create quadrant plots for each region
        for quadrant in quadrants:
            fig = create_plot(region, quadrant)
            pdf.savefig(fig)
            plt.close(fig)

print("All plots have been saved to 'airway_measurements.pdf'") 