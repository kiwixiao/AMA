import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

# Read the CSV files
tp_df = pd.read_csv('AirwaySurfaceTotalPressureForce.csv')
dvdt_df = pd.read_csv('dvdt.csv')
flow_df = pd.read_csv('FlowProfile.csv')

# Skip first 10 points for all dataframes
start_index = 10
tp_df = tp_df.iloc[start_index:]
dvdt_df = dvdt_df.iloc[start_index:]

def find_respiratory_cycle_boundaries(flow_data, time_data):
    """Find the start and end points of respiratory cycles based on flow data"""
    # Find all zero crossings
    zero_crossings = np.where(np.diff(np.signbit(flow_data)))[0]
    
    # Find first and last zero crossing
    if len(zero_crossings) >= 2:
        # Take first and last zero crossing
        start_idx = zero_crossings[0]
        end_idx = zero_crossings[-1]
        return [(start_idx, end_idx)]
    
    return []

def moving_average(data, window=50, passes=2):
    """Apply moving average with proper edge handling and multiple passes"""
    smoothed = data.copy()
    for _ in range(passes):
        weights = np.ones(window) / window
        # Handle edges by reflecting the data
        pad_data = np.pad(smoothed, (window//2, window//2), mode='reflect')
        # Ensure output length matches input length
        smoothed = np.convolve(pad_data, weights, mode='valid')[:len(data)]
    return smoothed

def analyze_temporal_relationship(region, quadrant=None):
    """Analyze temporal relationships between TP and acceleration for a region"""
    # Set up the columns
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
    tp_data = tp_df[tp_col].values
    dvdt_data = dvdt_df[dvdt_col].values
    time_data = tp_df['Time'].values
    
    # Smooth TP data with more aggressive smoothing
    tp_smooth = moving_average(tp_data, window=50, passes=2)
    
    # Normalize signals to make them comparable
    tp_norm = (tp_data - np.mean(tp_data)) / np.std(tp_data)
    tp_smooth_norm = (tp_smooth - np.mean(tp_smooth)) / np.std(tp_smooth)
    dvdt_norm = (dvdt_data - np.mean(dvdt_data)) / np.std(dvdt_data)
    
    # Find peaks and troughs for smoothed TP signal with larger minimum distance
    tp_peaks, _ = find_peaks(tp_smooth_norm, distance=50)  # Increased minimum distance
    tp_troughs, _ = find_peaks(-tp_smooth_norm, distance=50)  # Increased minimum distance
    dvdt_peaks, _ = find_peaks(dvdt_norm, distance=20)
    dvdt_troughs, _ = find_peaks(-dvdt_norm, distance=20)
    
    # Get respiratory cycles from flow data
    flow_time = flow_df['OSAMRI003FlowProfile: time (s)'].values
    flow_data = flow_df['OSAMRI003FlowProfile: Massflowrate'].values
    
    # Interpolate flow data to match our time points
    flow_interp = interp1d(flow_time, flow_data, bounds_error=False, fill_value="extrapolate")
    flow_resampled = flow_interp(time_data)
    
    # Find respiratory cycle
    cycles = find_respiratory_cycle_boundaries(flow_resampled, time_data)
    
    # Calculate cycle period
    if cycles:
        cycle_period = time_data[cycles[0][1]] - time_data[cycles[0][0]]
        print(f"Respiratory cycle period: {cycle_period:.2f}s")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot signals and peaks/troughs
    ax1.plot(time_data, tp_norm, label='Original Pressure Force', color='lightblue', alpha=0.3)
    ax1.plot(time_data, tp_smooth_norm, label='Smoothed Pressure Force', color='blue', alpha=0.7)
    ax1.plot(time_data, dvdt_norm, label='Acceleration', color='red', alpha=0.7)
    ax1.plot(time_data, flow_resampled/np.max(np.abs(flow_resampled)), 
             label='Flow', color='gray', alpha=0.3)
    
    # Plot peaks and troughs for smoothed TP signal
    ax1.scatter(time_data[tp_peaks], tp_smooth_norm[tp_peaks], color='blue', marker='^', 
                label='TP peaks')
    ax1.scatter(time_data[tp_troughs], tp_smooth_norm[tp_troughs], color='blue', marker='v', 
                label='TP troughs')
    ax1.scatter(time_data[dvdt_peaks], dvdt_norm[dvdt_peaks], color='red', marker='^', 
                label='Accel peaks')
    ax1.scatter(time_data[dvdt_troughs], dvdt_norm[dvdt_troughs], color='red', marker='v', 
                label='Accel troughs')
    
    # Add cycle boundaries
    for cycle in cycles:
        ax1.axvline(time_data[cycle[0]], color='gray', linestyle=':', alpha=0.3)
        ax1.axvline(time_data[cycle[1]], color='gray', linestyle=':', alpha=0.3)
    
    # Calculate phase differences using smoothed TP signal
    phase_diffs = []
    for cycle in cycles:
        cycle_start, cycle_end = cycle
        cycle_duration = time_data[cycle_end] - time_data[cycle_start]
        
        # Find peaks within this cycle
        tp_peaks_in_cycle = tp_peaks[(time_data[tp_peaks] >= time_data[cycle_start]) & 
                                   (time_data[tp_peaks] <= time_data[cycle_end])]
        dvdt_peaks_in_cycle = dvdt_peaks[(time_data[dvdt_peaks] >= time_data[cycle_start]) & 
                                       (time_data[dvdt_peaks] <= time_data[cycle_end])]
        
        if len(tp_peaks_in_cycle) > 0 and len(dvdt_peaks_in_cycle) > 0:
            # Calculate phase difference for each peak pair
            for tp_peak in tp_peaks_in_cycle:
                # Find closest acceleration peak
                closest_dvdt_peak = dvdt_peaks_in_cycle[np.argmin(np.abs(time_data[dvdt_peaks_in_cycle] - 
                                                                        time_data[tp_peak]))]
                time_diff = time_data[tp_peak] - time_data[closest_dvdt_peak]
                phase_diff = (time_diff / cycle_duration) * 360  # Convert to degrees
                phase_diffs.append(phase_diff)
    
    # Plot phase differences histogram
    if phase_diffs:
        ax2.hist(phase_diffs, bins=20, alpha=0.7)
        ax2.set_xlabel('Phase Difference (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Phase Differences')
        
        # Add mean phase difference
        mean_phase = np.mean(phase_diffs)
        ax2.axvline(mean_phase, color='r', linestyle='--', 
                   label=f'Mean Phase: {mean_phase:.1f}°')
        ax2.legend()
    
    # Add labels and title
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normalized Amplitude')
    ax1.set_title(f'{region}{title_suffix}\nTemporal Analysis of Pressure Force and Acceleration')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    
    # Add summary statistics
    if phase_diffs:
        stats_text = (
            f'Summary Statistics:\n'
            f'Mean Phase Difference: {np.mean(phase_diffs):.1f}°\n'
            f'Std Dev: {np.std(phase_diffs):.1f}°\n'
            f'Median: {np.median(phase_diffs):.1f}°\n'
            f'Respiratory Cycle Period: {cycle_period:.3f}s'
        )
        ax1.text(1.02, 0.02, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    return fig

# Define regions and quadrants
regions = ['Nasopharynx', 'Oropharynx', 'Larynx', 'Trachea']
quadrants = ['Anterior', 'Posterior', 'Left', 'Right']

# Create PDF with temporal analysis plots for all regions
with PdfPages('temporal_analysis.pdf') as pdf:
    # First create main region plots
    for region in regions:
        fig = analyze_temporal_relationship(region)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Then create quadrant plots
        for quadrant in quadrants:
            fig = analyze_temporal_relationship(region, quadrant)
            pdf.savefig(fig)
            plt.close(fig)

print("Temporal analysis plots have been saved to 'temporal_analysis.pdf'") 