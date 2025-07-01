"""
Analysis plotting module for CFD data visualization.
Creates professional plots for pressure, velocity, and acceleration analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from typing import Dict, Tuple

def setup_plotting_style() -> None:
    """Configure matplotlib plotting style for consistent, professional appearance."""
    plt.rcParams['figure.figsize'] = [20, 16]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['font.size'] = 10

def add_time_arrows(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray, times: np.ndarray) -> None:
    """
    Add arrows to show time progression in the plot.
    
    Args:
        ax: Matplotlib axes object
        x_data: X-axis data points
        y_data: Y-axis data points
        times: Time points for color mapping
    """
    ax.plot(x_data, y_data, color='blue', alpha=0.3, linestyle='-', linewidth=1, zorder=1, label='Path')
    for i in range(len(x_data)-1):
        x, y = x_data[i], y_data[i]
        dx, dy = x_data[i+1] - x, y_data[i+1] - y
        ax.annotate('', 
                   xy=(x + dx, y + dy), 
                   xytext=(x, y),
                   arrowprops=dict(arrowstyle='->',
                                 color='blue',
                                 alpha=0.5,
                                 connectionstyle='arc3'),
                   zorder=1.5)

def create_pressure_velocity_plot(ax: plt.Axes, velocity: np.ndarray, total_pressures: np.ndarray, 
                                times: np.ndarray, norm: plt.Normalize) -> plt.scatter:
    """
    Create Total Pressure vs Velocity magnitude plot.
    
    Args:
        ax: Matplotlib axes object
        velocity: Array of velocity magnitudes
        total_pressures: Array of total pressure values
        times: Array of time points
        norm: Normalization for color mapping
        
    Returns:
        Scatter plot object
    """
    add_time_arrows(ax, velocity, total_pressures, times)
    scatter = ax.scatter(velocity, total_pressures, 
                        c=times, cmap='viridis', norm=norm, marker='o', s=50,
                        label='Data Points', zorder=2)
    ax.set_xlabel('Velocity Magnitude (m/s)')
    ax.set_ylabel('Total Pressure (Pa)')
    ax.set_title('Total Pressure vs Velocity')
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    return scatter

def create_pressure_vdotn_plot(ax: plt.Axes, vdotn: np.ndarray, total_pressures: np.ndarray, 
                              times: np.ndarray, norm: plt.Normalize) -> plt.scatter:
    """
    Create Total Pressure vs VdotN plot.
    
    Args:
        ax: Matplotlib axes object
        vdotn: Array of VdotN values
        total_pressures: Array of total pressure values
        times: Array of time points
        norm: Normalization for color mapping
        
    Returns:
        Scatter plot object
    """
    add_time_arrows(ax, vdotn, total_pressures, times)
    scatter = ax.scatter(vdotn, total_pressures, 
                        c=times, cmap='viridis', norm=norm, marker='o', s=50,
                        label='Data Points', zorder=2)
    ax.set_xlabel('VdotN')
    ax.set_ylabel('Total Pressure (Pa)')
    ax.set_title('Total Pressure vs VdotN')
    ax.grid(True)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='VdotN = 0')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    return scatter

def create_pressure_acceleration_plot(ax: plt.Axes, acceleration: np.ndarray, total_pressures: np.ndarray, 
                                    times: np.ndarray, norm: plt.Normalize) -> plt.scatter:
    """
    Create Total Pressure vs Acceleration plot.
    
    Args:
        ax: Matplotlib axes object
        acceleration: Array of acceleration values
        total_pressures: Array of total pressure values
        times: Array of time points
        norm: Normalization for color mapping
        
    Returns:
        Scatter plot object
    """
    add_time_arrows(ax, acceleration, total_pressures, times)
    scatter = ax.scatter(acceleration, total_pressures, 
                        c=times, cmap='viridis', norm=norm, marker='o', s=50,
                        zorder=2)
    ax.set_xlabel('Acceleration (m/s²)')
    ax.set_ylabel('Total Pressure (Pa)')
    ax.set_title('Total Pressure vs Acceleration')
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    return scatter

def create_dpdt_acceleration_plot(ax: plt.Axes, acceleration: np.ndarray, dp_dt: np.ndarray, 
                                times: np.ndarray, norm: plt.Normalize) -> plt.scatter:
    """
    Create dP/dt vs Acceleration plot.
    
    Args:
        ax: Matplotlib axes object
        acceleration: Array of acceleration values
        dp_dt: Array of pressure rate of change values
        times: Array of time points
        norm: Normalization for color mapping
        
    Returns:
        Scatter plot object
    """
    add_time_arrows(ax, acceleration, dp_dt, times)
    scatter = ax.scatter(acceleration, dp_dt, 
                        c=times, cmap='viridis', norm=norm, marker='o', s=50,
                        zorder=2)
    ax.set_xlabel('Acceleration (m/s²)')
    ax.set_ylabel('dP/dt (Pa/s)')
    ax.set_title('dP/dt vs Acceleration')
    ax.grid(True)
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    return scatter

def plot_flow_profile(time: np.ndarray, flow: np.ndarray, 
                     first_crossing_time: float, last_crossing_time: float):
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
    plt.title('Flow Profile with Clean Breathing Cycle Highlighted', fontsize=17)
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
    plt.savefig('flow_profile_bounds.pdf', bbox_inches='tight', dpi=300)
    plt.close() 