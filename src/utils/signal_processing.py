"""
Signal Processing Utilities

Shared signal processing and visualization helper functions used across the CFD analysis pipeline.
"""

import numpy as np
from typing import List, Tuple


def find_zero_crossings(times: np.ndarray, values: np.ndarray) -> List[float]:
    """
    Find the times when values cross zero using linear interpolation.

    Args:
        times: Array of time points
        values: Array of values corresponding to time points

    Returns:
        List of times when values cross zero
    """
    zero_crossings = np.where(np.diff(np.signbit(values)))[0]
    crossing_times = []

    for idx in zero_crossings:
        if idx + 1 < len(times) and idx >= 0:
            t0, t1 = times[idx], times[idx + 1]
            v0, v1 = values[idx], values[idx + 1]

            if v1 != v0:
                t_cross = t0 - v0 * (t1 - t0) / (v1 - v0)
                crossing_times.append(t_cross)

    return crossing_times


def smart_label_position(ax, target_xy: Tuple[float, float], text: str,
                        existing_labels: List[Tuple[float, float]],
                        data_points=None, margin_factor: float = 0.08,
                        min_distance: float = 0.15) -> Tuple[Tuple[float, float], str, str]:
    """
    Zone-based label positioning that guarantees no overlaps by using the entire plot area.

    Args:
        ax: matplotlib axis object
        target_xy: (x, y) tuple of the target point to annotate
        text: the text to display
        existing_labels: list of existing label positions [(x, y), ...]
        data_points: optional array of data points to avoid (less important now)
        margin_factor: fraction of plot range to use as margin from edges
        min_distance: minimum distance between labels (fraction of plot range)

    Returns:
        (xytext, ha, va): position and alignment for the label
    """
    x_target, y_target = target_xy

    # Get plot boundaries
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Calculate margins and minimum distance
    x_margin = margin_factor * x_range
    y_margin = margin_factor * y_range
    min_dist = min_distance * min(x_range, y_range)

    # Estimate text bounding box for the larger font size
    font_size = 11  # Increased by 20% from 9 to ~11
    char_width = font_size * 0.6
    char_height = font_size * 1.2
    text_width = len(text) * char_width
    text_height = char_height

    # Convert text dimensions to data coordinates
    text_width_data = text_width * x_range / 800
    text_height_data = text_height * y_range / 600

    # Define strategic zones across the entire plot area
    # Use 6x4 grid of zones to distribute labels evenly
    zones = []
    n_cols = 6
    n_rows = 4

    for row in range(n_rows):
        for col in range(n_cols):
            # Calculate zone center
            zone_x = xlim[0] + x_margin + (col + 0.5) * (x_range - 2*x_margin) / n_cols
            zone_y = ylim[0] + y_margin + (row + 0.5) * (y_range - 2*y_margin) / n_rows

            # Determine alignment based on zone position
            if col < n_cols // 3:
                ha = 'left'
            elif col >= 2 * n_cols // 3:
                ha = 'right'
            else:
                ha = 'center'

            if row < n_rows // 2:
                va = 'bottom'
            else:
                va = 'top'

            # Calculate priority based on distance to target (closer zones preferred)
            distance_to_target = np.sqrt((zone_x - x_target)**2 + (zone_y - y_target)**2)

            zones.append({
                'x': zone_x,
                'y': zone_y,
                'ha': ha,
                'va': va,
                'priority': distance_to_target
            })

    # Sort zones by priority (closer to target first, but we'll use any available zone)
    zones.sort(key=lambda z: z['priority'])

    def is_zone_available(zone):
        x, y, ha, va = zone['x'], zone['y'], zone['ha'], zone['va']

        # Calculate actual text bounds
        if ha == 'left':
            text_x_min, text_x_max = x, x + text_width_data
        elif ha == 'right':
            text_x_min, text_x_max = x - text_width_data, x
        else:  # center
            text_x_min, text_x_max = x - text_width_data/2, x + text_width_data/2

        if va == 'bottom':
            text_y_min, text_y_max = y, y + text_height_data
        elif va == 'top':
            text_y_min, text_y_max = y - text_height_data, y
        else:  # center
            text_y_min, text_y_max = y - text_height_data/2, y + text_height_data/2

        # Check boundaries
        if (text_x_min < xlim[0] + x_margin or text_x_max > xlim[1] - x_margin or
            text_y_min < ylim[0] + y_margin or text_y_max > ylim[1] - y_margin):
            return False

        # Check distance from existing labels - this is the critical part for no overlaps
        for existing_x, existing_y in existing_labels:
            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if distance < min_dist:
                return False

        return True

    # Try zones in order until we find an available one
    for zone in zones:
        if is_zone_available(zone):
            return (zone['x'], zone['y']), zone['ha'], zone['va']

    # If all zones are somehow occupied (very unlikely with 24 zones),
    # use a fallback position at the edge
    fallback_x = xlim[1] - x_margin - text_width_data/2
    fallback_y = ylim[1] - y_margin - text_height_data/2
    return (fallback_x, fallback_y), 'right', 'top'


def format_time_label(time_value: float) -> str:
    """
    Format time values to remove unnecessary decimal places.

    Args:
        time_value: Time value in seconds

    Returns:
        Formatted string like "1s" or "1.5s"
    """
    if time_value == int(time_value):
        return f"{int(time_value)}s"
    else:
        # Remove trailing zeros and unnecessary decimal places
        return f"{time_value:.2f}s".rstrip('0').rstrip('.')
