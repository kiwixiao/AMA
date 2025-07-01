"""
Surface Painting Module

This module provides interactive STL surface painting capabilities for CFD analysis.
It allows users to paint arbitrary regions on STL surfaces and map them to CFD data
for regional analysis.

Components:
- stl_painter.py: Interactive 3D painting interface
- coordinate_mapper.py: Maps painted coordinates to CSV data
- regional_analyzer.py: Computes statistics for painted regions

This is an add-on module that does not affect existing pipeline functionality.
"""

from .stl_painter import STLPainter
from .coordinate_mapper import CoordinateMapper
from .regional_analyzer import RegionalAnalyzer

__all__ = ['STLPainter', 'CoordinateMapper', 'RegionalAnalyzer'] 