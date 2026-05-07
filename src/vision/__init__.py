"""
Vision module for the AI Personal Trainer.
Provides pose detection, angle calculation, and visualization.
"""

from .pose_detector import PoseDetector
from .angle_calculator import AngleCalculator
from .visualizer import Visualizer

__all__ = ["PoseDetector", "AngleCalculator", "Visualizer"]
