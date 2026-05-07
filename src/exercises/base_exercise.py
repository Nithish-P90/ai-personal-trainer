"""
BaseExercise: Abstract base class for all exercise detectors.
Each exercise defines its own joint chains, angle thresholds, and rep logic.
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from src.vision.angle_calculator import AngleCalculator


class BaseExercise(ABC):
    """
    Abstract base class for exercise detection.

    Every exercise must define:
    - Which joints to track
    - What angles constitute "good form"
    - How to count a rep (state machine)
    """

    def __init__(self, name: str, smoothing_window: int = 5):
        self.name = name
        self.rep_count = 0
        self.phase = "idle"  # States: idle, descending, ascending
        self.form_feedback = []  # List of current form issues
        self.angle_calc = AngleCalculator()
        self.smoothing_window = smoothing_window

        # Angle history buffers for temporal smoothing
        self._angle_buffers: dict[str, deque] = {}

    def _get_smoothed_angle(self, key: str, raw_angle: float) -> float:
        """Track and smooth an angle value over time."""
        if key not in self._angle_buffers:
            self._angle_buffers[key] = deque(maxlen=self.smoothing_window)
        self._angle_buffers[key].append(raw_angle)
        return float(np.mean(self._angle_buffers[key]))

    @abstractmethod
    def get_tracked_joints(self) -> list[str]:
        """Return list of landmark names this exercise cares about."""
        pass

    @abstractmethod
    def analyze_frame(self, landmarks: np.ndarray) -> dict:
        """
        Analyze a single frame's landmarks for this exercise.

        Args:
            landmarks: (33, 4) NumPy array of [x, y, z, visibility] per joint.

        Returns:
            dict with keys:
                - 'angles': dict of angle name → value
                - 'phase': current rep phase
                - 'rep_completed': True if a rep just finished
                - 'form_ok': True if form is good
                - 'feedback': list of feedback strings
                - 'highlight_joints': list of joint indices to highlight
        """
        pass

    def reset(self):
        """Reset rep count and state for a new set."""
        self.rep_count = 0
        self.phase = "idle"
        self.form_feedback = []
        self._angle_buffers.clear()
