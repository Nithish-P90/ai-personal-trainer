"""
AngleCalculator: Computes joint angles from 2D/3D landmark coordinates.
This is the mathematical core of the form analysis system.
"""

import numpy as np


class AngleCalculator:
    """Calculates angles between three connected body joints."""

    @staticmethod
    def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """
        Calculate the angle at point_b formed by the vectors BA and BC.

        Example: For a bicep curl, pass (shoulder, elbow, wrist) to get the elbow angle.

            A (shoulder)
             \\
              \\ angle here
               B (elbow) ---------> C (wrist)

        Args:
            point_a: (x, y) or (x, y, z) coordinates of joint A.
            point_b: (x, y) or (x, y, z) coordinates of the vertex joint B.
            point_c: (x, y) or (x, y, z) coordinates of joint C.

        Returns:
            Angle in degrees (0° to 180°).
        """
        a = np.array(point_a[:2], dtype=np.float64)
        b = np.array(point_b[:2], dtype=np.float64)
        c = np.array(point_c[:2], dtype=np.float64)

        # Vectors from vertex to each endpoint
        ba = a - b
        bc = c - b

        # Dot product and magnitudes
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)

        # Clamp to avoid NaN from floating point errors
        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))
        return float(angle)

    @staticmethod
    def calculate_angle_3d(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
        """
        Calculate the 3D angle at point_b using all three axes (x, y, z).
        This gives more accurate form analysis by accounting for depth.

        Args:
            point_a: (x, y, z) coordinates of joint A.
            point_b: (x, y, z) coordinates of the vertex joint B.
            point_c: (x, y, z) coordinates of joint C.

        Returns:
            Angle in degrees (0° to 180°).
        """
        a = np.array(point_a[:3], dtype=np.float64)
        b = np.array(point_b[:3], dtype=np.float64)
        c = np.array(point_c[:3], dtype=np.float64)

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine))
        return float(angle)

    @staticmethod
    def calculate_horizontal_angle(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Calculate the angle of the line AB relative to the horizontal axis.
        Useful for detecting torso lean (e.g., in overhead press or deadlift).

        Args:
            point_a: (x, y) of first joint.
            point_b: (x, y) of second joint.

        Returns:
            Angle in degrees (0° = perfectly horizontal, 90° = vertical).
        """
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        return float(angle)

    @staticmethod
    def calculate_lateral_offset(point: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate horizontal drift of a point relative to a reference.
        Useful for detecting if elbows are flaring out during curls.

        Args:
            point: (x, y) of the joint to check.
            reference: (x, y) of the reference joint.

        Returns:
            Horizontal pixel offset (positive = right, negative = left).
        """
        return float(point[0] - reference[0])

    @staticmethod
    def smooth_angle(angles_buffer: list, window_size: int = 5) -> float:
        """
        Apply temporal smoothing to reduce jitter in angle readings.
        Uses a simple moving average over the last N frames.

        Args:
            angles_buffer: List of recent angle values.
            window_size: Number of frames to average over.

        Returns:
            Smoothed angle value.
        """
        if not angles_buffer:
            return 0.0
        recent = angles_buffer[-window_size:]
        return float(np.mean(recent))
