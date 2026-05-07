"""
PoseDetector: A wrapper around MediaPipe Pose for real-time human pose estimation.
Provides 33 3D landmarks per frame from a webcam or video source.
"""

import mediapipe as mp
import numpy as np


class PoseDetector:
    """Wraps MediaPipe Pose to extract 33 body landmarks from an image frame."""

    # Landmark index map for quick reference
    # Full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    LANDMARKS = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the pose detector.

        Args:
            static_image_mode: If True, treats each frame independently (slower but more accurate).
                               If False, uses temporal tracking (faster, ideal for live video).
            model_complexity: 0 = Lite, 1 = Full, 2 = Heavy. Higher = more accurate, slower.
            smooth_landmarks: Reduces jitter across frames. Keep True for live video.
            min_detection_confidence: Minimum confidence for initial pose detection.
            min_tracking_confidence: Minimum confidence for frame-to-frame tracking.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.results = None

    def detect(self, frame_rgb: np.ndarray) -> bool:
        """
        Process a single RGB frame and extract pose landmarks.

        Args:
            frame_rgb: An RGB image (H, W, 3) as a NumPy array.

        Returns:
            True if a pose was detected, False otherwise.
        """
        self.results = self.pose.process(frame_rgb)
        return self.results.pose_landmarks is not None

    def get_landmarks(self) -> list | None:
        """
        Get the raw MediaPipe landmark list from the last processed frame.

        Returns:
            A list of 33 NormalizedLandmark objects, or None if no pose detected.
        """
        if self.results and self.results.pose_landmarks:
            return self.results.pose_landmarks.landmark
        return None

    def get_landmark_array(self, frame_width: int, frame_height: int) -> np.ndarray | None:
        """
        Get landmarks as a NumPy array of pixel coordinates + depth.

        Args:
            frame_width: Width of the source frame in pixels.
            frame_height: Height of the source frame in pixels.

        Returns:
            A (33, 4) NumPy array where each row is [x_px, y_px, z_depth, visibility],
            or None if no pose was detected.

            - x_px, y_px: Pixel coordinates in the frame.
            - z_depth: Relative depth (smaller = closer to camera).
            - visibility: Confidence that this landmark is visible (0.0 to 1.0).
        """
        landmarks = self.get_landmarks()
        if landmarks is None:
            return None

        coords = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            coords[i] = [
                lm.x * frame_width,
                lm.y * frame_height,
                lm.z,  # Relative depth — useful for 3D form analysis
                lm.visibility,
            ]
        return coords

    def get_point(self, name: str, frame_width: int, frame_height: int) -> tuple | None:
        """
        Get a single landmark's pixel coordinates by name.

        Args:
            name: Landmark name (e.g., 'left_shoulder', 'right_knee').
            frame_width: Width of the source frame.
            frame_height: Height of the source frame.

        Returns:
            (x, y, z, visibility) tuple, or None if not detected.
        """
        landmarks = self.get_landmarks()
        if landmarks is None:
            return None

        idx = self.LANDMARKS.get(name)
        if idx is None:
            raise ValueError(f"Unknown landmark: '{name}'. Valid: {list(self.LANDMARKS.keys())}")

        lm = landmarks[idx]
        return (lm.x * frame_width, lm.y * frame_height, lm.z, lm.visibility)

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
