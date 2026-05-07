"""
BicepCurlDetector: Detects bicep curl reps and analyzes form.

Key joints: Shoulder, Elbow, Wrist (both sides)
Good form signals:
  - Full range of motion (elbow angle goes from ~160° → ~30° → ~160°)
  - Elbow stays pinned near the torso (minimal lateral drift)
  - No shoulder swing (shoulder remains stable)
"""

import numpy as np

from .base_exercise import BaseExercise
from src.vision.pose_detector import PoseDetector


class BicepCurlDetector(BaseExercise):

    # Thresholds
    CURL_TOP_ANGLE = 50      # Elbow angle at top of curl (fully contracted)
    CURL_BOTTOM_ANGLE = 140  # Elbow angle at bottom (fully extended)
    ELBOW_DRIFT_THRESHOLD = 40  # Max px the elbow can drift from hip

    def __init__(self, side: str = "left"):
        """
        Args:
            side: 'left' or 'right' — which arm to track.
        """
        super().__init__(name=f"Bicep Curl ({side.capitalize()})")
        self.side = side

    def get_tracked_joints(self) -> list[str]:
        return [
            f"{self.side}_shoulder",
            f"{self.side}_elbow",
            f"{self.side}_wrist",
            f"{self.side}_hip",
        ]

    def analyze_frame(self, landmarks: np.ndarray) -> dict:
        L = PoseDetector.LANDMARKS

        shoulder = landmarks[L[f"{self.side}_shoulder"]]
        elbow = landmarks[L[f"{self.side}_elbow"]]
        wrist = landmarks[L[f"{self.side}_wrist"]]
        hip = landmarks[L[f"{self.side}_hip"]]

        # Elbow angle
        elbow_angle = self._get_smoothed_angle(
            "elbow",
            self.angle_calc.calculate_angle(shoulder, elbow, wrist),
        )

        # --- Form checks ---
        feedback = []
        highlight_joints = []
        form_ok = True

        # Elbow drift from torso
        elbow_drift = abs(elbow[0] - hip[0])
        if elbow_drift > self.ELBOW_DRIFT_THRESHOLD:
            feedback.append("Pin your elbow closer to your body")
            highlight_joints.append(L[f"{self.side}_elbow"])
            form_ok = False

        # Shoulder stability — shoulder shouldn't move much vertically during curl
        # (We track this across frames via smoothing)
        shoulder_y_smooth = self._get_smoothed_angle("shoulder_y", shoulder[1])
        shoulder_drift = abs(shoulder[1] - shoulder_y_smooth)
        if shoulder_drift > 20:
            feedback.append("Don't swing — keep your shoulder stable")
            highlight_joints.append(L[f"{self.side}_shoulder"])
            form_ok = False

        if form_ok and self.phase != "idle":
            feedback.append("Perfect curl form!")

        # --- Rep counting state machine ---
        rep_completed = False

        if self.phase == "idle" or self.phase == "ascending":
            if elbow_angle < self.CURL_TOP_ANGLE:
                self.phase = "descending"  # At the top of the curl
        elif self.phase == "descending":
            if elbow_angle > self.CURL_BOTTOM_ANGLE:
                self.phase = "ascending"  # Back to bottom
                self.rep_count += 1
                rep_completed = True

        return {
            "angles": {
                "elbow": elbow_angle,
                "elbow_drift_px": elbow_drift,
            },
            "phase": self.phase,
            "rep_completed": rep_completed,
            "form_ok": form_ok,
            "feedback": feedback,
            "highlight_joints": highlight_joints,
        }
