"""
SquatDetector: Detects squat reps and analyzes form.

Key joints: Hip, Knee, Ankle (both sides)
Good form signals:
  - Knee angle reaches 70°-110° at bottom
  - Knees don't drift past toes (x-offset check)
  - Back stays relatively upright (shoulder-hip angle)
"""

import numpy as np

from .base_exercise import BaseExercise
from src.vision.pose_detector import PoseDetector


class SquatDetector(BaseExercise):

    # Thresholds
    BOTTOM_ANGLE_MIN = 60   # Knee angle at deepest point
    BOTTOM_ANGLE_MAX = 110  # Must go below this to count as "at bottom"
    STANDING_ANGLE = 160    # Above this = standing (rep complete)
    KNEE_OVER_TOE_THRESHOLD = 30  # Max px the knee can be ahead of the ankle

    def __init__(self):
        super().__init__(name="Squat")

    def get_tracked_joints(self) -> list[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]

    def analyze_frame(self, landmarks: np.ndarray) -> dict:
        L = PoseDetector.LANDMARKS

        # Extract key joints
        l_hip = landmarks[L["left_hip"]]
        l_knee = landmarks[L["left_knee"]]
        l_ankle = landmarks[L["left_ankle"]]
        r_hip = landmarks[L["right_hip"]]
        r_knee = landmarks[L["right_knee"]]
        r_ankle = landmarks[L["right_ankle"]]
        l_shoulder = landmarks[L["left_shoulder"]]
        r_shoulder = landmarks[L["right_shoulder"]]

        # Calculate knee angles (both sides)
        left_knee_angle = self._get_smoothed_angle(
            "left_knee",
            self.angle_calc.calculate_angle(l_hip, l_knee, l_ankle),
        )
        right_knee_angle = self._get_smoothed_angle(
            "right_knee",
            self.angle_calc.calculate_angle(r_hip, r_knee, r_ankle),
        )

        # Use the average of both knees
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # Back angle: shoulder → hip vertical alignment
        mid_shoulder = (l_shoulder[:2] + r_shoulder[:2]) / 2
        mid_hip = (l_hip[:2] + r_hip[:2]) / 2
        back_angle = self.angle_calc.calculate_horizontal_angle(mid_hip, mid_shoulder)

        # --- Form checks ---
        feedback = []
        highlight_joints = []
        form_ok = True

        # Check knee-over-toe
        left_knee_over = l_knee[0] - l_ankle[0]
        right_knee_over = r_knee[0] - r_ankle[0]
        if abs(left_knee_over) > self.KNEE_OVER_TOE_THRESHOLD:
            feedback.append("Left knee drifting over toes")
            highlight_joints.append(L["left_knee"])
            form_ok = False
        if abs(right_knee_over) > self.KNEE_OVER_TOE_THRESHOLD:
            feedback.append("Right knee drifting over toes")
            highlight_joints.append(L["right_knee"])
            form_ok = False

        # Check back angle (too much forward lean)
        if back_angle < 50:
            feedback.append("Keep your chest up — too much forward lean")
            form_ok = False

        if form_ok and self.phase != "idle":
            feedback.append("Great form! Keep it up")

        # --- Rep counting state machine ---
        rep_completed = False

        if self.phase == "idle" or self.phase == "ascending":
            if avg_knee_angle < self.BOTTOM_ANGLE_MAX:
                self.phase = "descending"
        elif self.phase == "descending":
            if avg_knee_angle > self.STANDING_ANGLE:
                self.phase = "ascending"
                self.rep_count += 1
                rep_completed = True

        return {
            "angles": {
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
                "avg_knee": avg_knee_angle,
                "back": back_angle,
            },
            "phase": self.phase,
            "rep_completed": rep_completed,
            "form_ok": form_ok,
            "feedback": feedback,
            "highlight_joints": highlight_joints,
        }
