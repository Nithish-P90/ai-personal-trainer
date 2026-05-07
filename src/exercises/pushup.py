"""
PushupDetector: Detects push-up reps and analyzes form.

Key joints: Shoulder, Hip, Ankle (body line), Shoulder, Elbow, Wrist (arm angle)
Good form signals:
  - Body forms a straight line (shoulder-hip-ankle angle ~180°)
  - Arms reach ~90° at bottom, ~170° at top
  - No hip sag or pike
"""

import numpy as np

from .base_exercise import BaseExercise
from src.vision.pose_detector import PoseDetector


class PushupDetector(BaseExercise):

    # Thresholds
    ARM_BOTTOM_ANGLE = 90    # Elbow angle at the bottom of pushup
    ARM_TOP_ANGLE = 155      # Elbow angle at the top (locked out)
    BODY_LINE_MIN = 155      # Shoulder-hip-ankle angle: below this = sag or pike
    BODY_LINE_MAX = 190      # Above this means hips too high (pike)

    def __init__(self):
        super().__init__(name="Push-Up")

    def get_tracked_joints(self) -> list[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_ankle", "right_ankle",
        ]

    def analyze_frame(self, landmarks: np.ndarray) -> dict:
        L = PoseDetector.LANDMARKS

        l_shoulder = landmarks[L["left_shoulder"]]
        r_shoulder = landmarks[L["right_shoulder"]]
        l_elbow = landmarks[L["left_elbow"]]
        r_elbow = landmarks[L["right_elbow"]]
        l_wrist = landmarks[L["left_wrist"]]
        r_wrist = landmarks[L["right_wrist"]]
        l_hip = landmarks[L["left_hip"]]
        r_hip = landmarks[L["right_hip"]]
        l_ankle = landmarks[L["left_ankle"]]
        r_ankle = landmarks[L["right_ankle"]]

        # Arm angles (both sides)
        left_arm_angle = self._get_smoothed_angle(
            "left_arm",
            self.angle_calc.calculate_angle(l_shoulder, l_elbow, l_wrist),
        )
        right_arm_angle = self._get_smoothed_angle(
            "right_arm",
            self.angle_calc.calculate_angle(r_shoulder, r_elbow, r_wrist),
        )
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2

        # Body line angle (shoulder → hip → ankle)
        mid_shoulder = (l_shoulder[:2] + r_shoulder[:2]) / 2
        mid_hip = (l_hip[:2] + r_hip[:2]) / 2
        mid_ankle = (l_ankle[:2] + r_ankle[:2]) / 2
        body_line_angle = self._get_smoothed_angle(
            "body_line",
            self.angle_calc.calculate_angle(mid_shoulder, mid_hip, mid_ankle),
        )

        # --- Form checks ---
        feedback = []
        highlight_joints = []
        form_ok = True

        if body_line_angle < self.BODY_LINE_MIN:
            feedback.append("Hips sagging — tighten your core!")
            highlight_joints.extend([L["left_hip"], L["right_hip"]])
            form_ok = False
        elif body_line_angle > self.BODY_LINE_MAX:
            feedback.append("Hips too high — flatten your body line")
            highlight_joints.extend([L["left_hip"], L["right_hip"]])
            form_ok = False

        if form_ok and self.phase != "idle":
            feedback.append("Solid push-up form!")

        # --- Rep counting state machine ---
        rep_completed = False

        if self.phase == "idle" or self.phase == "ascending":
            if avg_arm_angle < self.ARM_BOTTOM_ANGLE:
                self.phase = "descending"  # At the bottom
        elif self.phase == "descending":
            if avg_arm_angle > self.ARM_TOP_ANGLE:
                self.phase = "ascending"  # Locked out at top
                self.rep_count += 1
                rep_completed = True

        return {
            "angles": {
                "left_arm": left_arm_angle,
                "right_arm": right_arm_angle,
                "avg_arm": avg_arm_angle,
                "body_line": body_line_angle,
            },
            "phase": self.phase,
            "rep_completed": rep_completed,
            "form_ok": form_ok,
            "feedback": feedback,
            "highlight_joints": highlight_joints,
        }
