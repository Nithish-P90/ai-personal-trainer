"""
Visualizer: Draws skeleton overlays, angle labels, and form feedback onto video frames.
This is what makes the live feed look like a real personal trainer app.
"""

import cv2
import numpy as np
import mediapipe as mp


# Color palette — premium dark-mode aesthetic
COLORS = {
    "skeleton": (0, 255, 200),       # Cyan-green for bones
    "joint": (255, 100, 100),        # Coral for joint dots
    "joint_highlight": (0, 200, 255), # Amber for highlighted joints
    "angle_text": (255, 255, 255),   # White for angle labels
    "good_form": (0, 255, 100),      # Green
    "bad_form": (0, 80, 255),        # Red-orange (BGR)
    "rep_counter": (255, 220, 0),    # Gold
    "background_box": (30, 30, 30),  # Dark grey for text backgrounds
    "title": (200, 200, 200),        # Light grey for titles
}

# Drawing specs for MediaPipe's built-in renderer
LANDMARK_STYLE = mp.solutions.drawing_utils.DrawingSpec(
    color=COLORS["joint"], thickness=2, circle_radius=3
)
CONNECTION_STYLE = mp.solutions.drawing_utils.DrawingSpec(
    color=COLORS["skeleton"], thickness=2, circle_radius=1
)


class Visualizer:
    """Draws pose information, angles, and coaching feedback onto frames."""

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw the full MediaPipe skeleton onto the frame.

        Args:
            frame: BGR image to draw on.
            results: MediaPipe Pose results object.

        Returns:
            Frame with skeleton overlay.
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=LANDMARK_STYLE,
                connection_drawing_spec=CONNECTION_STYLE,
            )
        return frame

    @staticmethod
    def draw_angle(
        frame: np.ndarray,
        point: tuple,
        angle: float,
        color: tuple = COLORS["angle_text"],
    ) -> np.ndarray:
        """
        Draw an angle label near a joint.

        Args:
            frame: BGR image.
            point: (x, y) pixel position of the joint.
            angle: Angle value in degrees.
            color: Text color (BGR).

        Returns:
            Frame with angle label.
        """
        x, y = int(point[0]), int(point[1])

        # Background pill for readability
        text = f"{int(angle)}°"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x - 5, y - th - 8),
            (x + tw + 5, y + 5),
            COLORS["background_box"],
            -1,
        )
        cv2.rectangle(
            frame,
            (x - 5, y - th - 8),
            (x + tw + 5, y + 5),
            color,
            1,
        )
        cv2.putText(
            frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        return frame

    @staticmethod
    def draw_rep_counter(
        frame: np.ndarray, reps: int, exercise_name: str = "Exercise"
    ) -> np.ndarray:
        """
        Draw a rep counter HUD in the top-left corner.

        Args:
            frame: BGR image.
            reps: Current rep count.
            exercise_name: Name of the active exercise.

        Returns:
            Frame with rep counter overlay.
        """
        h, w = frame.shape[:2]

        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 120), COLORS["background_box"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (10, 10), (280, 120), COLORS["rep_counter"], 2)

        # Exercise name
        cv2.putText(
            frame,
            exercise_name.upper(),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["title"],
            1,
        )

        # Rep count — big and bold
        cv2.putText(
            frame,
            f"REPS: {reps}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            COLORS["rep_counter"],
            3,
        )

        return frame

    @staticmethod
    def draw_form_feedback(
        frame: np.ndarray,
        message: str,
        is_good: bool = True,
    ) -> np.ndarray:
        """
        Draw a form feedback banner at the bottom of the screen.

        Args:
            frame: BGR image.
            message: Feedback text (e.g., "Great form!" or "Knees caving in!").
            is_good: True for positive feedback (green), False for corrections (red).

        Returns:
            Frame with feedback banner.
        """
        h, w = frame.shape[:2]
        color = COLORS["good_form"] if is_good else COLORS["bad_form"]

        # Banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 70), (w - 10, h - 10), COLORS["background_box"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Banner border
        cv2.rectangle(frame, (10, h - 70), (w - 10, h - 10), color, 2)

        # Status icon
        icon = "✓" if is_good else "✗"
        cv2.putText(
            frame,
            f"{icon} {message}",
            (25, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        return frame

    @staticmethod
    def draw_joint_highlight(
        frame: np.ndarray, point: tuple, color: tuple = COLORS["joint_highlight"], radius: int = 12
    ) -> np.ndarray:
        """
        Draw a glowing highlight circle around a specific joint.
        Useful for drawing attention to a problem joint.

        Args:
            frame: BGR image.
            point: (x, y) pixel coordinates.
            color: Highlight color.
            radius: Circle radius.

        Returns:
            Frame with highlighted joint.
        """
        x, y = int(point[0]), int(point[1])
        # Outer glow
        cv2.circle(frame, (x, y), radius + 4, color, 1)
        cv2.circle(frame, (x, y), radius, color, 2)
        # Inner dot
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        return frame
