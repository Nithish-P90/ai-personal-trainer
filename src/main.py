"""
AI Personal Trainer — Live Webcam Application
===============================================
Run this to start a live pose-tracking session.

Usage:
    python -m src.main                    # Default: squat detection
    python -m src.main --exercise squat
    python -m src.main --exercise bicep_curl_left
    python -m src.main --exercise pushup

Controls:
    Q / ESC  — Quit
    R        — Reset rep counter
    1        — Switch to Squat
    2        — Switch to Bicep Curl (Left)
    3        — Switch to Bicep Curl (Right)
    4        — Switch to Push-Up
"""

import sys
import argparse
import time

import cv2
import numpy as np

from src.vision.pose_detector import PoseDetector
from src.vision.visualizer import Visualizer, COLORS
from src.vision.angle_calculator import AngleCalculator
from src.exercises.squat import SquatDetector
from src.exercises.bicep_curl import BicepCurlDetector
from src.exercises.pushup import PushupDetector


# ─── Exercise Selection ────────────────────────────────────────────────
EXERCISES = {
    "1": ("Squat", SquatDetector),
    "2": ("Bicep Curl (L)", lambda: BicepCurlDetector(side="left")),
    "3": ("Bicep Curl (R)", lambda: BicepCurlDetector(side="right")),
    "4": ("Push-Up", PushupDetector),
}


def create_exercise(key: str):
    """Instantiate an exercise detector by key."""
    name, factory = EXERCISES[key]
    return factory() if callable(factory) else factory


def draw_exercise_menu(frame: np.ndarray, active_key: str) -> np.ndarray:
    """Draw exercise selection menu on the right side of the frame."""
    h, w = frame.shape[:2]
    menu_x = w - 250

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (menu_x, 10), (w - 10, 180), COLORS["background_box"], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (menu_x, 10), (w - 10, 180), COLORS["title"], 1)

    cv2.putText(frame, "EXERCISES", (menu_x + 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["title"], 1)

    y_offset = 60
    for key, (name, _) in EXERCISES.items():
        color = COLORS["rep_counter"] if key == active_key else COLORS["title"]
        prefix = ">" if key == active_key else " "
        cv2.putText(
            frame, f"{prefix} [{key}] {name}", (menu_x + 15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )
        y_offset += 28

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter in the top-right area."""
    h, w = frame.shape[:2]
    text = f"FPS: {int(fps)}"
    cv2.putText(frame, text, (w - 250, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["title"], 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="AI Personal Trainer — Live Mode")
    parser.add_argument(
        "--exercise", type=str, default="1",
        choices=list(EXERCISES.keys()),
        help="Exercise to start with (1=Squat, 2=Curl L, 3=Curl R, 4=Pushup)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Camera width",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Camera height",
    )
    args = parser.parse_args()

    # ─── Initialize Components ─────────────────────────────────────────
    print("🏋️ AI Personal Trainer — Starting Live Mode...")

    detector = PoseDetector(model_complexity=1, min_detection_confidence=0.6)
    visualizer = Visualizer()
    active_key = args.exercise
    exercise = create_exercise(active_key)

    # ─── Camera Setup ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("❌ Error: Cannot open camera. Check your webcam connection.")
        sys.exit(1)

    print(f"📷 Camera opened at {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"🎯 Active exercise: {exercise.name}")
    print("Press Q to quit, R to reset, 1-4 to switch exercises.")

    # ─── Main Loop ─────────────────────────────────────────────────────
    prev_time = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame dropped.")
                continue

            # Flip for mirror effect (more natural for self-monitoring)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ─── Pose Detection ────────────────────────────────────────
            pose_detected = detector.detect(frame_rgb)

            if pose_detected:
                # Draw skeleton
                frame = visualizer.draw_skeleton(frame, detector.results)

                # Get landmark array
                landmarks = detector.get_landmark_array(w, h)

                if landmarks is not None:
                    # ─── Exercise Analysis ─────────────────────────────
                    result = exercise.analyze_frame(landmarks)

                    # Draw angles on key joints
                    for joint_name, angle_val in result["angles"].items():
                        # Map angle name back to a landmark for positioning
                        if joint_name in PoseDetector.LANDMARKS:
                            idx = PoseDetector.LANDMARKS[joint_name]
                            pt = landmarks[idx]
                            frame = visualizer.draw_angle(frame, pt, angle_val)

                    # Highlight problem joints
                    for joint_idx in result["highlight_joints"]:
                        pt = landmarks[joint_idx]
                        frame = visualizer.draw_joint_highlight(
                            frame, pt, color=COLORS["bad_form"]
                        )

                    # Draw rep counter
                    frame = visualizer.draw_rep_counter(
                        frame, exercise.rep_count, exercise.name
                    )

                    # Draw form feedback
                    if result["feedback"]:
                        msg = result["feedback"][0]  # Show primary feedback
                        frame = visualizer.draw_form_feedback(
                            frame, msg, is_good=result["form_ok"]
                        )
            else:
                # No pose detected
                frame = visualizer.draw_form_feedback(
                    frame, "Step into frame to begin", is_good=True
                )

            # ─── UI Overlays ──────────────────────────────────────────
            frame = draw_exercise_menu(frame, active_key)

            # FPS counter
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time
            frame = draw_fps(frame, fps)

            # ─── Display ──────────────────────────────────────────────
            cv2.imshow("AI Personal Trainer", frame)

            # ─── Keyboard Controls ────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("r"):
                exercise.reset()
                print(f"🔄 Rep counter reset for {exercise.name}")
            elif chr(key) in EXERCISES:
                active_key = chr(key)
                exercise = create_exercise(active_key)
                print(f"🎯 Switched to: {exercise.name}")

    except KeyboardInterrupt:
        print("\n👋 Session ended.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("✅ Cleanup complete.")


if __name__ == "__main__":
    main()
