"""
Exercises module for the AI Personal Trainer.
Each exercise implements the BaseExercise interface.
"""

from .base_exercise import BaseExercise
from .squat import SquatDetector
from .bicep_curl import BicepCurlDetector
from .pushup import PushupDetector

EXERCISE_REGISTRY: dict[str, type[BaseExercise]] = {
    "squat": SquatDetector,
    "bicep_curl_left": lambda: BicepCurlDetector(side="left"),
    "bicep_curl_right": lambda: BicepCurlDetector(side="right"),
    "pushup": PushupDetector,
}

__all__ = [
    "BaseExercise",
    "SquatDetector",
    "BicepCurlDetector",
    "PushupDetector",
    "EXERCISE_REGISTRY",
]
