import cv2
import numpy as np


class FramePreprocessor:
    def __init__(self, target_size=(640, 360)):
        self.target_size = target_size

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (resized_bgr, rgb_normalized) tuple."""
        resized = cv2.resize(frame, self.target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return resized, rgb, normalized