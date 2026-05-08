import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceDetector as MpFaceDetector
from mediapipe.tasks.python.vision import FaceDetectorOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from dataclasses import dataclass
from typing import Optional
import urllib.request
import os


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[FaceDetector] Downloading face detection model (~1MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[FaceDetector] Model downloaded.")


@dataclass
class FaceDetection:
    bbox: tuple
    keypoints: np.ndarray
    confidence: float


class FaceDetector:
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self):
        _ensure_model()
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            min_detection_confidence=self.CONFIDENCE_THRESHOLD,
        )
        self._detector = MpFaceDetector.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray) -> Optional[FaceDetection]:
        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return None

        det = result.detections[0]
        score = det.categories[0].score
        if score < self.CONFIDENCE_THRESHOLD:
            return None

        bb = det.bounding_box
        x, y = max(0, bb.origin_x), max(0, bb.origin_y)

        kps = []
        for kp in det.keypoints:
            kps.append([int(kp.x * w), int(kp.y * h)])

        return FaceDetection(
            bbox=(x, y, bb.width, bb.height),
            keypoints=np.array(kps, dtype=np.float32),
            confidence=float(score),
        )

    def close(self):
        self._detector.close()