import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import Optional
import urllib.request
import os


LANDMARK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
LANDMARK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_landmark_model():
    if not os.path.exists(LANDMARK_MODEL_PATH):
        print("[LandmarkDetector] Downloading face landmark model (~30MB)...")
        urllib.request.urlretrieve(LANDMARK_MODEL_URL, LANDMARK_MODEL_PATH)
        print("[LandmarkDetector] Model downloaded.")


NOSE_TIP = 1
LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336
CHIN = 152
FOREHEAD = 10

FEATURE_INDICES = {
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "nose": [1, 2, 3, 4, 5, 6, 168, 197, 195, 5],
    "mouth": [61, 84, 17, 314, 291, 308, 402, 13, 82, 312],
    "left_brow": [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "jaw": [10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136, 172, 58, 132,
            93, 234, 127, 162, 21, 54, 103, 67, 109],
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                  288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                  150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}


class LandmarkDetector:
    def __init__(self):
        _ensure_landmark_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARK_MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
        return pts

    def get_feature_points(self, landmarks: np.ndarray, feature: str) -> np.ndarray:
        idx = FEATURE_INDICES.get(feature, [])
        return landmarks[idx, :2]

    def get_eye_centers(self, landmarks: np.ndarray):
        left = landmarks[min(LEFT_EYE_CENTER, len(landmarks) - 1), :2]
        right = landmarks[min(RIGHT_EYE_CENTER, len(landmarks) - 1), :2]
        return left, right

    def close(self):
        self._landmarker.close()