import cv2
import numpy as np
from filters.base import BaseFilter
from core.asset_store import AssetStore
from core.geometry import GeometricTransformer


class DogFilter(BaseFilter):
    """Dog ears + nose overlay anchored to face landmarks."""

    # Landmark indices
    LEFT_EAR_ANCHOR = 234   # left cheek outer
    RIGHT_EAR_ANCHOR = 454  # right cheek outer
    NOSE_ANCHOR = 4         # nose tip
    FOREHEAD_L = 103
    FOREHEAD_R = 332
    FOREHEAD_TOP = 10

    @property
    def name(self) -> str:
        return "Dog"

    def apply(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        lm = landmarks

        left_eye_c = lm[468, :2] if len(lm) > 468 else lm[33, :2]
        right_eye_c = lm[473, :2] if len(lm) > 473 else lm[263, :2]

        angle = GeometricTransformer.face_angle(left_eye_c, right_eye_c)
        eye_dist = GeometricTransformer.eye_distance(left_eye_c, right_eye_c)
        scale = eye_dist / 60.0

        result = frame.copy()

        # ---- Draw procedural dog ears ----
        result = self._draw_ears(result, lm, angle, scale, w, h)

        # ---- Draw nose ----
        nose_pt = lm[self.NOSE_ANCHOR, :2].astype(int)
        nose_r = int(eye_dist * 0.18)
        cv2.ellipse(result, tuple(nose_pt), (nose_r, int(nose_r * 0.7)),
                    angle, 0, 360, (20, 20, 120), -1)
        cv2.ellipse(result, tuple(nose_pt), (nose_r, int(nose_r * 0.7)),
                    angle, 0, 360, (0, 0, 80), 2)

        # Nose shine
        shine_offset = np.array([-nose_r // 4, -nose_r // 4], dtype=int)
        cv2.circle(result, tuple(nose_pt + shine_offset), max(2, nose_r // 5),
                   (160, 160, 220), -1)

        return result

    def _draw_ears(self, frame, lm, angle, scale, w, h):
        forehead_l = lm[self.FOREHEAD_L, :2]
        forehead_r = lm[self.FOREHEAD_R, :2]
        top = lm[self.FOREHEAD_TOP, :2]

        ear_w = int(35 * scale)
        ear_h = int(55 * scale)

        for side, anchor in [(-1, forehead_l), (1, forehead_r)]:
            cx, cy = int(anchor[0]), int(anchor[1])
            # Offset upward from forehead
            cy -= int(20 * scale)

            ear_angle = angle + side * 15

            # Outer ear (brown)
            pts_outer = self._ear_polygon(cx, cy, ear_w, ear_h, ear_angle)
            cv2.fillPoly(frame, [pts_outer], (30, 80, 140))

            # Inner ear (pink)
            pts_inner = self._ear_polygon(cx, cy,
                                          int(ear_w * 0.55), int(ear_h * 0.65),
                                          ear_angle)
            cv2.fillPoly(frame, [pts_inner], (120, 100, 200))

        return frame

    def _ear_polygon(self, cx, cy, w, h, angle_deg):
        """Triangle-ish ear shape."""
        pts = np.array([
            [0, h],
            [-w // 2, 0],
            [w // 2, 0],
        ], dtype=np.float32)
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = (R @ pts.T).T
        rotated[:, 0] += cx
        rotated[:, 1] += cy
        return rotated.astype(np.int32)