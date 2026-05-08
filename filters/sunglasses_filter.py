import cv2
import numpy as np
from filters.base import BaseFilter
from core.geometry import GeometricTransformer


class SunglassesFilter(BaseFilter):
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOT = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOT = 374

    @property
    def name(self) -> str:
        return "Sunglasses"

    def apply(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        lm = landmarks
        h, w = frame.shape[:2]

        left_eye_c = lm[468, :2] if len(lm) > 468 else lm[33, :2]
        right_eye_c = lm[473, :2] if len(lm) > 473 else lm[263, :2]
        angle = GeometricTransformer.face_angle(left_eye_c, right_eye_c)
        eye_dist = GeometricTransformer.eye_distance(left_eye_c, right_eye_c)

        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        result = frame.copy()

        pad_x = int(eye_dist * 0.55)
        pad_y = int(eye_dist * 0.38)

        for center in [left_eye_c, right_eye_c]:
            cx, cy = int(center[0]), int(center[1])
            axes = (int(eye_dist * 0.38), int(eye_dist * 0.26))

            # Lens
            lens_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            cv2.ellipse(lens_overlay, (cx, cy), axes, angle, 0, 360,
                        (20, 20, 20, 200), -1)

            # Tinted reflection
            ref_cx = cx - int(axes[0] * 0.2)
            ref_cy = cy - int(axes[1] * 0.3)
            cv2.ellipse(lens_overlay, (ref_cx, ref_cy),
                        (int(axes[0] * 0.35), int(axes[1] * 0.2)),
                        angle, 0, 360, (80, 80, 100, 60), -1)

            # Lens rim
            cv2.ellipse(lens_overlay, (cx, cy), axes, angle, 0, 360,
                        (30, 20, 10, 255), 2)

            alpha = lens_overlay[:, :, 3:4].astype(np.float32) / 255.0
            result = (result.astype(np.float32) * (1 - alpha) +
                      lens_overlay[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

        # Bridge between lenses
        lx, ly = int(left_eye_c[0]), int(left_eye_c[1])
        rx, ry = int(right_eye_c[0]), int(right_eye_c[1])
        mid_l = (int(lx + (rx - lx) * 0.38), int(ly + (ry - ly) * 0.38))
        mid_r = (int(lx + (rx - lx) * 0.62), int(ry + (ly - ry) * 0.38))
        cv2.line(result, mid_l, mid_r, (30, 20, 10), 2)

        # Temples
        left_outer = lm[self.LEFT_EYE_OUTER, :2].astype(int)
        right_outer = lm[self.RIGHT_EYE_OUTER, :2].astype(int)
        ear_l = lm[234, :2].astype(int)
        ear_r = lm[454, :2].astype(int)

        temple_color = (30, 20, 10)
        cv2.line(result, tuple(left_outer - [int(eye_dist*0.35), 0]), tuple(ear_l), temple_color, 2)
        cv2.line(result, tuple(right_outer + [int(eye_dist*0.35), 0]), tuple(ear_r), temple_color, 2)

        return result