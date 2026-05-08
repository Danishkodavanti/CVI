import cv2
import numpy as np
from filters.base import BaseFilter
from core.geometry import GeometricTransformer


class MaskFilter(BaseFilter):
    """Procedural face-paint / mask overlay."""

    MODES = ["phantom", "clown", "warrior"]

    def __init__(self, asset_store, mode: str = "phantom"):
        super().__init__(asset_store)
        self.mode = mode

    @property
    def name(self) -> str:
        return f"Mask:{self.mode}"

    def apply(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        if self.mode == "phantom":
            return self._phantom(frame, landmarks)
        elif self.mode == "clown":
            return self._clown(frame, landmarks)
        elif self.mode == "warrior":
            return self._warrior(frame, landmarks)
        return frame

    def _face_oval_pts(self, lm):
        OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132,
                93, 234, 127, 162, 21, 54, 103, 67, 109]
        return lm[OVAL, :2].astype(np.int32)

    def _blend(self, frame, overlay, alpha=0.55):
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    def _phantom(self, frame, lm):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        left_eye_c = lm[468, :2] if len(lm) > 468 else lm[33, :2]
        right_eye_c = lm[473, :2] if len(lm) > 473 else lm[263, :2]
        eye_dist = GeometricTransformer.eye_distance(left_eye_c, right_eye_c)
        angle = GeometricTransformer.face_angle(left_eye_c, right_eye_c)

        # White half-mask on left
        mask = np.zeros((h, w), dtype=np.uint8)
        face_pts = self._face_oval_pts(lm)
        cv2.fillPoly(mask, [face_pts], 255)

        mid_x = int((face_pts[:, 0].min() + face_pts[:, 0].max()) / 2)
        left_mask = mask.copy()
        left_mask[:, mid_x:] = 0

        white_layer = np.full_like(overlay, 230)
        alpha3 = np.stack([left_mask]*3, axis=2).astype(np.float32) / 255.0 * 0.75
        overlay = (overlay.astype(np.float32) * (1 - alpha3) +
                   white_layer.astype(np.float32) * alpha3).astype(np.uint8)

        # Black eye mask on left
        lec = left_eye_c.astype(int)
        axes = (int(eye_dist * 0.45), int(eye_dist * 0.28))
        cv2.ellipse(overlay, tuple(lec), axes, angle, 0, 360, (10, 10, 10), -1)

        # Vertical line down nose
        nose_top = lm[168, :2].astype(int)
        nose_bot = lm[2, :2].astype(int)
        cv2.line(overlay, tuple(nose_top), tuple(nose_bot), (10, 10, 10), 2)

        return overlay

    def _clown(self, frame, lm):
        overlay = frame.copy()
        left_eye_c = lm[468, :2] if len(lm) > 468 else lm[33, :2]
        right_eye_c = lm[473, :2] if len(lm) > 473 else lm[263, :2]
        eye_dist = GeometricTransformer.eye_distance(left_eye_c, right_eye_c)
        angle = GeometricTransformer.face_angle(left_eye_c, right_eye_c)

        # Red nose
        nose = lm[4, :2].astype(int)
        nr = int(eye_dist * 0.2)
        cv2.circle(overlay, tuple(nose), nr, (0, 0, 220), -1)
        cv2.circle(overlay, tuple(nose - nr//3), nr//4, (80, 80, 255), -1)

        # Colored eye circles
        for center, color in [(left_eye_c, (255, 50, 50)), (right_eye_c, (50, 50, 255))]:
            c = center.astype(int)
            cv2.circle(overlay, tuple(c), int(eye_dist * 0.35), color, -1)
            cv2.circle(overlay, tuple(c), int(eye_dist * 0.35), (255, 255, 255), 3)

        # Smile lines
        mouth_l = lm[61, :2].astype(int)
        mouth_r = lm[291, :2].astype(int)
        mid = ((mouth_l + mouth_r) / 2).astype(int)
        ext_l = mouth_l + (mouth_l - mid) * 2
        ext_r = mouth_r + (mouth_r - mid) * 2
        cv2.line(overlay, tuple(mouth_l), tuple(ext_l), (0, 0, 200), 3)
        cv2.line(overlay, tuple(mouth_r), tuple(ext_r), (0, 0, 200), 3)

        return self._blend(frame, overlay, 0.65)

    def _warrior(self, frame, lm):
        overlay = frame.copy()
        left_eye_c = lm[468, :2] if len(lm) > 468 else lm[33, :2]
        right_eye_c = lm[473, :2] if len(lm) > 473 else lm[263, :2]
        eye_dist = GeometricTransformer.eye_distance(left_eye_c, right_eye_c)

        # War stripes across cheeks
        l_cheek = lm[205, :2].astype(int)
        r_cheek = lm[425, :2].astype(int)
        stripe_w = int(eye_dist * 0.12)
        stripe_h = int(eye_dist * 0.6)

        for cheek, direction in [(l_cheek, 1), (r_cheek, -1)]:
            for i, color in enumerate([(0, 0, 180), (0, 120, 255), (0, 0, 180)]):
                offset_x = direction * i * stripe_w
                x = cheek[0] + offset_x
                pts = np.array([
                    [x, cheek[1] - stripe_h // 2],
                    [x + stripe_w, cheek[1] - stripe_h // 2],
                    [x + stripe_w, cheek[1] + stripe_h // 2],
                    [x, cheek[1] + stripe_h // 2]
                ], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color)

        # Dark eye shadow
        for center in [left_eye_c, right_eye_c]:
            c = center.astype(int)
            axes = (int(eye_dist * 0.4), int(eye_dist * 0.22))
            cv2.ellipse(overlay, tuple(c), axes, 0, 0, 360, (20, 10, 80), -1)

        return self._blend(frame, overlay, 0.55)