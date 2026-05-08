import cv2
import numpy as np
from filters.base import BaseFilter
from core.geometry import GeometricTransformer


class CrownFilter(BaseFilter):
    FOREHEAD_TOP = 10
    FOREHEAD_L = 103
    FOREHEAD_R = 332
    TEMPLE_L = 127
    TEMPLE_R = 356

    @property
    def name(self) -> str:
        return "Crown"

    def apply(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        lm = landmarks
        result = frame.copy()

        left_eye_c = lm[468, :2] if len(lm) > 468 else lm[33, :2]
        right_eye_c = lm[473, :2] if len(lm) > 473 else lm[263, :2]
        angle = GeometricTransformer.face_angle(left_eye_c, right_eye_c)
        eye_dist = GeometricTransformer.eye_distance(left_eye_c, right_eye_c)

        top = lm[self.FOREHEAD_TOP, :2]
        temple_l = lm[self.TEMPLE_L, :2]
        temple_r = lm[self.TEMPLE_R, :2]

        # Crown base bar
        base_y = int(top[1]) - int(eye_dist * 0.05)
        bar_l = int(temple_l[0])
        bar_r = int(temple_r[0])
        bar_h = int(eye_dist * 0.18)

        overlay = result.copy()

        # Gold base
        cv2.rectangle(overlay, (bar_l, base_y - bar_h), (bar_r, base_y),
                      (0, 180, 255), -1)
        cv2.rectangle(overlay, (bar_l, base_y - bar_h), (bar_r, base_y),
                      (0, 120, 200), 2)

        # Spikes
        n_spikes = 5
        spike_h = int(eye_dist * 0.38)
        spike_w = (bar_r - bar_l) // (n_spikes * 2 - 1)
        colors = [(0, 200, 255), (0, 150, 220), (0, 200, 255),
                  (0, 150, 220), (0, 200, 255)]
        gem_colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255),
                      (200, 50, 200), (50, 200, 200)]

        for i in range(n_spikes):
            sx = bar_l + i * 2 * spike_w + spike_w // 2
            sy = base_y - bar_h
            tip_x = sx + spike_w // 2
            tip_y = sy - spike_h - int(i == 2) * int(spike_h * 0.3)  # center taller

            pts = np.array([
                [sx, sy],
                [tip_x, tip_y],
                [sx + spike_w, sy]
            ], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], colors[i])
            cv2.polylines(overlay, [pts], True, (0, 100, 180), 2)

            # Gem on tip
            gem_r = max(4, spike_w // 5)
            cv2.circle(overlay, (tip_x, tip_y + gem_r + 2), gem_r, gem_colors[i], -1)
            cv2.circle(overlay, (tip_x, tip_y + gem_r + 2), gem_r, (255, 255, 255), 1)

        # Gems along base
        gem_y = base_y - bar_h // 2
        for i in range(n_spikes):
            gx = bar_l + i * 2 * spike_w + spike_w
            cv2.circle(overlay, (gx, gem_y), max(4, spike_w // 4),
                       gem_colors[(i + 2) % 5], -1)

        result = cv2.addWeighted(result, 0.15, overlay, 0.85, 0)
        return result