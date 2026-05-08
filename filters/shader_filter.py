import cv2
import numpy as np
from filters.base import BaseFilter


class ShaderFilter(BaseFilter):
    """Pixel-level FX: blur, HSV color shift, edge glow."""

    MODES = ["beauty", "vintage", "neon", "heatmap"]

    def __init__(self, asset_store, mode: str = "beauty"):
        super().__init__(asset_store)
        self.mode = mode

    @property
    def name(self) -> str:
        return f"Shader:{self.mode}"

    def apply(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        if self.mode == "beauty":
            return self._beauty(frame, landmarks)
        elif self.mode == "vintage":
            return self._vintage(frame)
        elif self.mode == "neon":
            return self._neon(frame)
        elif self.mode == "heatmap":
            return self._heatmap(frame)
        return frame

    # ---- Face mask for face-only effects ----
    def _face_mask(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132,
                93, 234, 127, 162, 21, 54, 103, 67, 109]
        pts = landmarks[OVAL, :2].astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _beauty(self, frame, landmarks):
        mask = self._face_mask(frame, landmarks)
        # Bilateral filter for skin smoothing
        smooth = cv2.bilateralFilter(frame, 15, 75, 75)
        mask3 = cv2.merge([mask, mask, mask])
        alpha = mask3.astype(np.float32) / 255.0
        result = (smooth.astype(np.float32) * alpha +
                  frame.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        # Slight brightness + saturation boost
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _vintage(self, frame):
        # Desaturate slightly + warm tone
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.6, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Warm overlay
        warm = np.zeros_like(result)
        warm[:, :] = [0, 20, 40]  # BGR: slight red-orange
        result = cv2.addWeighted(result, 0.85, warm, 0.15, 0)
        # Vignette
        h, w = result.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        vign = 1 - np.clip(((X - cx)**2 + (Y - cy)**2) / (cx**2 + cy**2), 0, 1)
        vign = np.stack([vign]*3, axis=2).astype(np.float32)
        result = (result.astype(np.float32) * vign).astype(np.uint8)
        return result

    def _neon(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Colorize edges
        hsv_edge = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2HSV)
        hsv_edge[:, :, 0] = 150  # cyan-ish hue
        hsv_edge[:, :, 1] = 255
        colored = cv2.cvtColor(hsv_edge, cv2.COLOR_HSV2BGR)
        # Glow: blur then add
        glow = cv2.GaussianBlur(colored, (11, 11), 0)
        dark = (frame.astype(np.float32) * 0.4).astype(np.uint8)
        result = cv2.add(dark, glow)
        return result

    def _heatmap(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)