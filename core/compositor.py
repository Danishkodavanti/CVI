import cv2
import numpy as np


class AlphaCompositor:
    """Alpha compositing with anti-alias and gamma correction."""

    GAMMA = 2.2

    @staticmethod
    def composite(base: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
        """
        Alpha-blend BGRA overlay onto BGR base frame.
        Both must be the same spatial size.
        """
        if overlay_bgra.shape[2] < 4:
            return base

        alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
        overlay_bgr = overlay_bgra[:, :, :3].astype(np.float32)
        base_f = base.astype(np.float32)

        blended = base_f * (1.0 - alpha) + overlay_bgr * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def composite_gamma(base: np.ndarray, overlay_bgra: np.ndarray,
                        gamma: float = 2.2) -> np.ndarray:
        """Gamma-correct composite."""
        if overlay_bgra.shape[2] < 4:
            return base

        alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
        inv_gamma = 1.0 / gamma

        base_lin = np.power(base.astype(np.float32) / 255.0, gamma)
        overlay_lin = np.power(overlay_bgra[:, :, :3].astype(np.float32) / 255.0, gamma)

        blended_lin = base_lin * (1.0 - alpha) + overlay_lin * alpha
        blended = np.power(np.clip(blended_lin, 0, 1), inv_gamma) * 255.0
        return blended.astype(np.uint8)

    @staticmethod
    def add_weighted(base: np.ndarray, overlay: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
        return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)