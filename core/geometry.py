import cv2
import numpy as np


class GeometricTransformer:
    """Affine warp, perspective, scale/rotation utilities."""

    @staticmethod
    def get_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """Compute affine matrix from 3 point pairs."""
        return cv2.getAffineTransform(
            src_pts[:3].astype(np.float32),
            dst_pts[:3].astype(np.float32)
        )

    @staticmethod
    def warp_affine(img: np.ndarray, M: np.ndarray,
                    dsize: tuple[int, int]) -> np.ndarray:
        return cv2.warpAffine(img, M, dsize,
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_TRANSPARENT)

    @staticmethod
    def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Estimate similarity (scale + rotation + translation) from 2+ point pairs."""
        src = src.astype(np.float32)
        dst = dst.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst)
        return M  # 2x3

    @staticmethod
    def scale_rotate_sprite(sprite: np.ndarray,
                            center: tuple[float, float],
                            scale: float,
                            angle_deg: float,
                            canvas_size: tuple[int, int]) -> np.ndarray:
        """
        Place sprite on blank canvas of canvas_size, scaled and rotated around center.
        Returns BGRA canvas.
        """
        h_s, w_s = sprite.shape[:2]
        cx, cy = center

        # Rotation matrix around sprite center then translate to face center
        M = cv2.getRotationMatrix2D((w_s / 2, h_s / 2), angle_deg, scale)
        new_w = int(w_s * scale)
        new_h = int(h_s * scale)
        scaled = cv2.warpAffine(sprite, M, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

        canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
        x_off = int(cx - new_w / 2)
        y_off = int(cy - new_h / 2)

        # Clip to canvas
        src_x1 = max(0, -x_off)
        src_y1 = max(0, -y_off)
        dst_x1 = max(0, x_off)
        dst_y1 = max(0, y_off)
        dst_x2 = min(canvas_size[0], x_off + new_w)
        dst_y2 = min(canvas_size[1], y_off + new_h)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

        return canvas

    @staticmethod
    def face_angle(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def eye_distance(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        return float(np.linalg.norm(right_eye - left_eye))