import cv2
import numpy as np


def draw_hud(frame: np.ndarray, filter_name: str, fps: float,
             all_filters: list[str], active_idx: int,
             show_landmarks: bool = False,
             landmarks: np.ndarray = None) -> np.ndarray:
    h, w = frame.shape[:2]
    result = frame.copy()

    # Semi-transparent top bar
    bar_h = 36
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    result[:bar_h] = cv2.addWeighted(result[:bar_h], 0.4, bar, 0.6, 0)

    # Filter name
    cv2.putText(result, f"Filter: {filter_name}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 180), 2)

    # FPS
    fps_str = f"{fps:.1f} FPS"
    tw, _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], 0
    cv2.putText(result, fps_str, (w - 90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Controls bar at bottom
    ctrl_y = h - 10
    controls = "[A/D] prev/next  [1-9] jump  [L] landmarks  [Q] quit"
    cv2.putText(result, controls, (8, ctrl_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Mini filter list on right
    panel_w = 160
    panel_x = w - panel_w - 5
    for i, name in enumerate(all_filters):
        ty = 50 + i * 18
        if ty > h - 30:
            break
        color = (0, 230, 180) if i == active_idx else (140, 140, 140)
        prefix = ">" if i == active_idx else " "
        cv2.putText(result, f"{prefix}{i+1:2d}. {name[:14]}", (panel_x, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Landmarks dots
    if show_landmarks and landmarks is not None:
        for pt in landmarks[:, :2]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(result, (x, y), 1, (0, 255, 100), -1)

    return result