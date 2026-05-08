import sys
import os
import time
import cv2
import numpy as np

# Resolve imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.preprocessor import FramePreprocessor
from core.face_detector import FaceDetector
from core.landmark_detector import LandmarkDetector
from core.overlay_engine import FilterOverlayEngine
from core.performance_gate import PerformanceGate
from utils.hud import draw_hud


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    preprocessor = FramePreprocessor(target_size=(640, 360))
    face_detector = FaceDetector()
    landmark_detector = LandmarkDetector()
    engine = FilterOverlayEngine(assets_dir=os.path.join(
        os.path.dirname(__file__), "assets"))
    gate = PerformanceGate()

    show_landmarks = False
    frame_count = 0
    fps_timer = time.perf_counter()
    display_fps = 0.0

    print("=" * 50)
    print("  Real-Time Face Filter — OpenCV + MediaPipe")
    print("  [A] Prev filter  [D] Next filter")
    print("  [1-9] Jump to filter  [L] Toggle landmarks")
    print("  [Q] Quit")
    print("=" * 50)

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, retrying...")
            time.sleep(0.01)
            continue

        # ── Pre-process ──────────────────────────────────────
        bgr_frame, rgb_frame, _ = preprocessor.process(raw_frame)

        # ── Face detection ───────────────────────────────────
        face = face_detector.detect(rgb_frame)
        landmarks = None

        if face is not None:
            # ── Landmark detection ────────────────────────────
            landmarks = landmark_detector.detect(rgb_frame)

        # ── Filter overlay ───────────────────────────────────
        if landmarks is not None:
            output = engine.apply(bgr_frame, landmarks)
        else:
            output = bgr_frame  # pass-through if no face

        # ── FPS calc ─────────────────────────────────────────
        frame_count += 1
        now = time.perf_counter()
        if now - fps_timer >= 0.5:
            display_fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ── HUD ──────────────────────────────────────────────
        output = draw_hud(
            output,
            filter_name=engine.active_filter_name,
            fps=display_fps,
            all_filters=engine.filter_names,
            active_idx=engine._idx,
            show_landmarks=show_landmarks,
            landmarks=landmarks,
        )

        # ── No-face indicator ────────────────────────────────
        if face is None:
            h, w = output.shape[:2]
            cv2.putText(output, "No face detected", (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        # ── Display ──────────────────────────────────────────
        cv2.imshow("Face Filter", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d') or key == ord('n'):
            engine.next_filter()
        elif key == ord('a') or key == ord('p'):
            engine.prev_filter()
        elif key == ord('l'):
            show_landmarks = not show_landmarks
        elif ord('1') <= key <= ord('9'):
            engine.set_filter(key - ord('1'))

    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    landmark_detector.close()
    print("Bye.")


if __name__ == "__main__":
    main()