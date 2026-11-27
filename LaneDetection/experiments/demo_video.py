import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config as cfg
from baseline import run_baseline
from proposed import run_proposed
from utils import line_x, load_params


def draw_lines(frame, lines, color=(0, 0, 255)):
    result = frame.copy()
    height, _ = result.shape[:2]
    y_top = int(cfg.ROI_TOP * height)
    for rho, theta, _ in lines:
        x_top = line_x(rho, theta, y_top)
        x_bottom = line_x(rho, theta, height - 1)
        if not np.isfinite(x_top) or not np.isfinite(x_bottom):
            continue
        cv2.line(
            result,
            (int(round(x_top)), y_top),
            (int(round(x_bottom)), height - 1),
            color,
            3,
            cv2.LINE_AA,
        )
    return result


def add_title(frame, text, color=(255, 255, 255)):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Phát video và so sánh trực quan baseline với phương pháp P."
    )
    parser.add_argument(
        "--video",
        default="data/demo/test_video.mp4",
        help="video nằm trong data/ (mặc định: data/demo/test_video.mp4)",
    )
    args = parser.parse_args()

    try:
        video_path = cfg.resolve_data_path(args.video)
    except ValueError as exc:
        parser.error(str(exc))
    if not video_path.is_file():
        parser.error(f"Không tìm thấy video: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Không thể mở video: {video_path}")

    params = load_params()
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = max(1, round(1000 / source_fps)) if source_fps > 0 else 1
    window_name = "So sanh Baseline va Proposed - Q/ESC: thoat, SPACE: tam dung"
    frame_index = 0
    paused = False
    smoothed_fps = 0.0
    previous_time = time.perf_counter()

    print(f"Đang phát: {video_path}")
    print("Nhấn SPACE để tạm dừng/tiếp tục; nhấn Q hoặc ESC để thoát.")
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = cv2.resize(
                    frame,
                    None,
                    fx=cfg.SCALE,
                    fy=cfg.SCALE,
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                baseline_lines = run_baseline(gray, params)
                proposed_lines = run_proposed(gray, params)
                baseline_view = draw_lines(frame, baseline_lines)
                proposed_view = draw_lines(frame, proposed_lines)

                now = time.perf_counter()
                elapsed = now - previous_time
                previous_time = now
                if elapsed > 0:
                    current_fps = 1.0 / elapsed
                    smoothed_fps = (
                        current_fps
                        if smoothed_fps == 0
                        else 0.85 * smoothed_fps + 0.15 * current_fps
                    )

                add_title(
                    baseline_view,
                    f"Baseline | frame {frame_index} | lines {len(baseline_lines)}",
                )
                add_title(
                    proposed_view,
                    f"Proposed P | frame {frame_index} | lines {len(proposed_lines)} | {smoothed_fps:.1f} FPS",
                )
                combined = np.hstack((baseline_view, proposed_view))
                cv2.imshow(window_name, combined)
                frame_index += 1

            key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                if not paused:
                    previous_time = time.perf_counter()
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
