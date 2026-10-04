"""웹캠에서 MediaPipe로 uniScale 인터랙션을 실행합니다.

실행:
    python3.11 main.py
    python3.11 main.py --camera 1 --width 960 --height 540
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from interactions.common import METHOD_KEYS
from interactions.controller import InteractionController
from scene import create_default_objects
from tracking.mediapipe_tracker import MediaPipeTracker
from tracking.types import FrameContext
from ui import draw_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe로 uniScale 인터랙션을 실행합니다.")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--no-mirror", action="store_true", help="좌우 반전 없이 원본 카메라 영상을 사용합니다.")
    return parser.parse_args()


def _camera_api() -> int:
    if sys.platform == "darwin":
        return int(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
    if sys.platform == "win32":
        return int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
    return int(cv2.CAP_ANY)


def _capture(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, _camera_api())
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if capture.isOpened() and width > 0 and height > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if capture.isOpened() and sys.platform != "darwin":
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _preview_frame(capture: cv2.VideoCapture, attempts: int = 20):
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    return None


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = _capture(index, width, height)
    if sys.platform != "darwin" or not capture.isOpened():
        return capture
    # AVFoundation은 지원하지 않는 해상도면 프레임을 주지 않는다.
    if _preview_frame(capture) is not None:
        return capture
    capture.release()
    time.sleep(0.3)
    return _capture(index, 0, 0)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = parse_args()
    mirror = not args.no_mirror
    root = Path(__file__).resolve().parent
    tracker = MediaPipeTracker(root / "models", mirror_handedness=mirror)
    controller = InteractionController()
    objects = create_default_objects()
    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        tracker.close()
        raise SystemExit(f"웹캠 {args.camera}번을 열 수 없습니다. --camera 번호로 다시 시도하세요.")

    window = "uniScale MediaPipe"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    previous = time.perf_counter()
    fps = 0.0
    waiting_for_camera = True
    missed = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                if not waiting_for_camera or missed > 60:
                    if sys.platform == "darwin":
                        print("카메라 프레임을 읽지 못했습니다. 시스템 설정 > 개인 정보 보호 및 보안 > 카메라에서 터미널 권한을 확인해 주세요.")
                    else:
                        print("카메라 프레임을 읽지 못했습니다.")
                    break
                missed += 1
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
                continue
            waiting_for_camera = False
            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = min(max(now - previous, 1e-4), 0.05)
            previous = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

            left, right, gaze = tracker.process(frame)
            ctx = FrameContext(
                time=now,
                dt=dt,
                width=frame.shape[1],
                height=frame.shape[0],
                left=left,
                right=right,
                gaze=gaze,
                objects=objects,
            )
            controller.update(ctx)
            lines = controller.status()
            if not gaze.tracked:
                lines.insert(1, "시선 없음 — 얼굴이 카메라에 보여야 고를 수 있습니다")
            if left is None and right is None:
                lines.insert(1, "손 없음")
            image = draw_scene(frame, ctx, controller, lines, fps, tracker.gain)
            cv2.imshow(window, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in METHOD_KEYS:
                controller.set_method(METHOD_KEYS[key])
            elif key in (ord("f"), ord("F")):
                controller.toggle_clutching()
            elif key in (ord("h"), ord("H")):
                controller.toggle_dominant()
            elif key in (ord("c"), ord("C")):
                tracker.request_calibration()
            elif key in (ord("p"), ord("P")):
                controller.uni_semi.toggle_palm_sign()
            elif key in (ord("r"), ord("R")):
                for obj in objects:
                    obj.reset_pose()
                controller.reset_interactions()
            elif key == ord("["):
                tracker.gain = max(2.0, tracker.gain - 0.5)
            elif key == ord("]"):
                tracker.gain = min(20.0, tracker.gain + 0.5)
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
