"""웹캠 프레임에서 MediaPipe 손 랜드마크와 홍채 시선을 읽습니다."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from tracking.types import GazeSample, TrackedHand, camera_points

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_IRIS = 468
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_IRIS = 473

INDEX_ON = 0.045
INDEX_OFF = 0.07
MIDDLE_ON = 0.05
MIDDLE_OFF = 0.08


class _Ema:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: np.ndarray | None = None

    def push(self, current: np.ndarray) -> np.ndarray:
        if self.value is None:
            self.value = current.copy()
        else:
            self.value = self.alpha * current + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class _PinchLatch:
    def __init__(self) -> None:
        self.index = False
        self.middle = False

    def update(self, world: np.ndarray) -> tuple[bool, bool]:
        index_distance = float(np.linalg.norm(world[8] - world[4]))
        middle_distance = float(np.linalg.norm(world[12] - world[4]))
        self.index = _latch(self.index, index_distance, INDEX_ON, INDEX_OFF)
        self.middle = _latch(self.middle, middle_distance, MIDDLE_ON, MIDDLE_OFF)
        return self.index, self.middle

    def reset(self) -> None:
        self.index = False
        self.middle = False


def _latch(active: bool, distance: float, on_threshold: float, off_threshold: float) -> bool:
    if active:
        return distance <= off_threshold
    return distance <= on_threshold


def _landmarks_to_array(landmarks, count: int) -> np.ndarray:
    array = np.zeros((count, 3), dtype=np.float64)
    for index, landmark in enumerate(landmarks[:count]):
        array[index, 0] = landmark.x
        array[index, 1] = landmark.y
        array[index, 2] = landmark.z
    return array


def ensure_model(path: Path, url: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"모델 받는 중: {path.name}")
    urllib.request.urlretrieve(url, path)


class MediaPipeTracker:
    """거울 화면 기준으로 사용자 왼손/오른손과 시선 점을 만듭니다."""

    def __init__(self, model_dir: Path, mirror_handedness: bool = True) -> None:
        import mediapipe as mp

        self.mirror_handedness = mirror_handedness
        self.gain = 8.0
        self.bias_x = 0.0
        self.bias_y = 0.0
        self._calibrate = False
        self._timestamp = 0
        self._gaze_smooth: np.ndarray | None = None
        self._smooth = {
            "Left": (_Ema(0.55), _Ema(0.55), _PinchLatch()),
            "Right": (_Ema(0.55), _Ema(0.55), _PinchLatch()),
        }

        hand_path = model_dir / "hand_landmarker.task"
        face_path = model_dir / "face_landmarker.task"
        ensure_model(hand_path, HAND_MODEL_URL)
        ensure_model(face_path, FACE_MODEL_URL)

        base = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        self._mp = mp
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(hand_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        try:
            self._faces = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base(model_asset_path=str(face_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        except Exception:
            self._hands.close()
            raise

    def close(self) -> None:
        self._hands.close()
        self._faces.close()

    def request_calibration(self) -> None:
        self._calibrate = True

    def process(self, frame_bgr: np.ndarray) -> tuple[TrackedHand | None, TrackedHand | None, GazeSample]:
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 33

        left = right = None
        try:
            hand_result = self._hands.detect_for_video(image, self._timestamp)
            left, right = self._read_hands(hand_result, width, height)
        except Exception as error:
            print(f"손 추적 오류: {error}")

        gaze = GazeSample(False, np.array([width * 0.5, height * 0.5], dtype=np.float64))
        try:
            face_result = self._faces.detect_for_video(image, self._timestamp)
            gazed = self._read_gaze(face_result, width, height)
            if gazed is not None:
                gaze = gazed
        except Exception as error:
            print(f"얼굴 추적 오류: {error}")

        return left, right, gaze

    def _read_hands(self, result, width: int, height: int) -> tuple[TrackedHand | None, TrackedHand | None]:
        found: dict[str, TrackedHand] = {}
        landmarks = result.hand_landmarks or []
        worlds = result.hand_world_landmarks or []
        categories = result.handedness or []
        parsed = []

        for index, hand_landmarks in enumerate(landmarks):
            if index >= len(worlds) or index >= len(categories) or not categories[index]:
                continue
            label = categories[index][0].category_name
            if self.mirror_handedness:
                label = "Left" if label == "Right" else "Right"
            image = _landmarks_to_array(hand_landmarks, 21)
            world = _landmarks_to_array(worlds[index], 21)
            parsed.append((label, float(image[0, 0]), image, world))

        if len(parsed) == 2 and parsed[0][0] == parsed[1][0]:
            parsed = sorted(parsed, key=lambda item: item[1])
            labels = ("Left", "Right")
            parsed = [(labels[i], item[1], item[2], item[3]) for i, item in enumerate(parsed)]

        seen = set()
        for label, _x, image, world in parsed:
            if label in seen:
                continue
            seen.add(label)
            image_ema, world_ema, pinch = self._smooth[label]
            image = image_ema.push(image)
            world = world_ema.push(world)
            index_pinch, middle_pinch = pinch.update(world)
            camera, pixels = camera_points(image, world, width, height)
            found[label] = TrackedHand(
                handedness=label,
                world=world,
                camera=camera,
                pixel=pixels,
                index_pinching=index_pinch,
                middle_pinching=middle_pinch,
                tracked=True,
            )

        for label in ("Left", "Right"):
            if label not in seen:
                _image_ema, _world_ema, pinch = self._smooth[label]
                _image_ema.reset()
                _world_ema.reset()
                pinch.reset()

        return found.get("Left"), found.get("Right")

    def _read_gaze(self, result, width: int, height: int) -> GazeSample | None:
        faces = result.face_landmarks or []
        if not faces:
            self._gaze_smooth = None
            return None
        face = faces[0]
        if len(face) <= RIGHT_IRIS:
            return None

        samples = []
        for outer, inner, iris in (
            (LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS),
            (RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_IRIS),
        ):
            offset = _eye_offset(face, outer, inner, iris, width, height)
            if offset is not None:
                samples.append(offset)
        if not samples:
            return None

        offset = np.mean(samples, axis=0)
        raw_x = 0.5 + float(offset[0]) * self.gain
        raw_y = 0.5 + float(offset[1]) * self.gain
        if self._calibrate:
            self.bias_x = raw_x - 0.5
            self.bias_y = raw_y - 0.5
            self._calibrate = False

        nx = float(np.clip(raw_x - self.bias_x, 0.0, 1.0))
        ny = float(np.clip(raw_y - self.bias_y, 0.0, 1.0))
        pixel = np.array([nx * width, ny * height], dtype=np.float64)
        if self._gaze_smooth is None:
            self._gaze_smooth = pixel
        else:
            self._gaze_smooth = 0.45 * pixel + 0.55 * self._gaze_smooth

        direction = np.array(
            [float(self._gaze_smooth[0] / width - 0.5), float(0.5 - self._gaze_smooth[1] / height), 1.0]
        )
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        return GazeSample(
            tracked=True,
            pixel=self._gaze_smooth.copy(),
            origin=np.array([0.0, 0.0, 0.0]),
            direction=direction,
        )


def _eye_offset(face, outer: int, inner: int, iris: int, width: int, height: int) -> np.ndarray | None:
    outer_point = np.array([face[outer].x * width, face[outer].y * height], dtype=np.float64)
    inner_point = np.array([face[inner].x * width, face[inner].y * height], dtype=np.float64)
    iris_point = np.array([face[iris].x * width, face[iris].y * height], dtype=np.float64)
    eye_width = float(np.linalg.norm(inner_point - outer_point))
    if eye_width < 1.0:
        return None
    center = (outer_point + inner_point) * 0.5
    return (iris_point - center) / eye_width
