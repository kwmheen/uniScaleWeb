"""손·시선 샘플과 프레임 컨텍스트.

MediaPipe 랜드마크를 Unity HandInput / GazeInput에 대응하는 값으로 바꿉니다.
월드 좌표는 손 중심 기준 미터, 카메라 좌표는 양손이 공유하는 대략적 미터 공간입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# 손목–중지 MCP 실제 길이. 이미지 크기를 미터로 바꿀 때 사용합니다.
REAL_PALM_M = 0.09


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vector / length


def empty_landmarks() -> np.ndarray:
    return np.zeros((21, 3), dtype=np.float64)


def empty_pixels() -> np.ndarray:
    return np.zeros((21, 2), dtype=np.float64)


@dataclass
class TrackedHand:
    """한 손의 추적 결과. handedness는 사용자 기준 Left / Right."""

    handedness: str
    world: np.ndarray
    camera: np.ndarray
    pixel: np.ndarray
    index_pinching: bool = False
    middle_pinching: bool = False
    tracked: bool = True

    def world_distance(self, start: int, end: int) -> float:
        if not self.tracked:
            return 0.0
        return float(np.linalg.norm(self.world[start] - self.world[end]))

    @property
    def index_thumb_distance(self) -> float:
        return self.world_distance(INDEX_TIP, THUMB_TIP)

    @property
    def middle_thumb_distance(self) -> float:
        return self.world_distance(MIDDLE_TIP, THUMB_TIP)

    @property
    def index_position(self) -> np.ndarray:
        return self.camera[INDEX_TIP]

    @property
    def wrist_pixel(self) -> np.ndarray:
        return self.pixel[WRIST]


@dataclass
class GazeSample:
    tracked: bool
    pixel: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class FrameContext:
    time: float
    dt: float
    width: int
    height: int
    left: TrackedHand | None
    right: TrackedHand | None
    gaze: GazeSample
    objects: list

    def dominant(self, dominant_right: bool) -> TrackedHand | None:
        return self.right if dominant_right else self.left

    def nondominant(self, dominant_right: bool) -> TrackedHand | None:
        return self.left if dominant_right else self.right

    def gazed_object(self):
        if not self.gaze.tracked:
            return None
        x, y = float(self.gaze.pixel[0]), float(self.gaze.pixel[1])
        hits = [obj for obj in self.objects if obj.contains(x, y, self.width, self.height)]
        if not hits:
            return None
        hits.sort(key=lambda obj: obj.area)
        return hits[0]


def camera_points(image_xyz: np.ndarray, world: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """정규화 이미지 좌표와 손 로컬 월드를 화면 픽셀, 공유 카메라 미터 좌표로 변환합니다."""
    pixels = np.column_stack(
        [
            image_xyz[:, 0] * width,
            image_xyz[:, 1] * height,
        ]
    ).astype(np.float64)
    palm_px = float(np.linalg.norm(pixels[MIDDLE_MCP] - pixels[WRIST]))
    palm_px = max(palm_px, 8.0)
    meters_per_pixel = REAL_PALM_M / palm_px

    camera = np.zeros((21, 3), dtype=np.float64)
    camera[:, 0] = (pixels[:, 0] - width * 0.5) * meters_per_pixel
    camera[:, 1] = (height * 0.5 - pixels[:, 1]) * meters_per_pixel
    # 손 전체가 카메라에 가까울수록 손바닥이 크게 보여 깊이가 작아집니다.
    depth = 0.55 * (160.0 / palm_px)
    camera[:, 2] = depth + (world[:, 2] - world[WRIST, 2])
    return camera, pixels


def palm_normal(world: np.ndarray, is_right: bool) -> np.ndarray:
    """손바닥이 카메라를 향할 때 -Z 쪽을 가리키도록 맞춘 법선."""
    wrist = world[WRIST]
    index_mcp = world[INDEX_MCP]
    pinky_mcp = world[PINKY_MCP]
    if is_right:
        normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    else:
        normal = np.cross(pinky_mcp - wrist, index_mcp - wrist)
    return normalize(normal)


def palm_line_angle_deg(pixels: np.ndarray) -> float:
    """손목 → 중지 MCP의 화면 롤. 위쪽이 0°, 시계 방향이 양수."""
    line = pixels[MIDDLE_MCP] - pixels[WRIST]
    if float(line[0] ** 2 + line[1] ** 2) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(line[0], -line[1])))


def handedness_is_right(hand: TrackedHand | None) -> bool:
    return hand is not None and hand.handedness == "Right"
