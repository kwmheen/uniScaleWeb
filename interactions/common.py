"""스케일 인터랙션이 공유하는 모드와 clutching 상태."""

from __future__ import annotations

from enum import Enum

import numpy as np


class HandMotionMode(Enum):
    CLUTCHING = "Clutching"
    CLUTCHING_FREE = "ClutchingFree"


class ScaleSpeedLevel(Enum):
    SLOW = 0
    NORMAL = 1
    FAST = 2


class ScaleMethod(Enum):
    UNI_DEPTH = "uniDepth"
    UNI_ANGLE = "uniAngle"
    UNI_MICRO = "uniMicro"
    UNI_SEMI = "uniSemi"
    BI_SEMI = "biSemi"
    BI_DISTANCE = "biDistance"


METHOD_ORDER = (
    ScaleMethod.UNI_DEPTH,
    ScaleMethod.UNI_ANGLE,
    ScaleMethod.UNI_MICRO,
    ScaleMethod.UNI_SEMI,
    ScaleMethod.BI_SEMI,
    ScaleMethod.BI_DISTANCE,
)

METHOD_KEYS = {
    ord("1"): ScaleMethod.UNI_DEPTH,
    ord("2"): ScaleMethod.UNI_ANGLE,
    ord("3"): ScaleMethod.UNI_MICRO,
    ord("4"): ScaleMethod.UNI_SEMI,
    ord("5"): ScaleMethod.BI_SEMI,
    ord("6"): ScaleMethod.BI_DISTANCE,
}


def speed_level(abs_value: float, slow: float, normal: float) -> ScaleSpeedLevel:
    if abs_value <= slow:
        return ScaleSpeedLevel.SLOW
    if abs_value <= normal:
        return ScaleSpeedLevel.NORMAL
    return ScaleSpeedLevel.FAST


def speed_multiplier(
    level: ScaleSpeedLevel,
    slow: float,
    normal: float,
    fast: float,
    cap: float,
) -> float:
    base = {
        ScaleSpeedLevel.SLOW: slow,
        ScaleSpeedLevel.FAST: fast,
    }.get(level, normal)
    return min(base, cap)


def normalize_angle_delta(angle_delta: float) -> float:
    while angle_delta > 180.0:
        angle_delta -= 360.0
    while angle_delta < -180.0:
        angle_delta += 360.0
    return angle_delta


class ImmediateClutch:
    """기준점을 넘는 즉시 스케일 방향을 뒤집는 ClutchingFree."""

    def __init__(self) -> None:
        self.direction = 1.0
        self.ready = False

    def reset(self) -> None:
        self.direction = 1.0
        self.ready = False

    def step(self, delta: float) -> float:
        if not self.ready:
            self.direction = 1.0 if delta > 0.0 else -1.0
            self.ready = True
        crossed = (self.direction > 0.0 and delta < 0.0) or (self.direction < 0.0 and delta > 0.0)
        if crossed:
            self.direction *= -1.0
        return self.direction


class DelayedClutch:
    """기준점을 일정 시간 넘긴 뒤에만 방향을 뒤집는 ClutchingFree."""

    def __init__(self) -> None:
        self.direction = 1.0
        self.ready = False
        self.changing = False
        self.change_time = 0.0

    def reset(self) -> None:
        self.direction = 1.0
        self.ready = False
        self.changing = False
        self.change_time = 0.0

    def step(self, delta: float, now: float, threshold: float, stability: float) -> float:
        if not self.ready:
            self.direction = 1.0 if delta > 0.0 else -1.0
            self.ready = True

        past = (-delta >= threshold) if self.direction > 0.0 else (delta >= threshold)
        if past:
            if not self.changing:
                self.changing = True
                self.change_time = now
            elif now - self.change_time >= stability:
                self.direction *= -1.0
                self.changing = False
        else:
            self.changing = False
        return self.direction


def clamp_scale(scale: float, minimum: float, maximum: float) -> float:
    return float(np.clip(scale, minimum, maximum))


def apply_ratio(initial: float, ratio: float, minimum: float, maximum: float) -> float:
    return clamp_scale(initial * ratio, minimum, maximum)


def apply_delta(current: float, delta: float, minimum: float, maximum: float) -> float:
    return clamp_scale(current * (1.0 + delta), minimum, maximum)


def calculate_angle_between_plane_and_line(
    plane_point1: np.ndarray,
    plane_point2: np.ndarray,
    plane_point3: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    plane: str,
) -> float:
    """Unity uniMicro.CalculateAngleBetweenPlaneAndLine과 같은 투영 각도."""
    if plane == "XZ":
        vec1 = np.array([plane_point2[0] - plane_point1[0], plane_point2[2] - plane_point1[2]])
        vec2 = np.array([plane_point3[0] - plane_point1[0], plane_point3[2] - plane_point1[2]])
        line = np.array([line_end[0] - line_start[0], line_end[2] - line_start[2]])
    elif plane == "YZ":
        vec1 = np.array([plane_point2[1] - plane_point1[1], plane_point2[2] - plane_point1[2]])
        vec2 = np.array([plane_point3[1] - plane_point1[1], plane_point3[2] - plane_point1[2]])
        line = np.array([line_end[1] - line_start[1], line_end[2] - line_start[2]])
    else:
        vec1 = np.array([plane_point2[0] - plane_point1[0], plane_point2[1] - plane_point1[1]])
        vec2 = np.array([plane_point3[0] - plane_point1[0], plane_point3[1] - plane_point1[1]])
        line = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])

    if np.linalg.norm(vec1) < 1e-3 or np.linalg.norm(vec2) < 1e-3 or np.linalg.norm(line) < 1e-3:
        return 0.0

    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    line = line / np.linalg.norm(line)
    plane_direction = vec1 + vec2
    plane_norm = np.linalg.norm(plane_direction)
    if plane_norm < 1e-8:
        return 0.0
    plane_direction = plane_direction / plane_norm
    dot = float(np.clip(np.dot(plane_direction, line), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))
