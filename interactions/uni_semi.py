"""시선으로 고르고, 비우세손 손바닥이 카메라를 향한 동안 검지–엄지 거리로 크기를 바꿉니다."""

from __future__ import annotations

import numpy as np

from interactions.common import (
    DelayedClutch,
    HandMotionMode,
    apply_delta,
    apply_ratio,
    speed_level,
    speed_multiplier,
)
from tracking.types import FrameContext, palm_normal


class UniSemi:
    def __init__(self) -> None:
        self.dominant_right = True
        self.lock_threshold = 0.55
        self.unlock_threshold = 0.4
        self.x_rotation_adjustment = 20.0
        self.palm_sign = -1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.zoom_sensitivity = 0.75
        self.hand_motion_mode = HandMotionMode.CLUTCHING
        self.continuous_zoom_speed = 0.25
        self.direction_change_threshold = 0.02
        self.direction_stability_time = 0.2
        self.slow_threshold = 0.025
        self.normal_threshold = 0.05
        self.max_scale_speed = 2.0
        self.slow_speed_multiplier = 0.25
        self.normal_speed_multiplier = 1.25
        self.fast_speed_multiplier = 1.25

        self.selected = None
        self.initial_scale = 1.0
        self.locked = False
        self.initial_distance = 0.0
        self.scaling_ready = False
        self.clutch = DelayedClutch()
        self.z_dot = 0.0
        self.current_distance = 0.0

    def set_clutching_free(self, enabled: bool) -> None:
        self.hand_motion_mode = HandMotionMode.CLUTCHING_FREE if enabled else HandMotionMode.CLUTCHING

    def set_dominant_right(self, dominant_right: bool) -> None:
        self.dominant_right = dominant_right

    def toggle_palm_sign(self) -> None:
        self.palm_sign *= -1.0

    def reset(self) -> None:
        self._deselect()
        self.locked = False

    def update(self, ctx: FrameContext) -> None:
        hand = ctx.nondominant(self.dominant_right)
        self.z_dot = self._z_dot(hand)
        self._update_lock(self._palm_in_lock_position(hand))
        gazed = ctx.gazed_object()

        if gazed is not None and self.locked and self.selected is None:
            self._select(hand, gazed)
        elif not self.locked and self.selected is not None:
            self._deselect()

        if self.selected is not None and self.locked:
            self._update_scaling(hand, ctx)

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        lock = "잠금" if self.locked else "해제"
        return [f"손바닥 {lock} Z {self.z_dot:+.2f}  거리 {self.current_distance:.3f}m  선택 {name}"]

    def _z_dot(self, hand) -> float:
        if hand is None or not hand.tracked:
            return 0.0
        normal = palm_normal(hand.world, hand.handedness == "Right")
        angle = np.radians(self.x_rotation_adjustment)
        toward_camera = np.array([0.0, -np.sin(angle), -np.cos(angle)], dtype=np.float64)
        return float(self.palm_sign * np.dot(normal, toward_camera))

    def _palm_in_lock_position(self, hand) -> bool:
        if hand is None or not hand.tracked:
            return False
        threshold = self.unlock_threshold if self.locked else self.lock_threshold
        return self.z_dot > threshold

    def _update_lock(self, palm_in_position: bool) -> None:
        self.locked = palm_in_position

    def _distance(self, hand) -> float:
        if hand is None or not hand.tracked:
            return 0.0
        return hand.index_thumb_distance

    def _select(self, hand, target) -> None:
        self.selected = target
        self.initial_scale = target.scale
        self.initial_distance = self._distance(hand)
        self.scaling_ready = self.initial_distance > 0.0
        self.clutch.reset()

    def _deselect(self) -> None:
        self.selected = None
        self.initial_scale = 1.0
        self.initial_distance = 0.0
        self.scaling_ready = False
        self.clutch.reset()

    def _update_scaling(self, hand, ctx: FrameContext) -> None:
        if not self.scaling_ready or self.selected is None:
            return
        current = self._distance(hand)
        self.current_distance = current
        if current <= 0.0:
            return

        if self.hand_motion_mode == HandMotionMode.CLUTCHING_FREE:
            delta = current - self.initial_distance
            direction = self.clutch.step(
                delta,
                ctx.time,
                self.direction_change_threshold,
                self.direction_stability_time,
            )
            level = speed_level(abs(delta), self.slow_threshold, self.normal_threshold)
            mult = speed_multiplier(
                level,
                self.slow_speed_multiplier,
                self.normal_speed_multiplier,
                self.fast_speed_multiplier,
                self.max_scale_speed,
            )
            scale_delta = direction * self.continuous_zoom_speed * mult * ctx.dt
            self.selected.scale = apply_delta(self.selected.scale, scale_delta, self.min_scale, self.max_scale)
            return

        ratio = current / self.initial_distance
        zoom = 1.0 + (ratio - 1.0) * self.zoom_sensitivity
        self.selected.scale = apply_ratio(self.initial_scale, zoom, self.min_scale, self.max_scale)
