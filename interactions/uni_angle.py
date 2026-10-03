"""시선으로 고르고, 비우세손 검지 핀치 중 손바닥 롤 각도로 크기를 바꿉니다."""

from __future__ import annotations

from interactions.common import (
    HandMotionMode,
    ImmediateClutch,
    apply_delta,
    apply_ratio,
    normalize_angle_delta,
    speed_level,
    speed_multiplier,
)
from tracking.types import FrameContext, palm_line_angle_deg


class UniAngle:
    def __init__(self) -> None:
        self.dominant_right = True
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.scale_sensitivity = 1.0
        self.rotation_sensitivity = 1.0
        self.hand_motion_mode = HandMotionMode.CLUTCHING
        self.continuous_zoom_speed = 0.75
        self.slow_threshold = 5.0
        self.normal_threshold = 10.0
        self.max_scale_speed = 2.0
        self.slow_speed_multiplier = 0.25
        self.normal_speed_multiplier = 1.0
        self.fast_speed_multiplier = 1.25

        self.selected = None
        self.initial_scale = 1.0
        self.was_pinching = False
        self.initial_angle = 0.0
        self.tilt_ready = False
        self.clutch = ImmediateClutch()
        self.current_angle = 0.0

    def set_clutching_free(self, enabled: bool) -> None:
        self.hand_motion_mode = HandMotionMode.CLUTCHING_FREE if enabled else HandMotionMode.CLUTCHING

    def set_dominant_right(self, dominant_right: bool) -> None:
        self.dominant_right = dominant_right

    def reset(self) -> None:
        self._deselect()
        self.was_pinching = False

    def update(self, ctx: FrameContext) -> None:
        hand = ctx.nondominant(self.dominant_right)
        pinching = hand is not None and hand.tracked and hand.index_pinching
        self.current_angle = palm_line_angle_deg(hand.pixel) if hand is not None and hand.tracked else 0.0
        gazed = ctx.gazed_object()

        if gazed is not None and pinching and not self.was_pinching and self.selected is None:
            self._select(gazed)
        elif not pinching and self.was_pinching and self.selected is not None:
            self._deselect()

        if self.selected is not None:
            if pinching:
                if not self.was_pinching:
                    self._initialize_tilt()
                else:
                    self._update_scale(ctx)
            elif self.was_pinching:
                self._reset_tilt()

        if not pinching and self.was_pinching:
            self.clutch.reset()
        self.was_pinching = pinching

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        return [f"손바닥 각도 {self.current_angle:+.1f}°  선택 {name}"]

    def _select(self, target) -> None:
        self.selected = target
        self.initial_scale = target.scale

    def _initialize_tilt(self) -> None:
        self.initial_angle = self.current_angle
        self.tilt_ready = True
        self.clutch.reset()

    def _reset_tilt(self) -> None:
        self.tilt_ready = False
        self.initial_angle = 0.0

    def _deselect(self) -> None:
        self.selected = None
        self.initial_scale = 1.0
        self._reset_tilt()
        self.clutch.reset()

    def _angle_delta(self) -> float:
        if not self.tilt_ready:
            return 0.0
        return normalize_angle_delta(self.current_angle - self.initial_angle)

    def _update_scale(self, ctx: FrameContext) -> None:
        if not self.tilt_ready or self.selected is None:
            return
        delta = self._angle_delta()
        if self.hand_motion_mode == HandMotionMode.CLUTCHING_FREE:
            direction = self.clutch.step(delta)
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

        ratio = 1.0 + (delta / 90.0) * self.scale_sensitivity * self.rotation_sensitivity
        self.selected.scale = apply_ratio(self.initial_scale, ratio, self.min_scale, self.max_scale)
