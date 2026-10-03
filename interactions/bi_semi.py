"""시선으로 고르고, 우세손 중지–엄지 세미핀치 동안 비우세손 검지–엄지 거리로 크기를 바꿉니다."""

from __future__ import annotations

from interactions.common import (
    DelayedClutch,
    HandMotionMode,
    apply_delta,
    apply_ratio,
    speed_level,
    speed_multiplier,
)
from tracking.types import FrameContext


class BiSemi:
    def __init__(self) -> None:
        self.dominant_right = True
        self.middle_thumb_threshold = 0.04
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
        self.was_dh_pinching = False
        self.initial_distance = 0.0
        self.scaling_ready = False
        self.clutch = DelayedClutch()
        self.dh_distance = 0.0
        self.ndh_distance = 0.0

    def set_clutching_free(self, enabled: bool) -> None:
        self.hand_motion_mode = HandMotionMode.CLUTCHING_FREE if enabled else HandMotionMode.CLUTCHING

    def set_dominant_right(self, dominant_right: bool) -> None:
        self.dominant_right = dominant_right

    def reset(self) -> None:
        self._deselect()
        self.was_dh_pinching = False

    def update(self, ctx: FrameContext) -> None:
        dh = ctx.dominant(self.dominant_right)
        dh_pinching = self._dh_pinching(dh)
        self.dh_distance = 0.0 if dh is None else dh.middle_thumb_distance
        gazed = ctx.gazed_object()

        if gazed is not None and dh_pinching and not self.was_dh_pinching and self.selected is None:
            self._select(ctx, gazed)
        elif not dh_pinching and self.was_dh_pinching and self.selected is not None:
            self._deselect()

        if self.selected is not None and dh_pinching:
            self._update_scaling(ctx)

        self.was_dh_pinching = dh_pinching

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        pinch = "세미핀치" if self.was_dh_pinching else "대기"
        return [
            f"DH {pinch} {self.dh_distance:.3f}m  NDH {self.ndh_distance:.3f}m  선택 {name}",
        ]

    def _dh_pinching(self, hand) -> bool:
        if hand is None or not hand.tracked:
            return False
        if hand.middle_pinching:
            return True
        return hand.middle_thumb_distance <= self.middle_thumb_threshold

    def _ndh_distance(self, ctx: FrameContext) -> float:
        hand = ctx.nondominant(self.dominant_right)
        if hand is None or not hand.tracked:
            return 0.0
        return hand.index_thumb_distance

    def _select(self, ctx: FrameContext, target) -> None:
        self.selected = target
        self.initial_scale = target.scale
        self.initial_distance = self._ndh_distance(ctx)
        self.scaling_ready = self.initial_distance > 0.0
        self.clutch.reset()

    def _deselect(self) -> None:
        self.selected = None
        self.initial_scale = 1.0
        self.initial_distance = 0.0
        self.scaling_ready = False
        self.clutch.reset()

    def _update_scaling(self, ctx: FrameContext) -> None:
        if not self.scaling_ready or self.selected is None:
            return
        current = self._ndh_distance(ctx)
        self.ndh_distance = current
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
