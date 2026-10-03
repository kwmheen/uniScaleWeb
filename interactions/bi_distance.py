"""시선으로 고르고, 양손 검지 사이 거리로 크기를 바꿉니다."""

from __future__ import annotations

import numpy as np

from interactions.common import (
    HandMotionMode,
    ImmediateClutch,
    apply_delta,
    apply_ratio,
    speed_level,
    speed_multiplier,
)
from tracking.types import INDEX_TIP, FrameContext


class BiDistance:
    def __init__(self) -> None:
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.zoom_sensitivity = 1.0
        self.pinch_stability_time = 0.1
        self.min_hand_separation = 0.02
        self.hand_motion_mode = HandMotionMode.CLUTCHING
        self.continuous_zoom_speed = 0.25
        self.slow_threshold = 0.05
        self.normal_threshold = 0.1
        self.max_scale_speed = 2.0
        self.slow_speed_multiplier = 0.25
        self.normal_speed_multiplier = 1.0
        self.fast_speed_multiplier = 1.25

        self.selected = None
        self.initial_distance = 0.0
        self.initial_scale = 1.0
        self.was_left_pinching = False
        self.was_right_pinching = False
        self.pinch_start_time = 0.0
        self.pinch_stable = False
        self.clutch = ImmediateClutch()
        self.current_distance = 0.0

    def set_clutching_free(self, enabled: bool) -> None:
        self.hand_motion_mode = HandMotionMode.CLUTCHING_FREE if enabled else HandMotionMode.CLUTCHING

    def reset(self) -> None:
        self._deselect()

    def update(self, ctx: FrameContext) -> None:
        left_pinching = ctx.left is not None and ctx.left.tracked and ctx.left.index_pinching
        right_pinching = ctx.right is not None and ctx.right.tracked and ctx.right.index_pinching
        both = left_pinching and right_pinching
        only_one = left_pinching != right_pinching

        if only_one:
            if self.selected is not None:
                self._deselect()
            self.was_left_pinching = left_pinching
            self.was_right_pinching = right_pinching
            return

        self._check_stability(ctx, both, left_pinching, right_pinching)
        gazed = ctx.gazed_object()

        if gazed is not None and both and self.pinch_stable and self.selected is None:
            self._select(ctx, gazed)

        if self.selected is not None and both and self.pinch_stable:
            self._update_zoom(ctx)
        elif not both and self.selected is not None:
            self._deselect()

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        return [
            f"양손 거리 {self.current_distance:.3f}m  선택 {name}",
        ]

    def _distance(self, ctx: FrameContext) -> float:
        if ctx.left is None or ctx.right is None:
            return 0.0
        if not ctx.left.tracked or not ctx.right.tracked:
            return 0.0
        distance = float(np.linalg.norm(ctx.left.camera[INDEX_TIP] - ctx.right.camera[INDEX_TIP]))
        return max(distance, self.min_hand_separation)

    def _check_stability(self, ctx: FrameContext, both: bool, left_pinching: bool, right_pinching: bool) -> None:
        started = both and (not self.was_left_pinching or not self.was_right_pinching)
        if started:
            self.pinch_start_time = ctx.time
            self.pinch_stable = False
            current = self._distance(ctx)
            if current > 0.0:
                self.initial_distance = current
                self.clutch.reset()

        if both and not self.pinch_stable and ctx.time - self.pinch_start_time >= self.pinch_stability_time:
            self.pinch_stable = True

        if not both:
            self.pinch_stable = False
            self.clutch.reset()

        self.was_left_pinching = left_pinching
        self.was_right_pinching = right_pinching

    def _select(self, ctx: FrameContext, target) -> None:
        self.selected = target
        self.initial_distance = self._distance(ctx)
        self.initial_scale = target.scale
        self.clutch.reset()

    def _deselect(self) -> None:
        self.selected = None
        self.initial_distance = 0.0
        self.initial_scale = 1.0
        self.pinch_stable = False
        self.clutch.reset()

    def _update_zoom(self, ctx: FrameContext) -> None:
        if self.selected is None:
            return
        current = self._distance(ctx)
        self.current_distance = current
        if self.initial_distance <= 0.0 or current <= 0.0:
            return

        if self.hand_motion_mode == HandMotionMode.CLUTCHING_FREE:
            delta = current - self.initial_distance
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

        ratio = current / self.initial_distance
        zoom = float(np.power(ratio, self.zoom_sensitivity))
        self.selected.scale = apply_ratio(self.initial_scale, zoom, self.min_scale, self.max_scale)
