"""시선으로 고르고, 비우세손 검지 핀치의 앞뒤 이동으로 크기를 바꿉니다."""

from __future__ import annotations

from interactions.common import (
    HandMotionMode,
    ImmediateClutch,
    apply_delta,
    apply_ratio,
    speed_level,
    speed_multiplier,
)
from tracking.types import INDEX_TIP, FrameContext


class UniDepth:
    def __init__(self) -> None:
        self.dominant_right = True
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.zoom_sensitivity = 1.0
        self.push_pull_sensitivity = 1.0
        self.hand_motion_mode = HandMotionMode.CLUTCHING
        self.continuous_zoom_speed = 0.25
        self.slow_threshold = 0.025
        self.normal_threshold = 0.05
        self.max_scale_speed = 2.0
        self.slow_speed_multiplier = 0.25
        self.normal_speed_multiplier = 1.0
        self.fast_speed_multiplier = 1.25

        self.selected = None
        self.initial_scale = 1.0
        self.was_pinching = False
        self.initial_z = 0.0
        self.ready = False
        self.clutch = ImmediateClutch()
        self.current_z = 0.0

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
        self.current_z = float(hand.camera[INDEX_TIP, 2]) if hand is not None and hand.tracked else 0.0
        gazed = ctx.gazed_object()

        if gazed is not None and pinching and not self.was_pinching and self.selected is None:
            self._select(gazed)
        elif not pinching and self.was_pinching and self.selected is not None:
            self._deselect()

        if self.selected is not None:
            if pinching:
                if not self.was_pinching:
                    self._initialize(hand)
                else:
                    self._update_zoom(ctx)
            elif self.was_pinching:
                self._reset_depth()

        if not pinching and self.was_pinching:
            self.clutch.reset()
        self.was_pinching = pinching

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        delta = self.current_z - self.initial_z if self.ready else 0.0
        return [f"깊이 Δ {delta:+.3f}m  선택 {name}"]

    def _select(self, target) -> None:
        self.selected = target
        self.initial_scale = target.scale

    def _initialize(self, hand) -> None:
        if hand is None or not hand.tracked:
            return
        self.initial_z = float(hand.camera[INDEX_TIP, 2])
        self.ready = True
        self.clutch.reset()

    def _reset_depth(self) -> None:
        self.ready = False
        self.initial_z = 0.0

    def _deselect(self) -> None:
        self.selected = None
        self.initial_scale = 1.0
        self._reset_depth()
        self.clutch.reset()

    def _update_zoom(self, ctx: FrameContext) -> None:
        if not self.ready or self.selected is None:
            return
        z_distance = self.current_z - self.initial_z
        if self.hand_motion_mode == HandMotionMode.CLUTCHING_FREE:
            # 카메라에 가까워지면(Z 감소) 작아지도록 둡니다.
            direction = self.clutch.step(z_distance)
            level = speed_level(abs(z_distance), self.slow_threshold, self.normal_threshold)
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

        zoom = 1.0 + (z_distance * self.push_pull_sensitivity * self.zoom_sensitivity)
        self.selected.scale = apply_ratio(self.initial_scale, zoom, self.min_scale, self.max_scale)
