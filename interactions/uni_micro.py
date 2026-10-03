"""시선으로 고르고, 비우세손의 미세 탭 각도로 잡은 뒤 엄지 이동으로 크기를 바꿉니다."""

from __future__ import annotations

from collections import deque

import numpy as np

from interactions.common import HandMotionMode, apply_delta, apply_ratio, calculate_angle_between_plane_and_line
from tracking.types import INDEX_MCP, INDEX_TIP, THUMB_MCP, THUMB_TIP, FrameContext, palm_normal


class UniMicro:
    def __init__(self) -> None:
        self.dominant_right = True
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.zoom_sensitivity = 2.0
        self.hand_motion_mode = HandMotionMode.CLUTCHING
        self.continuous_zoom_speed = 0.25
        self.tap_on_angle = 30.0
        self.tap_off_angle = 80.0
        self.position_holding_time = 0.5
        self.position_change_threshold = 0.1
        self.position_holding_ignore = 1.0
        self.use_moving_average = True
        self.moving_average_window = 10

        self.selected = None
        self.initial_scale = 1.0
        self.was_tabbing = False
        self.tab_angle_state = False
        self.tab_start_time = 0.0
        self.ready = False
        self.initial_thumb_knuckle = 0.0
        self.initial_thumb_tip = 0.0
        self.last_scale_change_time = 0.0
        self.last_scale = 0.0
        self.zoom_queue: deque[float] = deque()
        self.combined_angle = 180.0
        self.zoom_factor = 0.0

    def set_clutching_free(self, enabled: bool) -> None:
        self.hand_motion_mode = HandMotionMode.CLUTCHING_FREE if enabled else HandMotionMode.CLUTCHING

    def set_dominant_right(self, dominant_right: bool) -> None:
        self.dominant_right = dominant_right

    def reset(self) -> None:
        self._deselect()
        self.was_tabbing = False

    def update(self, ctx: FrameContext) -> None:
        hand = ctx.nondominant(self.dominant_right)
        self.combined_angle = self._combined_angle(hand)
        tabbing = self._tabbing(hand)

        if tabbing and not self.was_tabbing:
            self.tab_start_time = ctx.time

        gazed = ctx.gazed_object()
        if gazed is not None and tabbing and self.selected is None:
            self._select(gazed, ctx.time)
        elif self._should_terminate(ctx, tabbing) and self.selected is not None:
            self._deselect()

        if self.selected is not None:
            if tabbing:
                if not self.was_tabbing:
                    self._initialize(hand)
                else:
                    self._update_zoom(ctx, hand)
                    self._track_scale(ctx)
            elif self._should_terminate(ctx, tabbing):
                self._reset_gesture()

        self.was_tabbing = tabbing

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        tab = "탭" if self.tab_angle_state else "해제"
        return [f"마이크로 {tab} {self.combined_angle:.1f}°  줌 {self.zoom_factor:+.3f}  선택 {name}"]

    def _to_local(self, hand):
        origin = hand.world[0]
        across = hand.world[INDEX_MCP] - hand.world[17]
        up = palm_normal(hand.world, hand.handedness == "Right")
        x_axis = across / max(float(np.linalg.norm(across)), 1e-8)
        z_axis = up
        y_axis = np.cross(z_axis, x_axis)
        y_length = float(np.linalg.norm(y_axis))
        if y_length < 1e-8:
            return lambda point: point - origin
        y_axis = y_axis / y_length
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-8)
        rotation = np.stack([x_axis, y_axis, z_axis], axis=0)

        def convert(point: np.ndarray) -> np.ndarray:
            return rotation @ (point - origin)

        return convert

    def _combined_angle(self, hand) -> float:
        if hand is None or not hand.tracked:
            return 180.0
        convert = self._to_local(hand)
        thumb_mcp = convert(hand.world[THUMB_MCP])
        index_mcp = convert(hand.world[INDEX_MCP])
        index_tip = convert(hand.world[INDEX_TIP])
        thumb_tip = convert(hand.world[THUMB_TIP])
        angle_xz = calculate_angle_between_plane_and_line(
            thumb_mcp, index_mcp, index_tip, thumb_mcp, thumb_tip, "XZ"
        )
        angle_yz = calculate_angle_between_plane_and_line(
            thumb_mcp, index_mcp, index_tip, thumb_mcp, thumb_tip, "YZ"
        )
        return float(np.sqrt(angle_xz * angle_xz + angle_yz * angle_yz))

    def _tabbing(self, hand) -> bool:
        if hand is None or not hand.tracked:
            self.tab_angle_state = False
            return False
        if not self.tab_angle_state:
            if self.combined_angle < self.tap_on_angle:
                self.tab_angle_state = True
        elif self.combined_angle >= self.tap_off_angle:
            self.tab_angle_state = False
        return self.tab_angle_state

    def _should_terminate(self, ctx: FrameContext, tabbing: bool) -> bool:
        if not tabbing and self.was_tabbing:
            return True
        if ctx.time - self.tab_start_time < self.position_holding_ignore:
            return False
        if self.selected is None:
            return False
        return ctx.time - self.last_scale_change_time >= self.position_holding_time

    def _track_scale(self, ctx: FrameContext) -> None:
        if self.selected is None:
            return
        if self.last_scale == 0.0:
            self.last_scale = self.selected.scale
            self.last_scale_change_time = ctx.time
            return
        if abs(self.selected.scale - self.last_scale) > self.position_change_threshold:
            self.last_scale_change_time = ctx.time
            self.last_scale = self.selected.scale

    def _select(self, target, now: float) -> None:
        self._reset_gesture()
        self.selected = target
        self.initial_scale = target.scale
        self.last_scale = target.scale
        self.last_scale_change_time = now

    def _initialize(self, hand) -> None:
        if hand is None or not hand.tracked:
            return
        thumb = self._horizontal(hand.world[THUMB_TIP])
        knuckle = self._horizontal(hand.world[INDEX_MCP])
        tip = self._horizontal(hand.world[INDEX_TIP])
        self.initial_thumb_knuckle = float(np.linalg.norm(thumb - knuckle))
        self.initial_thumb_tip = float(np.linalg.norm(thumb - tip))
        self.ready = True

    def _reset_gesture(self) -> None:
        self.ready = False
        self.initial_thumb_knuckle = 0.0
        self.initial_thumb_tip = 0.0
        self.zoom_queue.clear()
        self.tab_angle_state = False
        self.zoom_factor = 0.0

    def _deselect(self) -> None:
        self.selected = None
        self.initial_scale = 1.0
        self.tab_start_time = 0.0
        self.last_scale = 0.0
        self.last_scale_change_time = 0.0
        self._reset_gesture()

    @staticmethod
    def _horizontal(point: np.ndarray) -> np.ndarray:
        return np.array([point[0], 0.0, point[2]], dtype=np.float64)

    def _raw_zoom(self, hand) -> float:
        if hand is None or not hand.tracked:
            return 0.0
        thumb = self._horizontal(hand.world[THUMB_TIP])
        knuckle = self._horizontal(hand.world[INDEX_MCP])
        tip = self._horizontal(hand.world[INDEX_TIP])
        thumb_knuckle = float(np.linalg.norm(thumb - knuckle))
        thumb_tip = float(np.linalg.norm(thumb - tip))
        knuckle_delta = thumb_knuckle - self.initial_thumb_knuckle
        tip_delta = thumb_tip - self.initial_thumb_tip
        zoom = -tip_delta + knuckle_delta
        average = (self.initial_thumb_knuckle + self.initial_thumb_tip) * 0.5
        if average > 0.0:
            zoom /= average
        return float(zoom)

    def _filtered_zoom(self, raw: float) -> float:
        if not self.use_moving_average:
            return raw
        self.zoom_queue.append(raw)
        while len(self.zoom_queue) > self.moving_average_window:
            self.zoom_queue.popleft()
        return float(sum(self.zoom_queue) / len(self.zoom_queue))

    def _update_zoom(self, ctx: FrameContext, hand) -> None:
        if not self.ready or self.selected is None:
            return
        self.zoom_factor = self._filtered_zoom(self._raw_zoom(hand))
        if abs(self.zoom_factor) <= 0.001:
            return
        if self.hand_motion_mode == HandMotionMode.CLUTCHING_FREE:
            scale_delta = self.zoom_factor * self.continuous_zoom_speed * ctx.dt
            self.selected.scale = apply_delta(self.selected.scale, scale_delta, self.min_scale, self.max_scale)
            return
        multiplier = 1.0 + self.zoom_factor * self.zoom_sensitivity
        self.selected.scale = apply_ratio(self.initial_scale, multiplier, self.min_scale, self.max_scale)
