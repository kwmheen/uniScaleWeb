"""시선으로 고른 뒤 우세손 검지 핀치로 잡고, 손 이동으로 위치를 옮깁니다."""

from __future__ import annotations

import numpy as np

from interactions.common import ScaleMethod
from tracking.types import FrameContext


class DhGazePinch:
    def __init__(self) -> None:
        self.dominant_right = True
        self.movement_sensitivity = 1.5
        self.active_scale_method = ScaleMethod.UNI_DEPTH

        self.selected = None
        self.was_pinching = False
        self.hand_anchor = np.zeros(2, dtype=np.float64)
        self.object_anchor = (0.5, 0.5)
        self.yielded = False

    def set_dominant_right(self, dominant_right: bool) -> None:
        self.dominant_right = dominant_right

    def reset(self) -> None:
        self._deselect()
        self.was_pinching = False

    def update(self, ctx: FrameContext) -> None:
        if self._should_yield(ctx):
            self.yielded = True
            self.was_pinching = self._pinching(ctx)
            return

        self.yielded = False
        pinching = self._pinching(ctx)
        if pinching and not self.was_pinching:
            self._on_pinch_start(ctx)
        elif not pinching and self.was_pinching:
            self._deselect()
        self.was_pinching = pinching

        if self.selected is not None:
            self._move(ctx)

    def status(self) -> list[str]:
        name = self.selected.name if self.selected is not None else "-"
        state = "일시정지" if self.yielded else ("이동" if self.selected is not None else "대기")
        return [f"시선 핀치 {state}  선택 {name}"]

    def _should_yield(self, ctx: FrameContext) -> bool:
        if self.active_scale_method != ScaleMethod.BI_DISTANCE:
            return False
        left = ctx.left is not None and ctx.left.tracked and ctx.left.index_pinching
        right = ctx.right is not None and ctx.right.tracked and ctx.right.index_pinching
        return left and right

    def _pinching(self, ctx: FrameContext) -> bool:
        hand = ctx.dominant(self.dominant_right)
        return hand is not None and hand.tracked and hand.index_pinching

    def _hand_pixel(self, ctx: FrameContext) -> np.ndarray | None:
        hand = ctx.dominant(self.dominant_right)
        if hand is None or not hand.tracked:
            return None
        return hand.wrist_pixel

    def _on_pinch_start(self, ctx: FrameContext) -> None:
        gazed = ctx.gazed_object()
        pixel = self._hand_pixel(ctx)
        if gazed is None or pixel is None:
            return
        self.selected = gazed
        self.hand_anchor = pixel.copy()
        self.object_anchor = (gazed.nx, gazed.ny)

    def _deselect(self) -> None:
        self.selected = None
        self.hand_anchor = np.zeros(2, dtype=np.float64)
        self.object_anchor = (0.5, 0.5)

    def _move(self, ctx: FrameContext) -> None:
        if self.selected is None or ctx.width <= 0 or ctx.height <= 0:
            return
        pixel = self._hand_pixel(ctx)
        if pixel is None:
            return
        delta = (pixel - self.hand_anchor) * self.movement_sensitivity
        self.selected.nx = float(np.clip(self.object_anchor[0] + delta[0] / ctx.width, 0.08, 0.92))
        self.selected.ny = float(np.clip(self.object_anchor[1] + delta[1] / ctx.height, 0.1, 0.9))
