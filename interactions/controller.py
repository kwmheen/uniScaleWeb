"""활성 스케일 방식과 시선 핀치 이동을 함께 갱신합니다."""

from __future__ import annotations

from interactions.bi_distance import BiDistance
from interactions.bi_semi import BiSemi
from interactions.common import METHOD_ORDER, HandMotionMode, ScaleMethod
from interactions.dh_gaze_pinch import DhGazePinch
from interactions.uni_angle import UniAngle
from interactions.uni_depth import UniDepth
from interactions.uni_micro import UniMicro
from interactions.uni_semi import UniSemi
from tracking.types import FrameContext


class InteractionController:
    def __init__(self) -> None:
        self.method = ScaleMethod.UNI_DEPTH
        self.clutching_free = False
        self.dominant_right = True
        self.bi_distance = BiDistance()
        self.bi_semi = BiSemi()
        self.uni_angle = UniAngle()
        self.uni_depth = UniDepth()
        self.uni_micro = UniMicro()
        self.uni_semi = UniSemi()
        self.move = DhGazePinch()
        self._scales = {
            ScaleMethod.UNI_DEPTH: self.uni_depth,
            ScaleMethod.UNI_ANGLE: self.uni_angle,
            ScaleMethod.UNI_MICRO: self.uni_micro,
            ScaleMethod.UNI_SEMI: self.uni_semi,
            ScaleMethod.BI_SEMI: self.bi_semi,
            ScaleMethod.BI_DISTANCE: self.bi_distance,
        }
        self._apply_settings()

    @property
    def active(self):
        return self._scales[self.method]

    def set_method(self, method: ScaleMethod) -> None:
        if method == self.method:
            return
        self.active.reset()
        self.method = method
        self._apply_settings()

    def toggle_clutching(self) -> None:
        self.clutching_free = not self.clutching_free
        self._apply_settings()

    def toggle_dominant(self) -> None:
        self.dominant_right = not self.dominant_right
        self._apply_settings()

    def update(self, ctx: FrameContext) -> None:
        self.move.active_scale_method = self.method
        self.move.update(ctx)
        self.active.update(ctx)

    def status(self) -> list[str]:
        clutch = "ClutchingFree" if self.clutching_free else "Clutching"
        dominant = "오른손" if self.dominant_right else "왼손"
        lines = [
            f"{self.method.value}  |  {clutch}  |  우세손 {dominant}",
            *self.move.status(),
            *self.active.status(),
        ]
        return lines

    def scale_selected(self):
        return self.active.selected

    def move_selected(self):
        return self.move.selected

    def _apply_settings(self) -> None:
        for interaction in self._scales.values():
            interaction.set_clutching_free(self.clutching_free)
            if hasattr(interaction, "set_dominant_right"):
                interaction.set_dominant_right(self.dominant_right)
        self.move.set_dominant_right(self.dominant_right)
        self.move.active_scale_method = self.method

    def reset_interactions(self) -> None:
        for interaction in self._scales.values():
            interaction.reset()
        self.move.reset()

    def cycle_method(self, step: int) -> None:
        index = METHOD_ORDER.index(self.method)
        self.set_method(METHOD_ORDER[(index + step) % len(METHOD_ORDER)])

    @property
    def motion_mode(self) -> HandMotionMode:
        return HandMotionMode.CLUTCHING_FREE if self.clutching_free else HandMotionMode.CLUTCHING
