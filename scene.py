"""화면에 놓인 선택 가능한 카드."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneObject:
    name: str
    nx: float
    ny: float
    nw: float
    nh: float
    color: tuple[int, int, int]
    scale: float = 1.0
    home_nx: float = 0.0
    home_ny: float = 0.0
    home_scale: float = 1.0

    def __post_init__(self) -> None:
        self.home_nx = self.nx
        self.home_ny = self.ny
        self.home_scale = self.scale

    @property
    def area(self) -> float:
        return self.nw * self.nh * self.scale * self.scale

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        center_x = self.nx * width
        center_y = self.ny * height
        half_w = self.nw * width * self.scale * 0.5
        half_h = self.nh * height * self.scale * 0.5
        return abs(x - center_x) <= half_w and abs(y - center_y) <= half_h

    def rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        box_w = self.nw * width * self.scale
        box_h = self.nh * height * self.scale
        left = int(self.nx * width - box_w * 0.5)
        top = int(self.ny * height - box_h * 0.5)
        return left, top, int(box_w), int(box_h)

    def reset_pose(self) -> None:
        self.nx = self.home_nx
        self.ny = self.home_ny
        self.scale = self.home_scale


def create_default_objects() -> list[SceneObject]:
    return [
        SceneObject("빨강", 0.22, 0.30, 0.16, 0.20, (60, 60, 220)),
        SceneObject("초록", 0.50, 0.30, 0.16, 0.20, (70, 170, 70)),
        SceneObject("파랑", 0.78, 0.30, 0.16, 0.20, (220, 120, 40)),
    ]
