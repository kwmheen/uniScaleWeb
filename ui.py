"""카메라 화면 위에 카드, 손, 시선, 상태 글을 그립니다."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tracking.types import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_FONT: ImageFont.ImageFont | None = None
_FONT_PATHS = (
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    global _FONT
    if _FONT is not None and getattr(_FONT, "size", size) == size:
        return _FONT
    for path in _FONT_PATHS:
        try:
            _FONT = ImageFont.truetype(path, size)
            return _FONT
        except OSError:
            continue
    _FONT = ImageFont.load_default()
    return _FONT


def draw_scene(frame: np.ndarray, ctx, controller, lines: list[str], fps: float, gaze_gain: float) -> np.ndarray:
    canvas = frame.copy()
    _draw_objects(canvas, ctx, controller)
    _draw_hands(canvas, ctx, controller)
    _draw_gaze(canvas, ctx)
    return _draw_text(canvas, ctx, lines, fps, gaze_gain)


def _draw_objects(canvas: np.ndarray, ctx, controller) -> None:
    scale_selected = controller.scale_selected()
    move_selected = controller.move_selected()
    for obj in ctx.objects:
        left, top, width, height = obj.rect(ctx.width, ctx.height)
        right = left + width
        bottom = top + height
        overlay = canvas.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), obj.color, -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        border = (255, 255, 255)
        thickness = 2
        if obj is scale_selected and obj is move_selected:
            border = (255, 255, 0)
            thickness = 4
        elif obj is scale_selected:
            border = (255, 255, 255)
            thickness = 3
        elif obj is move_selected:
            border = (0, 220, 255)
            thickness = 3
        cv2.rectangle(canvas, (left, top), (right, bottom), border, thickness)


def _draw_hands(canvas: np.ndarray, ctx, controller) -> None:
    for hand, color in ((ctx.left, (255, 180, 40)), (ctx.right, (40, 140, 255))):
        if hand is None or not hand.tracked:
            continue
        draw_color = (80, 220, 80) if hand.index_pinching or hand.middle_pinching else color
        for start, end in HAND_CONNECTIONS:
            p1 = _point(hand.pixel[start])
            p2 = _point(hand.pixel[end])
            cv2.line(canvas, p1, p2, draw_color, 2, cv2.LINE_AA)
        for landmark in hand.pixel:
            cv2.circle(canvas, _point(landmark), 3, draw_color, -1, cv2.LINE_AA)

    if ctx.left is not None and ctx.right is not None and controller.method.value == "biDistance":
        cv2.line(
            canvas,
            _point(ctx.left.pixel[INDEX_TIP]),
            _point(ctx.right.pixel[INDEX_TIP]),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    active = ctx.left if controller.dominant_right else ctx.right
    if active is not None and controller.method.value in {"uniAngle", "uniSemi"}:
        cv2.line(
            canvas,
            _point(active.pixel[WRIST]),
            _point(active.pixel[MIDDLE_MCP]),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            _point(active.pixel[THUMB_TIP]),
            _point(active.pixel[INDEX_TIP]),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _draw_gaze(canvas: np.ndarray, ctx) -> None:
    if not ctx.gaze.tracked:
        return
    center = _point(ctx.gaze.pixel)
    hit = ctx.gazed_object() is not None
    color = (0, 255, 0) if hit else (0, 255, 255)
    cv2.circle(canvas, center, 10, color, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def _draw_text(canvas: np.ndarray, ctx, lines: list[str], fps: float, gaze_gain: float) -> np.ndarray:
    help_line = "1-6 모드   F clutching   H 우세손   C 시선보정   P 손바닥반전   R 리셋   [ ] 시선감도   Q 종료"
    all_lines = [f"{fps:4.1f} FPS    시선감도 {gaze_gain:.1f}", *lines, help_line]

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = _font(18)
    line_height = 24
    panel_top = canvas.shape[0] - (len(all_lines) * line_height + 16)
    draw.rectangle((8, panel_top, canvas.shape[1] - 8, canvas.shape[0] - 8), fill=(0, 0, 0, 150))

    for obj in ctx.objects:
        left, top, _width, _height = obj.rect(canvas.shape[1], canvas.shape[0])
        draw.text((left + 8, top + 8), f"{obj.name}  {obj.scale:.2f}", font=font, fill=(255, 255, 255, 255))

    y = panel_top + 8
    for line in all_lines:
        draw.text((16, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _point(landmark: np.ndarray) -> tuple[int, int]:
    return int(landmark[0]), int(landmark[1])
