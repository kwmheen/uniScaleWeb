"""카메라 없이 인터랙션 상태 기계를 확인합니다."""

from __future__ import annotations

import unittest

import numpy as np

from interactions.bi_distance import BiDistance
from interactions.bi_semi import BiSemi
from interactions.common import ScaleMethod, calculate_angle_between_plane_and_line
from interactions.controller import InteractionController
from interactions.dh_gaze_pinch import DhGazePinch
from interactions.uni_angle import UniAngle
from interactions.uni_depth import UniDepth
from interactions.uni_micro import UniMicro
from interactions.uni_semi import UniSemi
from scene import SceneObject
from tracking.types import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP, THUMB_MCP, THUMB_TIP, WRIST, GazeSample, TrackedHand, FrameContext


def hand(name: str, index_pinch: bool = False, middle_pinch: bool = False) -> TrackedHand:
    return TrackedHand(
        handedness=name,
        world=np.zeros((21, 3), dtype=np.float64),
        camera=np.zeros((21, 3), dtype=np.float64),
        pixel=np.zeros((21, 2), dtype=np.float64),
        index_pinching=index_pinch,
        middle_pinching=middle_pinch,
    )


def box() -> SceneObject:
    return SceneObject("상자", 0.5, 0.5, 0.2, 0.2, (0, 0, 255))


def context(obj, gaze, left=None, right=None, time=0.0, dt=0.016) -> FrameContext:
    return FrameContext(
        time=time,
        dt=dt,
        width=1000,
        height=1000,
        left=left,
        right=right,
        gaze=GazeSample(True, np.array(gaze, dtype=np.float64)),
        objects=[obj],
    )


class InteractionTests(unittest.TestCase):
    def test_bi_distance_clutching_uses_index_separation_ratio(self):
        interaction = BiDistance()
        obj = box()
        left = hand("Left", index_pinch=True)
        right = hand("Right", index_pinch=True)
        left.camera[INDEX_TIP] = (0.0, 0.0, 0.0)
        right.camera[INDEX_TIP] = (0.2, 0.0, 0.0)

        interaction.update(context(obj, (500, 500), left, right, time=0.0))
        self.assertIsNone(interaction.selected)

        interaction.update(context(obj, (500, 500), left, right, time=0.1))
        self.assertIs(interaction.selected, obj)
        self.assertAlmostEqual(obj.scale, 1.0, places=4)

        right.camera[INDEX_TIP] = (0.4, 0.0, 0.0)
        interaction.update(context(obj, (500, 500), left, right, time=0.2))
        self.assertAlmostEqual(obj.scale, 2.0, places=4)

    def test_bi_distance_single_pinch_does_not_select(self):
        interaction = BiDistance()
        obj = box()
        left = hand("Left", index_pinch=True)
        interaction.update(context(obj, (500, 500), left, hand("Right"), time=1.0))
        self.assertIsNone(interaction.selected)

    def test_bi_semi_scales_from_nondominant_pinch_distance(self):
        interaction = BiSemi()
        obj = box()
        dominant = hand("Right", middle_pinch=True)
        other = hand("Left")
        other.world[THUMB_TIP] = (0.0, 0.0, 0.0)
        other.world[INDEX_TIP] = (0.05, 0.0, 0.0)

        interaction.update(context(obj, (500, 500), other, dominant, time=0.0))
        self.assertIs(interaction.selected, obj)

        other.world[INDEX_TIP] = (0.10, 0.0, 0.0)
        interaction.update(context(obj, (500, 500), other, dominant, time=0.1))
        self.assertAlmostEqual(obj.scale, 1.75, places=4)

    def test_gaze_pinch_moves_from_wrist_and_yields_to_both_hands(self):
        move = DhGazePinch()
        obj = box()
        dominant = hand("Right", index_pinch=True)
        dominant.pixel[WRIST] = (100.0, 100.0)
        move.update(context(obj, (500, 500), hand("Left"), dominant, time=0.0))
        self.assertIs(move.selected, obj)

        dominant.pixel[WRIST] = (150.0, 120.0)
        move.update(context(obj, (500, 500), hand("Left"), dominant, time=0.1))
        self.assertAlmostEqual(obj.nx, 0.575, places=4)
        self.assertAlmostEqual(obj.ny, 0.53, places=4)

        yielded = DhGazePinch()
        yielded.active_scale_method = ScaleMethod.BI_DISTANCE
        target = box()
        left = hand("Left", index_pinch=True)
        right = hand("Right", index_pinch=True)
        right.pixel[WRIST] = (100.0, 100.0)
        yielded.update(context(target, (500, 500), left, right))
        self.assertIsNone(yielded.selected)
        self.assertTrue(yielded.yielded)

    def test_uni_depth_push_toward_camera_shrinks(self):
        interaction = UniDepth()
        obj = box()
        other = hand("Left", index_pinch=True)
        other.camera[INDEX_TIP, 2] = 0.5
        interaction.update(context(obj, (500, 500), other, hand("Right"), time=0.0))
        other.camera[INDEX_TIP, 2] = 0.4
        interaction.update(context(obj, (500, 500), other, hand("Right"), time=0.1))
        self.assertAlmostEqual(obj.scale, 0.9, places=4)

    def test_uni_angle_roll_changes_scale(self):
        interaction = UniAngle()
        obj = box()
        other = hand("Left", index_pinch=True)
        other.pixel[WRIST] = (100.0, 200.0)
        other.pixel[MIDDLE_MCP] = (100.0, 100.0)
        interaction.update(context(obj, (500, 500), other, hand("Right"), time=0.0))
        other.pixel[MIDDLE_MCP] = (200.0, 100.0)
        interaction.update(context(obj, (500, 500), other, hand("Right"), time=0.1))
        self.assertAlmostEqual(obj.scale, 1.5, places=4)

    def test_uni_semi_locks_when_palm_faces_user(self):
        interaction = UniSemi()
        obj = box()
        toward_camera = hand("Left")
        toward_camera.world[WRIST] = (0.0, 0.0, 0.0)
        toward_camera.world[INDEX_MCP] = (1.0, 0.0, 0.0)
        toward_camera.world[PINKY_MCP] = (0.0, 1.0, 0.0)
        interaction.update(context(obj, (500, 500), toward_camera, hand("Right"), time=0.0))
        self.assertFalse(interaction.locked)

        other = hand("Left")
        other.world[WRIST] = (0.0, 0.0, 0.0)
        other.world[INDEX_MCP] = (0.0, 1.0, 0.0)
        other.world[PINKY_MCP] = (1.0, 0.0, 0.0)
        other.world[THUMB_TIP] = (0.0, 0.0, 0.0)
        other.world[INDEX_TIP] = (0.05, 0.0, 0.0)
        interaction.update(context(obj, (500, 500), other, hand("Right"), time=0.0))
        self.assertTrue(interaction.locked)
        self.assertIs(interaction.selected, obj)

        other.world[INDEX_TIP] = (0.10, 0.0, 0.0)
        interaction.update(context(obj, (500, 500), other, hand("Right"), time=0.1))
        self.assertAlmostEqual(obj.scale, 1.75, places=4)

        closed = hand("Left")
        missed = box()
        interaction.reset()
        interaction.update(context(missed, (500, 500), closed, hand("Right"), time=0.2))
        self.assertFalse(interaction.locked)
        self.assertIsNone(interaction.selected)

    def test_uni_micro_tap_hysteresis_and_zoom(self):
        interaction = UniMicro()
        tracked = hand("Left")
        interaction.combined_angle = 10.0
        self.assertTrue(interaction._tabbing(tracked))
        interaction.combined_angle = 50.0
        self.assertTrue(interaction._tabbing(tracked))
        interaction.combined_angle = 90.0
        self.assertFalse(interaction._tabbing(tracked))

        zoom = UniMicro()
        obj = box()
        zoom.selected = obj
        zoom.initial_scale = 1.0
        zoom.ready = True
        sample = hand("Left")
        sample.world[THUMB_TIP] = (0.0, 0.0, 0.0)
        sample.world[INDEX_MCP] = (0.1, 0.0, 0.0)
        sample.world[INDEX_TIP] = (0.2, 0.0, 0.0)
        zoom.initial_thumb_knuckle = 0.1
        zoom.initial_thumb_tip = 0.2
        sample.world[INDEX_TIP] = (0.25, 0.0, 0.0)
        zoom._update_zoom(context(obj, (500, 500), sample, time=0.0), sample)
        self.assertAlmostEqual(obj.scale, 1.0 + (-0.05 / 0.15) * 2.0, places=4)

    def test_plane_line_angle_is_zero_when_aligned(self):
        aligned = calculate_angle_between_plane_and_line(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            "XZ",
        )
        perpendicular = calculate_angle_between_plane_and_line(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, -1.0]),
            "XZ",
        )
        self.assertAlmostEqual(aligned, 0.0, places=4)
        self.assertAlmostEqual(perpendicular, 90.0, places=4)

    def test_controller_switches_method_without_keeping_selection(self):
        controller = InteractionController()
        obj = box()
        other = hand("Left", index_pinch=True)
        other.camera[INDEX_TIP, 2] = 0.5
        controller.update(context(obj, (500, 500), other, hand("Right"), time=0.0))
        self.assertIs(controller.uni_depth.selected, obj)
        controller.set_method(ScaleMethod.UNI_ANGLE)
        self.assertIsNone(controller.uni_depth.selected)
        self.assertEqual(controller.method, ScaleMethod.UNI_ANGLE)


if __name__ == "__main__":
    unittest.main()
