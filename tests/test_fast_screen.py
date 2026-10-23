"""Тесты быстрой пиксельной классификации экранов (FastScreen)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.fast_screen import FastScreenAnalyzer


class FlatFrame:
    """Синтетический кадр с одним цветом."""

    def __init__(self, value: int, shape=(2436, 1080, 3)) -> None:
        self.frame = np.full(shape, value, dtype=np.uint8)


class FastScreenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = FastScreenAnalyzer()

    def test_flat_frame_has_no_nav_panel_or_popup(self) -> None:
        frame = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        self.assertFalse(self.analyzer.has_bottom_nav(frame))
        self.assertEqual(self.analyzer.overlay_level(frame), 0.0)
        self.assertFalse(self.analyzer.has_dialog_panel(frame))
        self.assertEqual(self.analyzer.classify(frame), "unknown")

    def test_none_frame_is_unknown(self) -> None:
        self.assertEqual(self.analyzer.classify(None), "unknown")
        self.assertFalse(self.analyzer.has_bottom_nav(None))
        self.assertFalse(self.analyzer.has_dialog_panel(None))
        self.assertEqual(self.analyzer.overlay_level(None), 0.0)

    def test_city_like_frame_with_bottom_nav(self) -> None:
        frame = np.full((2436, 1080, 3), 60, dtype=np.uint8)
        nav = np.array([130, 140, 150], dtype=np.uint8)  # BGR, тёплый
        frame[int(0.92 * 2436) : int(0.99 * 2436), :] = nav
        self.assertTrue(self.analyzer.has_bottom_nav(frame))
        self.assertEqual(self.analyzer.classify(frame), "city")

    def test_popup_frame_detected_by_overlay_level(self) -> None:
        frame = np.full((2436, 1080, 3), 200, dtype=np.uint8)
        frame[int(0.12 * 2436) : int(0.18 * 2436), 0:int(0.08 * 1080)] = 20
        frame[int(0.12 * 2436) : int(0.18 * 2436), int(0.92 * 1080) : 1080] = 20
        frame[int(0.82 * 2436) : int(0.88 * 2436), 0:int(0.08 * 1080)] = 20
        frame[int(0.82 * 2436) : int(0.88 * 2436), int(0.92 * 1080) : 1080] = 20
        self.assertGreater(self.analyzer.overlay_level(frame), 0.5)
        self.assertEqual(self.analyzer.classify(frame), "popup")

    def test_dialog_panel_detected(self) -> None:
        frame = np.full((2436, 1080, 3), 60, dtype=np.uint8)
        frame[int(0.60 * 2436) : int(0.85 * 2436), int(0.1 * 1080) : int(0.9 * 1080)] = 200
        self.assertTrue(self.analyzer.has_dialog_panel(frame))
        self.assertEqual(self.analyzer.classify(frame), "dialog")

    def test_screen_changed_and_menu_opened(self) -> None:
        before = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        after = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        after[0:100, 0:100] = 220
        self.assertTrue(self.analyzer.screen_changed(before, after))
        self.assertFalse(self.analyzer.screen_changed(before, before.copy()))

        panel_frame = np.full((2436, 1080, 3), 60, dtype=np.uint8)
        panel_frame[int(0.60 * 2436) : int(0.85 * 2436), int(0.1 * 1080) : int(0.9 * 1080)] = 200
        self.assertTrue(self.analyzer.menu_opened(before, panel_frame))


if __name__ == "__main__":
    unittest.main()
