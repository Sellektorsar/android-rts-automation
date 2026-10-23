"""Тесты автопилота науки (ResearchTask)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src.fsm import GameState
from src.perception import Classification
from src.tasks.research import ResearchTask
from src.vision import Vision


class FakeClient:
    def __init__(self) -> None:
        self.taps = []

    def frame_age(self):
        return 0.0

    def get_screenshot(self, fresh: bool = True):
        return np.full((2436, 1080, 3), 100, dtype=np.uint8)

    def tap(self, x: int, y: int, jitter_radius: int = 0) -> None:
        self.taps.append((x, y))


class ResearchTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = Vision(templates_dir="assets")
        self.client = FakeClient()

    def test_starts_research_in_research_menu(self) -> None:
        task = ResearchTask(self.client, self.vision, interval_sec=1, reflection=None)
        task.enabled = True
        task.require_templates = mock.Mock(return_value=True)
        with mock.patch.object(task.classifier, "classify", return_value=Classification(state=GameState.RESEARCH_MENU)):
            with mock.patch.object(task.vision, "find_template", return_value=(300, 1600, 0.92)):
                self.assertTrue(task.execute(np.full((2436, 1080, 3), 100, dtype=np.uint8)))
        self.assertEqual(self.client.taps, [(300, 1600)])

    def test_skips_research_outside_research_menu(self) -> None:
        task = ResearchTask(self.client, self.vision, interval_sec=1, reflection=None)
        task.enabled = True
        task.require_templates = mock.Mock(return_value=True)
        with mock.patch.object(task.classifier, "classify", return_value=Classification(state=GameState.CITY_VIEW)):
            self.assertFalse(task.execute(np.full((2436, 1080, 3), 100, dtype=np.uint8)))
        self.assertEqual(self.client.taps, [])


if __name__ == "__main__":
    unittest.main()
