"""Регрессия восприятия экрана ежедневных заданий.

Живой кадр `quests_probe.png` содержит много зелёных кнопок «Забрать
награды». До появления якоря `title_daily_quests` классификатор принимал
этот экран за `popup_reward`. Тест фиксирует правильный приоритет.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from src.fsm import GameState
from src.perception import ScreenClassifier
from src.vision import Vision


class QuestsPerceptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = Vision(templates_dir="assets")
        cls.classifier = ScreenClassifier(cls.vision, ocr_scale=0.5)

    def test_daily_quests_is_not_reward_popup(self):
        path = Path("logs/inspect/quests_probe.png")
        if not path.exists():
            self.skipTest("нет локального кадра quests_probe.png")
        frame = cv2.imread(str(path))
        self.assertIsNotNone(frame)

        result = self.classifier.classify(frame)

        self.assertEqual(result.state, GameState.QUESTS_MENU)
        self.assertFalse(result.is_popup)
        self.assertEqual(result.source, "anchor")
        self.assertIn("title_daily_quests", result.evidence)


if __name__ == "__main__":
    unittest.main()
