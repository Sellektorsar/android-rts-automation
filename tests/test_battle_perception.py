"""Регрессия восприятия боевого экрана.

Кадр `battle_stage6.jpg` — реальный бой этапа 6 («Победить лидера»).
Якорь `battle_pause` (кнопка паузы в левом верхнем углу) должен
классифицировать его как `tutorial_battle`, но не срабатывать на городе,
карте кампании, брифинге, Королевской Дороге и ежедневных заданиях.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from src.fsm import GameState
from src.perception import ScreenClassifier
from src.vision import Vision


class BattlePerceptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = Vision(templates_dir="assets")
        cls.classifier = ScreenClassifier(cls.vision, ocr_scale=0.5)

    def test_battle_frame_is_tutorial_battle(self):
        path = Path("tests/fixtures/battle_stage6.jpg")
        if not path.exists():
            self.skipTest("нет локального кадра battle_stage6.jpg")
        frame = cv2.imread(str(path))
        self.assertIsNotNone(frame)

        result = self.classifier.classify(frame)

        self.assertEqual(result.state, GameState.TUTORIAL_BATTLE)
        self.assertEqual(result.source, "anchor")
        self.assertIn("battle_pause", result.evidence)

    def test_battle_pause_not_on_city(self):
        path = Path("logs/inspect/city_now.png")
        if not path.exists():
            self.skipTest("нет локального кадра city_now.png")
        frame = cv2.imread(str(path))
        self.assertIsNotNone(frame)

        result = self.classifier.classify(frame)

        self.assertEqual(result.state, GameState.CITY_VIEW)


if __name__ == "__main__":
    unittest.main()
