"""Тесты автопилота боя: фазы BattleAuto и действия BattleTask."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.battle_auto import BattleAuto, BattlePhase
from src.fsm import GameState
from src.perception import Classification
from src.reflection import ReflectionLog
from src.tasks.battle import BattleTask
from src.world_model import WorldModel
from src.vision import Vision


class FakeClient:
    def __init__(self) -> None:
        self.taps = []
        self.swipes = []
        self.frames = [
            np.full((2436, 1080, 3), 100, dtype=np.uint8),
        ]

    def frame_age(self):
        return 0.0

    def get_screenshot(self, fresh: bool = True):
        return self.frames[0]

    def tap(self, x: int, y: int, jitter_radius: int = 0) -> None:
        self.taps.append((x, y))

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 350,
        smooth: bool = True,
    ) -> None:
        self.swipes.append((x1, y1, x2, y2, duration_ms))


class BattleAutoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.battle = BattleAuto()

    def test_starts_in_battle(self) -> None:
        self.assertEqual(self.battle.phase, BattlePhase.IN_BATTLE)

    def test_kites_circular_cycle_with_cooldown(self) -> None:
        self.assertEqual(self.battle.next_move(now=100.0), (540, 1980, 540, 1720, 1100))
        # Cooldown ещё не истёк — второй кайт подряд не выдаём.
        self.assertIsNone(self.battle.next_move(now=100.5))
        self.assertEqual(
            self.battle.next_move(now=100.0 + self.battle.move_cooldown_sec + 0.01),
            (540, 1980, 780, 1850, 1100),
        )

    def test_custom_move_cycle(self) -> None:
        battle = BattleAuto(move_cycle=[[10, 20, 30, 40, 50]])
        self.assertEqual(battle.next_move(now=1.0), (10, 20, 30, 40, 50))

    def test_skill_selection_blocks_kiting_until_confirmed(self) -> None:
        self.battle.observe(GameState.SKILL_SELECTION)
        self.assertEqual(self.battle.phase, BattlePhase.SKILL_SELECTION)
        self.assertIsNone(self.battle.next_move(now=10.0))

        self.battle.mark_skill_sent()
        self.assertEqual(self.battle.phase, BattlePhase.IN_BATTLE)
        self.assertEqual(self.battle.next_move(now=10.0), (540, 1980, 540, 1720, 1100))

    def test_outcome_phases(self) -> None:
        self.battle.observe(GameState.VICTORY_SCREEN)
        self.assertEqual(self.battle.phase, BattlePhase.VICTORY)

        self.battle.handle_outcome("defeat")
        self.assertEqual(self.battle.phase, BattlePhase.DEFEAT)

        self.battle.handle_outcome("exit")
        self.assertEqual(self.battle.phase, BattlePhase.EXIT)

        self.battle.observe(GameState.CITY_VIEW)
        self.assertEqual(self.battle.phase, BattlePhase.EXIT)


class BattleTaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = Vision(templates_dir="assets")
        self.tmp = tempfile.TemporaryDirectory()
        self.reflection = ReflectionLog(str(Path(self.tmp.name) / "reflections.json"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_battle_phase_kites_and_marks_success(self) -> None:
        client = FakeClient()
        task = BattleTask(client, self.vision, interval_sec=1, reflection=self.reflection)

        frame = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        with mock.patch.object(
            task.classifier,
            "classify",
            return_value=Classification(state=GameState.TUTORIAL_BATTLE),
        ):
            self.assertTrue(task.can_run())
            self.assertTrue(task.execute(frame))

        self.assertEqual(len(client.swipes), 1)
        self.assertEqual(client.swipes[0][:4], (540, 1980, 540, 1720))
        self.assertEqual(task.battle.phase, BattlePhase.IN_BATTLE)
        self.assertEqual(self.reflection.task_stats("BattleAuto")["successes"], 1)

    def test_skill_selection_confirms_after_first_card(self) -> None:
        client = FakeClient()
        task = BattleTask(
            client, self.vision, interval_sec=1, reflection=self.reflection, world=WorldModel()
        )
        frame = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        task.vision.add_template("btn_skill_confirm", np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertTrue(task.can_run())
        with mock.patch.object(
            task.vision,
            "find_template",
            side_effect=[None, (300, 1600, 0.95)],
        ):
            self.assertTrue(task._handle_skill_selection(frame))

        self.assertEqual(client.taps, [(183, 1096), (300, 1600)])
        self.assertEqual(task.battle.phase, BattlePhase.IN_BATTLE)
        self.assertTrue(task.battle.skill_confirmed)
        self.assertEqual(self.reflection.task_stats("BattleAuto")["successes"], 1)

    def test_victory_phase_does_not_tap(self) -> None:
        client = FakeClient()
        world = WorldModel()
        task = BattleTask(
            client, self.vision, interval_sec=1, reflection=self.reflection, world=world
        )
        task.battle.observe(GameState.VICTORY_SCREEN)

        frame = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        with mock.patch.object(
            task.classifier,
            "classify",
            return_value=Classification(state=GameState.VICTORY_SCREEN),
        ):
            self.assertFalse(task.execute(frame))

        self.assertEqual(client.swipes, [])
        self.assertEqual(task.battle.phase, BattlePhase.VICTORY)
        self.assertEqual(world.get_events()[-1]["kind"], "battle:victory")


if __name__ == "__main__":
    unittest.main()
