import tempfile
import time
import unittest
from pathlib import Path

from src.world_model import WorldModel


class WorldModelTimerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = WorldModel(str(Path(self.tmp.name) / "world_model.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_and_expire_upgrade(self):
        self.assertTrue(self.model.start_upgrade("wall", 60))
        self.assertEqual(self.model.builders_available(), 0)
        self.assertIsNotNone(self.model.next_event())
        self.assertGreater(self.model.seconds_until_next_event(), 0)

        self.model.state["builders"]["builder_1"]["timer_end"] = time.time() - 1
        completed = self.model.clean_expired_timers()
        self.assertIn("building:wall", completed)
        self.assertEqual(self.model.state["buildings"]["wall"]["level"], 2)
        self.assertTrue(self.model.is_builder_free())

    def test_second_builder_requires_unlock(self):
        self.model.state["account"]["has_second_builder"] = False
        self.assertTrue(self.model.start_upgrade("wall", 60))
        self.assertFalse(self.model.start_upgrade("farm", 60))
        self.assertEqual(self.model.builders_available(), 0)

    def test_second_builder_adds_a_slot(self):
        self.model.state["account"]["has_second_builder"] = True
        self.assertTrue(self.model.start_upgrade("wall", 60))
        self.assertTrue(self.model.start_upgrade("farm", 60))
        self.assertEqual(self.model.state["builders"]["builder_2"]["target"], "farm")
        self.assertEqual(self.model.builders_available(), 0)

    def test_summary_contains_builders(self):
        summary = self.model.get_summary()
        self.assertIn("builders=", summary)

    def test_upgrade_uses_current_level_from_panel(self):
        self.model.state["buildings"]["wall"]["level"] = 4
        self.assertTrue(self.model.start_upgrade("wall", 60, current_level=4))
        self.model.state["builders"]["builder_1"]["timer_end"] = time.time() - 1
        self.model.clean_expired_timers()
        self.assertEqual(self.model.state["buildings"]["wall"]["level"], 5)

    def test_upgrade_without_level_uses_existing(self):
        self.model.state["buildings"]["wall"]["level"] = 5
        self.assertTrue(self.model.start_upgrade("wall", 60))
        self.model.state["builders"]["builder_1"]["timer_end"] = time.time() - 1
        self.model.clean_expired_timers()
        self.assertEqual(self.model.state["buildings"]["wall"]["level"], 6)

    def test_castle_timer_raises_account_castle_level(self):
        self.model.state["account"]["castle_level"] = 5
        self.model.state["buildings"]["castle"]["level"] = 5
        self.assertTrue(self.model.start_upgrade("castle", 60, current_level=5))
        self.model.state["builders"]["builder_1"]["timer_end"] = time.time() - 1
        completed = self.model.clean_expired_timers()
        self.assertIn("building:castle", completed)
        self.assertEqual(self.model.state["buildings"]["castle"]["level"], 6)
        self.assertEqual(self.model.state["account"]["castle_level"], 6)

    def test_campaign_defaults_and_update(self):
        self.assertEqual(self.model.state["campaign"]["mode"], "Спаситель порядка")
        self.model.update_campaign(
            chapter=4,
            chapter_title="Вербовка Войск I",
            stage=7,
            stage_objective="Выжить в течение 120 сек.",
            stage_new_skill="Удар дракона",
            difficulty="Простой",
        )
        campaign = self.model.state["campaign"]
        self.assertEqual(campaign["chapter"], 4)
        self.assertEqual(campaign["stage"], 7)
        self.assertEqual(campaign["stage_new_skill"], "Удар дракона")
        self.assertEqual(campaign["difficulty"], "Простой")

    def test_campaign_update_keeps_known_values_on_none(self):
        self.model.update_campaign(chapter=4, stage=7)
        self.model.update_campaign(chapter=None, stage=None)
        self.assertEqual(self.model.state["campaign"]["chapter"], 4)
        self.assertEqual(self.model.state["campaign"]["stage"], 7)

    def test_record_and_get_events(self):
        self.model.record_event("battle:victory", "stage_7")
        events = self.model.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "battle:victory")
        self.assertEqual(events[0]["detail"], "stage_7")

        self.model.record_event("crisis:food", "world_forest")
        self.assertEqual(len(self.model.get_events()), 2)
        self.assertEqual(self.model.get_events(limit=1)[0]["kind"], "crisis:food")


if __name__ == "__main__":
    unittest.main()
