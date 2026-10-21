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


if __name__ == "__main__":
    unittest.main()
