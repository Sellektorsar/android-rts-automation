import tempfile
import time
import unittest
from pathlib import Path

from src.reflection import ReflectionLog
from src.tasks.base_task import BaseTask


class DummyVision:
    def has_template(self, name):
        return True


class DummyTask(BaseTask):
    def __init__(self, reflection, interval_sec=60):
        super().__init__(
            "Dummy", client=None, vision=DummyVision(), interval_sec=interval_sec,
            reflection=reflection,
        )

    def execute(self, screen):
        return False


class ReflectionBackoffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = ReflectionLog(str(Path(self.tmp.name) / "reflections.json"))
        self.task = DummyTask(self.log, interval_sec=60)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_history_keeps_base_interval(self):
        self.assertEqual(self.task.current_interval(), 60.0)

    def test_repeated_failures_slow_the_task_down(self):
        for _ in range(4):
            self.log.record_failure("Dummy", "timeout", "panel did not open")
        self.assertGreaterEqual(self.task.current_interval(), 60.0 * 1.75)

    def test_recovery_restores_base_interval(self):
        for _ in range(2):
            self.log.record_failure("Dummy", "timeout", "panel did not open")
        for _ in range(6):
            self.log.record_success("Dummy")
        self.assertEqual(self.task.current_interval(), 60.0)

    def test_success_is_recorded_once_per_attempt(self):
        self.assertTrue(self.task.can_run())
        self.task.mark_executed()
        self.task.mark_executed()
        stats = self.log.task_stats("Dummy")
        self.assertEqual(stats["successes"], 1)

    def test_failure_is_recorded_once_per_attempt(self):
        self.assertTrue(self.task.can_run())
        self.task.report_failure("no_panel", "panel missing")
        self.task.report_failure("no_panel", "panel missing")
        stats = self.log.task_stats("Dummy")
        self.assertEqual(stats["failures"], 1)


if __name__ == "__main__":
    unittest.main()
