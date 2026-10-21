import time
import unittest

from src.tasks.base_task import BaseTask


class DummyVision:
    def __init__(self, available):
        self.available = set(available)

    def has_template(self, name):
        return name in self.available


class DummyTask(BaseTask):
    def __init__(self, interval_sec=10):
        super().__init__("Dummy", client=None, vision=DummyVision({"a"}), interval_sec=interval_sec)

    def execute(self, screen):
        return False


class BaseTaskThrottleTest(unittest.TestCase):
    def test_attempt_is_throttled_even_on_failure(self):
        task = DummyTask(interval_sec=10)
        self.assertTrue(task.can_run())
        # Сразу после попытки задача не должна запускаться снова, даже если
        # execute вернул False.
        self.assertFalse(task.can_run())

    def test_run_allowed_after_interval(self):
        task = DummyTask(interval_sec=5)
        self.assertTrue(task.can_run())
        task.last_attempt_time = time.time() - 6
        self.assertTrue(task.can_run())

    def test_success_also_throttles(self):
        task = DummyTask(interval_sec=10)
        self.assertTrue(task.can_run())
        task.mark_executed()
        self.assertFalse(task.can_run())

    def test_missing_templates_reported_once(self):
        task = DummyTask()
        self.assertFalse(task.require_templates(["a", "b"]))
        self.assertIn("b", task._missing_templates_reported)
        self.assertTrue(task.require_templates(["a"]))


if __name__ == "__main__":
    unittest.main()
