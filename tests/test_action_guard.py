import time
import unittest

import numpy as np

from src.action_guard import ActionGuard


class FakeVision:
    def __init__(self, present_marker=200):
        self.present_marker = present_marker

    def has_template(self, name):
        return True

    def find_best(self, frame, names, threshold=0.8, region=None, scale=1.0):
        if frame is not None and int(frame[0, 0, 0]) >= self.present_marker:
            return (10, 20, 0.99, names[0])
        return None


class FakeClient:
    def __init__(self, frames, age=0.0):
        self.frames = list(frames)
        self.calls = 0
        self.age = age
        self.taps = []

    def frame_age(self):
        return self.age

    def get_screenshot(self, fresh=False):
        index = min(self.calls, len(self.frames) - 1)
        self.calls += 1
        return self.frames[index]

    def tap(self, x, y, jitter_radius=0):
        self.taps.append((x, y))


def frame(value=0):
    return np.full((120, 80, 3), value, dtype=np.uint8)


def random_frame(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (120, 80, 3), dtype=np.uint8)


class ActionGuardTest(unittest.TestCase):
    def test_rejects_stale_frame(self):
        guard = ActionGuard(FakeClient([frame(100)], age=8.0), FakeVision())
        ok, reason = guard.is_actionable(frame(100), max_age=3.0)
        self.assertFalse(ok)
        self.assertIn("stale_frame", reason)

    def test_rejects_dark_frame(self):
        guard = ActionGuard(FakeClient([frame(0)], age=0.0), FakeVision())
        ok, reason = guard.is_actionable(frame(0), max_age=3.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "dark_frame")

    def test_waits_for_template_to_appear(self):
        client = FakeClient([frame(0), frame(0), frame(220)])
        guard = ActionGuard(client, FakeVision(), poll_interval=0.001)
        result = guard.wait_for_match("button", timeout=1.0)
        self.assertTrue(result.ok)
        self.assertEqual(result.match[0], 10)
        self.assertGreaterEqual(client.calls, 3)

    def test_waits_until_template_disappears(self):
        client = FakeClient([frame(220), frame(220), frame(100)])
        guard = ActionGuard(client, FakeVision(), poll_interval=0.001)
        self.assertTrue(guard.wait_until_gone("button", timeout=1.0))

    def test_wait_for_change_detects_difference(self):
        client = FakeClient([random_frame(1), random_frame(2)])
        guard = ActionGuard(client, FakeVision(), poll_interval=0.001)
        after, changed = guard.wait_for_change(random_frame(1), timeout=1.0)
        self.assertTrue(changed)
        self.assertIsNotNone(after)

    def test_tap_does_not_submit_on_stale_frame(self):
        client = FakeClient([frame(100)], age=9.0)
        guard = ActionGuard(client, FakeVision())
        result = guard.tap(5, 6)
        self.assertFalse(result.ok)
        self.assertEqual(client.taps, [])

    def test_refreshes_stale_frame_once(self):
        class RefreshingClient(FakeClient):
            def __init__(self, frames):
                super().__init__(frames)
                self.age_calls = 0

            def frame_age(self):
                self.age_calls += 1
                return 9.0 if self.age_calls == 1 else 0.0

        client = RefreshingClient([frame(100)])
        guard = ActionGuard(client, FakeVision())
        fresh, reason = guard.refresh_if_stale(frame(100), max_age=3.0)
        self.assertEqual(reason, "refreshed")
        self.assertIsNotNone(fresh)
        self.assertEqual(client.calls, 1)

    def test_refreshes_dark_frame(self):
        client = FakeClient([frame(100)])
        guard = ActionGuard(client, FakeVision())
        fresh, reason = guard.refresh_if_stale(frame(0), max_age=3.0)
        self.assertEqual(reason, "refreshed")
        self.assertIsNotNone(fresh)
        self.assertGreater(float(fresh.mean()), 2.0)

    def test_refresh_fails_if_fresh_frame_is_still_stale(self):
        client = FakeClient([frame(100)], age=9.0)
        guard = ActionGuard(client, FakeVision())
        fresh, reason = guard.refresh_if_stale(frame(100), max_age=3.0)
        self.assertIsNone(fresh)
        self.assertIn("stale_frame", reason)


if __name__ == "__main__":
    unittest.main()
