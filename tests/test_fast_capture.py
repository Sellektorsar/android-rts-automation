"""Базовые проверки контракта FastCapture без запуска ADB/ffmpeg."""

from __future__ import annotations

import unittest

import numpy as np

from src.bot_engine import BotEngine
from src.fast_capture import FastCapture


class FastCaptureContractTest(unittest.TestCase):
    def test_frame_bytes_matches_geometry(self):
        capture = FastCapture("adb", "127.0.0.1:5555", width=360, height=812)
        self.assertEqual(capture.frame_bytes, 360 * 812 * 3)

    def test_read_without_stream_returns_none(self):
        capture = FastCapture("adb", "127.0.0.1:5555")
        self.assertIsNone(capture.read(timeout_sec=0.1))

    def test_stop_is_idempotent(self):
        capture = FastCapture("adb", "127.0.0.1:5555")
        capture.stop()
        capture.stop()
        self.assertIsNone(capture._adb_proc)
        self.assertIsNone(capture._ffmpeg_proc)

    def test_restart_not_requested_by_default(self):
        capture = FastCapture("adb", "127.0.0.1:5555")
        self.assertFalse(capture._restart_requested)
        self.assertIsNone(capture.read(timeout_sec=0.05))

    def test_read_without_tools_does_not_crash_on_restart_request(self):
        capture = FastCapture("adb", "127.0.0.1:5555")
        capture._restart_requested = True
        self.assertIsNone(capture.read(timeout_sec=0.05))
        self.assertFalse(capture._restart_requested)


class BotEngineFastCaptureTest(unittest.TestCase):
    def test_fast_frame_is_resized_to_device_coordinates(self):
        fast_frame = np.full((1218, 540, 3), 100, dtype=np.uint8)
        engine = BotEngine.__new__(BotEngine)
        engine.fast_capture = type(
            "Capture",
            (),
            {"read": lambda self, timeout_sec: fast_frame},
        )()
        engine.client = type(
            "Client",
            (),
            {
                "screen_size": (1080, 2436),
                "get_screenshot": lambda self, fresh=False: None,
            },
        )()

        frame = engine._capture_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (2436, 1080, 3))

    def test_fast_capture_starts_with_discovered_serial(self):
        capture = type(
            "Capture",
            (),
            {
                "serial": "old:1234",
                "start": lambda self: True,
            },
        )()
        engine = BotEngine.__new__(BotEngine)
        engine.fast_capture = capture
        engine.client = type("Client", (), {"ip": "192.168.1.2", "port": 45678})()

        engine._start_fast_capture()

        self.assertEqual(capture.serial, "192.168.1.2:45678")
        self.assertIs(engine.fast_capture, capture)


if __name__ == "__main__":
    unittest.main()
