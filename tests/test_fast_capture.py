"""Базовые проверки контракта FastCapture без запуска ADB/ffmpeg."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
