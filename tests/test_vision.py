import unittest

import cv2
import numpy as np

from src.vision import Vision


class VisionFindAllTest(unittest.TestCase):
    def setUp(self):
        self.vision = Vision(templates_dir="assets")
        rng = np.random.default_rng(42)
        self.template = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.image[50:70, 60:80] = self.template
        self.image[120:140, 30:50] = self.template
        self.vision.add_template("test_noise", self.template)

    def test_finds_two_matches(self):
        matches = self.vision.find_all(self.image, "test_noise", threshold=0.99, min_distance=10)
        centers = {(x, y) for x, y, _ in matches}
        self.assertEqual(centers, {(70, 60), (40, 130)})

    def test_region_limits_search(self):
        matches = self.vision.find_all(
            self.image, "test_noise", threshold=0.99, min_distance=10, region=(0, 100, 200, 100)
        )
        self.assertEqual([(x, y) for x, y, _ in matches], [(40, 130)])

    def test_scale_returns_original_coordinates(self):
        matches = self.vision.find_all(
            self.image, "test_noise", threshold=0.95, min_distance=10, scale=0.5
        )
        centers = {(x, y) for x, y, _ in matches}
        self.assertEqual(centers, {(70, 60), (40, 130)})


class VisionMultiscaleTest(unittest.TestCase):
    """Шаблон вставляется в кадр в другом размере — find_template_multiscale
    должен найти его и вернуть относительный масштаб."""

    def setUp(self):
        self.vision = Vision(templates_dir="assets")
        rng = np.random.default_rng(7)
        self.template = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        self.vision.add_template("test_ms", self.template)

    def test_finds_scaled_up_template(self):
        scaled = cv2.resize(self.template, (45, 30), interpolation=cv2.INTER_LINEAR)
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[90:120, 150:195] = scaled
        found = self.vision.find_template_multiscale(
            frame, "test_ms", threshold=0.9, scales=(0.8, 1.0, 1.5)
        )
        self.assertIsNotNone(found)
        cx, cy, conf, rel = found
        self.assertAlmostEqual(cx, 172, delta=2)
        self.assertAlmostEqual(cy, 105, delta=2)
        self.assertGreaterEqual(conf, 0.9)
        self.assertEqual(rel, 1.5)

    def test_finds_scaled_down_template_in_region(self):
        scaled = cv2.resize(self.template, (15, 10), interpolation=cv2.INTER_AREA)
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[200:210, 250:265] = scaled
        found = self.vision.find_template_multiscale(
            frame,
            "test_ms",
            threshold=0.9,
            region=(200, 150, 200, 150),
            scales=(1.0, 0.5),
        )
        self.assertIsNotNone(found)
        cx, cy, conf, rel = found
        self.assertAlmostEqual(cx, 257, delta=2)
        self.assertAlmostEqual(cy, 205, delta=2)
        self.assertGreaterEqual(conf, 0.9)
        self.assertEqual(rel, 0.5)

    def test_returns_none_below_threshold(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        self.assertIsNone(
            self.vision.find_template_multiscale(
                frame, "test_ms", threshold=0.9, scales=(1.0,)
            )
        )


class VisionDiffTest(unittest.TestCase):
    def test_region_ignores_changes_outside_it(self):
        first = np.zeros((200, 200, 3), dtype=np.uint8)
        second = first.copy()
        second[0:50, 0:50] = 255  # изменение вне проверяемой области

        self.assertGreater(Vision.diff_score(first, second), 0.0)
        self.assertEqual(Vision.diff_score(first, second, region=(100, 100, 80, 80)), 0.0)

    def test_region_reports_change_inside_it(self):
        first = np.zeros((200, 200, 3), dtype=np.uint8)
        second = first.copy()
        second[120:160, 120:160] = 255

        self.assertGreater(Vision.diff_score(first, second, region=(100, 100, 80, 80)), 0.0)


if __name__ == "__main__":
    unittest.main()
