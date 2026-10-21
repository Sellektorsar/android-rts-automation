import unittest

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
