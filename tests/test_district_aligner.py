"""
Тесты для DistrictAligner (SIFT-гомография квартала города, ADR-022)
и его интеграции в BuildingUpgradeTask.
"""

from pathlib import Path
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.district_aligner import DistrictAligner
from src.tasks.building_upgrade import BuildingUpgradeTask


class TestDistrictAligner(unittest.TestCase):
    def setUp(self):
        self.ref_path = Path("assets/reference/district_ref.png")
        self.live_path = Path("logs/inspect/align_live.png")
        self.aligner = DistrictAligner(ref_path=self.ref_path, min_inliers=15)

    def test_ref_mask_dimensions(self):
        mask = DistrictAligner.make_ref_mask((2436, 1080))
        self.assertEqual(mask.shape, (2436, 1080))
        self.assertEqual(mask[100, 100], 0)  # Top HUD masked
        self.assertEqual(mask[1000, 500], 255)  # City district kept

    def test_live_mask_dimensions(self):
        mask = DistrictAligner.make_live_mask((2436, 1080))
        self.assertEqual(mask.shape, (2436, 1080))
        self.assertEqual(mask[2000, 500], 0)  # Bottom nav masked
        self.assertEqual(mask[1200, 500], 255)  # District kept

    def test_align_with_real_frames(self):
        if not self.ref_path.exists() or not self.live_path.exists():
            self.skipTest("Эталонный или живой кадр отсутствует")

        live = cv2.imread(str(self.live_path))
        H = self.aligner.get_homography(live)
        self.assertIsNotNone(H)
        self.assertEqual(H.shape, (3, 3))

        # Тест трансформации точки (600, 1030)
        pt = self.aligner.transform_point((600, 1030), live)
        self.assertIsNotNone(pt)
        x, y = pt
        self.assertAlmostEqual(x, 434, delta=5)
        self.assertAlmostEqual(y, 1055, delta=5)

        # Проверка кэширования: тот же объект кадра возвращает тот же H без перерасчёта
        H_cached = self.aligner.get_homography(live)
        self.assertIs(H, H_cached)

    def test_synthetic_translation(self):
        if not self.ref_path.exists():
            self.skipTest("Эталонный кадр отсутствует")

        ref = cv2.imread(str(self.ref_path))
        # Сдвиг на dx=+50, dy=+30
        dx, dy = 50, 30
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))

        H = self.aligner.get_homography(shifted)
        self.assertIsNotNone(H)

        test_pt = (500, 1000)
        mapped = self.aligner.transform_point(test_pt, shifted)
        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped[0], test_pt[0] + dx, delta=3)
        self.assertAlmostEqual(mapped[1], test_pt[1] + dy, delta=3)

    def test_alignment_fails_on_black_or_noise(self):
        black_frame = np.zeros((2436, 1080, 3), dtype=np.uint8)
        H_black = self.aligner.get_homography(black_frame)
        self.assertIsNone(H_black)
        pt_black = self.aligner.transform_point((600, 1000), black_frame)
        self.assertIsNone(pt_black)

        noise_frame = np.random.randint(0, 256, (2436, 1080, 3), dtype=np.uint8)
        H_noise = self.aligner.get_homography(noise_frame)
        self.assertIsNone(H_noise)
        pt_noise = self.aligner.transform_point((600, 1000), noise_frame)
        self.assertIsNone(pt_noise)

    def test_out_of_bounds_detection(self):
        if not self.ref_path.exists() or not self.live_path.exists():
            self.skipTest("Кадры отсутствуют")

        live = cv2.imread(str(self.live_path))
        # Точка далеко за пределами экрана
        pt = self.aligner.transform_point((-5000, -5000), live, check_bounds=True)
        self.assertIsNone(pt)

    def test_building_upgrade_task_resolve_spot_integration(self):
        if not self.ref_path.exists() or not self.live_path.exists():
            self.skipTest("Кадры отсутствуют")

        mock_client = MagicMock()
        mock_vision = MagicMock()
        task = BuildingUpgradeTask(
            mock_client,
            mock_vision,
            spots=[{"name": "farm", "point": [600, 1030]}],
            aligner=self.aligner,
        )
        live = cv2.imread(str(self.live_path))
        resolved = task._resolve_spot(live, task.spots[0])
        self.assertIsNotNone(resolved)
        name, (tap_x, tap_y), center = resolved
        self.assertEqual(name, "farm")
        self.assertAlmostEqual(tap_x, 434, delta=5)
        self.assertAlmostEqual(tap_y, 1055, delta=5)

    def test_building_upgrade_task_resolve_spot_blocks_blind_click(self):
        mock_client = MagicMock()
        mock_vision = MagicMock()
        task = BuildingUpgradeTask(
            mock_client,
            mock_vision,
            spots=[{"name": "farm", "point": [600, 1030]}],
            aligner=self.aligner,
        )
        black_frame = np.zeros((2436, 1080, 3), dtype=np.uint8)
        resolved = task._resolve_spot(black_frame, task.spots[0])
        self.assertIsNone(resolved)  # Слепой клик строго заблокирован!


if __name__ == "__main__":
    unittest.main()
