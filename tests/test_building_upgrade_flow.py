import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.reflection import ReflectionLog
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.vision import Vision
from src.world_model import WorldModel


class FakeClient:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.taps = []

    def frame_age(self):
        return 0.0

    def get_screenshot(self, fresh=True):
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def tap(self, x, y, jitter_radius=0):
        self.taps.append((x, y))


def paste(frame, template, center_x, center_y):
    height, width = template.shape[:2]
    x1 = center_x - width // 2
    y1 = center_y - height // 2
    frame[y1 : y1 + height, x1 : x1 + width] = template


def build_frames(vision, *, active_confirm=True):
    """Синтетические кадры без данных аккаунта: серый фон + реальные шаблоны."""
    menu = np.full((2436, 1080, 3), 100, dtype=np.uint8)
    panel = np.full((2436, 1080, 3), 100, dtype=np.uint8)
    city = np.full((2436, 1080, 3), 100, dtype=np.uint8)

    radial = vision.templates_cache[BuildingUpgradeTask.RADIAL_TEMPLATE]
    required = vision.templates_cache[BuildingUpgradeTask.REQUIRED_TEMPLATE]
    confirm = vision.templates_cache[BuildingUpgradeTask.CONFIRM_TEMPLATE].copy()

    paste(menu, radial, 590, 1358)
    paste(panel, required, 540, 1455)
    if not active_confirm:
        confirm = cv2.cvtColor(cv2.cvtColor(confirm, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    paste(panel, confirm, 820, 2315)
    return menu, panel, city


class BuildingUpgradeFlowTest(unittest.TestCase):
    def setUp(self):
        self.vision = Vision(templates_dir="assets")
        self.tmp = tempfile.TemporaryDirectory()
        self.reflection = ReflectionLog(
            str(Path(self.tmp.name) / "reflections.json")
        )
        self.world = WorldModel(str(Path(self.tmp.name) / "world_model.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_flow_taps_only_radial_and_blue_confirm(self):
        menu, panel, city = build_frames(self.vision)
        client = FakeClient([panel, city])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            world=self.world,
        )

        with mock.patch.object(
            task,
            "_read_upgrade_context",
            return_value={"name": "wall", "current_level": 4, "target_level": 5, "duration": 712},
        ):
            self.assertTrue(task.can_run())
            self.assertTrue(task.execute(menu))

        self.assertEqual(client.taps, [(590, 1358), (820, 2315)])
        self.assertEqual(self.reflection.task_stats("BuildingUpgrade")["successes"], 1)
        self.assertEqual(self.world.state["builders"]["builder_1"]["target"], "wall")
        self.assertTrue(self.world.state["builders"]["builder_1"]["busy"])
        self.assertEqual(self.world.state["buildings"]["wall"]["level"], 4)
        remaining = self.world.state["builders"]["builder_1"]["timer_end"] - time.time()
        self.assertAlmostEqual(remaining, 712, delta=5)

    def test_inactive_gray_confirm_is_not_pressed(self):
        menu, panel, _ = build_frames(self.vision, active_confirm=False)
        client = FakeClient([panel])
        task = BuildingUpgradeTask(
            client, self.vision, interval_sec=1, reflection=self.reflection
        )

        self.assertTrue(task.can_run())
        self.assertFalse(task.execute(menu))
        self.assertEqual(client.taps, [(590, 1358)])

    def test_spot_formats(self):
        self.assertEqual(BuildingUpgradeTask._parse_spots(None)[0][1], (639, 1500))
        self.assertEqual(
            BuildingUpgradeTask._parse_spots([[100, 200]])[0],
            (None, (100, 200)),
        )
        self.assertEqual(
            BuildingUpgradeTask._parse_spots([{"name": "wall", "point": [1, 2]}])[0],
            ("wall", (1, 2)),
        )
        self.assertEqual(
            BuildingUpgradeTask._parse_spots([("farm", [3, 4])])[0],
            ("farm", (3, 4)),
        )

    @unittest.skipUnless(
        Path("logs/inspect/building_upgrade_wall.png").exists(),
        "локальный скриншот панели улучшения отсутствует",
    )
    def test_ocr_reads_wall_levels_and_timer(self):
        panel = cv2.imread("logs/inspect/building_upgrade_wall.png")
        task = BuildingUpgradeTask(
            FakeClient([panel]), self.vision, interval_sec=1, reflection=self.reflection
        )
        context = task._read_upgrade_context(panel)
        self.assertEqual(context.get("name"), "wall")
        self.assertEqual(context.get("current_level"), 4)
        self.assertEqual(context.get("target_level"), 5)
        self.assertEqual(context.get("duration"), 712)


if __name__ == "__main__":
    unittest.main()
