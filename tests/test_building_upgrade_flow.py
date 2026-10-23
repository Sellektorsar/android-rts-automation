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


class FakeGuard:
    """Минимальный guard: кадр считается свежим, шаги панели не вызываются."""

    def refresh_if_stale(self, screen, max_age=8.0):
        return screen, "ok"


class FakeSweepClient(FakeClient):
    """FakeClient со свайпами: для тестов обхода камерой."""

    def __init__(self, frames, screen_size=(1080, 2436)):
        super().__init__(frames)
        self.screen_size = screen_size
        self.swipes = []

    def swipe(self, x1, y1, x2, y2, duration_ms=350, smooth=True):
        self.swipes.append((x1, y1, x2, y2))


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
        self._knowledge_path = BuildingUpgradeTask.KNOWLEDGE_PATH
        BuildingUpgradeTask.KNOWLEDGE_PATH = Path(self.tmp.name) / "building_spots.json"

    def tearDown(self):
        BuildingUpgradeTask.KNOWLEDGE_PATH = self._knowledge_path
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

    def test_capped_building_marks_spot_and_skips_until_castle_level_up(self):
        """Серая кнопка + «Перейти»: точка помечается и пропускается."""
        menu, panel, city = build_frames(self.vision, active_confirm=False)
        paste(panel, self.vision.templates_cache["btn_go_castle"], 942, 1532)
        paste(panel, self.vision.templates_cache["btn_back_arrow"], 111, 169)

        # Guard отдаёт панель, затем задача перепроверяет кап на свежем
        # кадре — панель после серой кнопки остаётся на экране.
        client = FakeClient([menu, panel, panel, city, city])
        self.world.state["account"]["castle_level"] = 5
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            world=self.world,
            spots=[{"name": "farm", "point": [400, 1400]}],
        )

        with mock.patch.object(
            task,
            "_read_upgrade_context",
            return_value={"name": "farm", "current_level": 5, "target_level": 6},
        ):
            self.assertFalse(task.execute(city))

        # Панель закрыта стрелкой, серая кнопка «Улучшить» не нажата.
        self.assertEqual(
            client.taps, [(400, 1400), (590, 1358), (111, 169)]
        )
        self.assertEqual(task._capped.get("pt:farm:400,1400"), 5)

        # Повторный такт: упёршаяся точка пропускается, новых кликов нет.
        client.frames = [city]
        client.index = 0
        taps_before = list(client.taps)
        self.assertFalse(task.execute(city))
        self.assertEqual(client.taps, taps_before)

        # Замок поднялся — кап снимается.
        self.world.state["account"]["castle_level"] = 6
        task._looks_like_upgrade_panel = mock.Mock(return_value=False)
        task._resolve_spot = mock.Mock(return_value=None)
        task.sweep["enabled"] = False
        self.assertFalse(task.execute(city))
        self.assertEqual(task._capped, {})

    def test_castle_flow_goes_through_go_and_presses_only_blue_confirm(self):
        menu, cap_panel, city = build_frames(self.vision, active_confirm=False)
        paste(cap_panel, self.vision.templates_cache["btn_go_castle"], 942, 1532)
        paste(cap_panel, self.vision.templates_cache["btn_back_arrow"], 111, 169)

        castle_panel = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        paste(
            castle_panel,
            self.vision.templates_cache[BuildingUpgradeTask.REQUIRED_TEMPLATE],
            540,
            1455,
        )
        paste(
            castle_panel,
            self.vision.templates_cache[BuildingUpgradeTask.CONFIRM_TEMPLATE],
            820,
            2315,
        )

        self.world.state["account"]["castle_level"] = 5
        self.world.state["buildings"]["castle"]["level"] = 5
        key = "pt:farm:400,1400"
        BuildingUpgradeTask.KNOWLEDGE_PATH.write_text(
            '{"%s": {"status": "capped", "castle_level": 5}}' % key,
            encoding="utf-8",
        )
        client = FakeClient([menu, cap_panel, castle_panel, city, city])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            world=self.world,
            spots=[{"name": "farm", "point": [400, 1400]}],
            castle={"enabled": True, "retry_sec": 60},
        )

        with mock.patch.object(
            task,
            "_read_upgrade_context",
            return_value={
                "name": "castle",
                "current_level": 5,
                "target_level": 6,
                "duration": 3600,
            },
        ):
            self.assertTrue(task.execute(city))

        self.assertEqual(
            client.taps, [(400, 1400), (590, 1358), (942, 1532), (820, 2315)]
        )
        builder = self.world.state["builders"]["builder_1"]
        self.assertEqual(builder["target"], "castle")
        self.assertTrue(builder["busy"])

    def _fast_clock(self):
        state = {"t": 0.0}

        def tick():
            state["t"] += 1.0
            return state["t"]

        return mock.patch(
            "src.tasks.building_upgrade.time.monotonic", side_effect=tick
        ), mock.patch("src.tasks.building_upgrade.time.sleep")

    def test_busy_building_is_skipped_with_cooldown(self):
        """Меню улучшающегося здания («Ускорить») не крутит точку бесконечно."""
        city = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        client = FakeClient([city] * 80)
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            spots=[
                {"name": "camp", "point": [455, 1800]},
                {"name": "farm", "point": [600, 1030]},
            ],
        )
        task.guard = FakeGuard()
        task._find_radial = mock.Mock(return_value=None)
        task._looks_like_upgrade_panel = mock.Mock(return_value=False)
        task._overlay_text_near = mock.Mock(return_value="ускоритьосталось")

        clock, sleep = self._fast_clock()
        with clock, sleep:
            self.assertFalse(task.execute(city))
            busy = task._spot_is_busy("pt:camp:455,1800")

        self.assertTrue(busy)
        self.assertEqual(client.taps, [(455, 1800)])
        self.assertNotIn("pt:camp:455,1800", task._spot_failures)
        self.assertEqual(task._spot_index, 1)

    def test_unrecognized_menu_retries_are_capped(self):
        """Неопознанное меню: 4 повтора, затем точка пропускается."""
        city = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        client = FakeClient([city] * 400)
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            spots=[
                {"name": "camp", "point": [455, 1800]},
                {"name": "farm", "point": [600, 1030]},
            ],
        )
        task.guard = FakeGuard()
        task._find_radial = mock.Mock(return_value=None)
        task._looks_like_upgrade_panel = mock.Mock(return_value=False)
        task._overlay_text_near = mock.Mock(return_value="длинныйтекстменю")

        clock, sleep = self._fast_clock()
        with clock, sleep:
            for _ in range(4):
                self.assertFalse(task.execute(city))
            self.assertFalse(task.execute(city))

        self.assertEqual(
            client.taps,
            [(455, 1800), (455, 1800), (455, 1800), (455, 1800), (600, 1030)],
        )
        self.assertNotIn("pt:camp:455,1800", task._spot_failures)

    @unittest.skipUnless(
        Path("logs/inspect/stuck_plot_build.png").exists(),
        "локальный скриншот экрана строительства участка отсутствует",
    )
    def test_plot_build_screen_is_closed_by_back_arrow(self):
        """Экран «Построить» не считается панелью улучшения и закрывается стрелкой."""
        frame = cv2.imread("logs/inspect/stuck_plot_build.png")
        client = FakeClient([frame, frame])
        task = BuildingUpgradeTask(
            client, self.vision, interval_sec=1, reflection=self.reflection
        )
        self.assertFalse(task.execute(frame))
        self.assertEqual(len(client.taps), 1)
        x, y = client.taps[0]
        self.assertAlmostEqual(x, 111, delta=12)
        self.assertAlmostEqual(y, 169, delta=12)

    def test_spot_formats(self):
        self.assertEqual(BuildingUpgradeTask._parse_spots(None)[0], ("wall", (639, 1500), None, 0))
        self.assertEqual(
            BuildingUpgradeTask._parse_spots([[100, 200]])[0],
            (None, (100, 200), None, 0),
        )
        self.assertEqual(
            BuildingUpgradeTask._parse_spots([{"name": "wall", "point": [1, 2]}])[0],
            ("wall", (1, 2), None, 0),
        )
        self.assertEqual(
            BuildingUpgradeTask._parse_spots([("farm", [3, 4])])[0],
            ("farm", (3, 4), None, 0),
        )
        self.assertEqual(
            BuildingUpgradeTask._parse_spots(
                [{"name": "wall", "template": "building_wall_tower"}]
            )[0],
            ("wall", None, "building_wall_tower", 0),
        )
        self.assertEqual(
            BuildingUpgradeTask._parse_spots(
                [{"name": "wall", "template": "building_wall_tower", "tap_dy": 30}]
            )[0],
            ("wall", None, "building_wall_tower", 30),
        )

    def test_capped_hydration_from_passport_only_for_current_castle(self):
        self.world.state["account"]["castle_level"] = 5
        key = "pt:farm:400,1400"
        BuildingUpgradeTask.KNOWLEDGE_PATH.write_text(
            '{"%s": {"status": "capped", "castle_level": 5}, '
            '"pt:farm:500,1500": {"status": "capped", "castle_level": 4}}' % key,
            encoding="utf-8",
        )
        task = BuildingUpgradeTask(
            FakeClient([np.zeros((10, 10, 3), dtype=np.uint8)]),
            self.vision,
            interval_sec=1,
            world=self.world,
            spots=[{"name": "farm", "point": [400, 1400]}],
        )
        self.assertEqual(task._capped, {key: 5})

    def test_castle_disabled_by_default(self):
        task = BuildingUpgradeTask(FakeClient([]), self.vision, interval_sec=1)
        self.assertFalse(task.castle["enabled"])
        self.assertFalse(task._castle_ready())

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

    @unittest.skipUnless(
        Path("logs/inspect/sawmill_radial_1.png").exists(),
        "локальный скриншот широкого радиала отсутствует",
    )
    def test_wide_production_radial_is_found(self):
        frame = cv2.imread("logs/inspect/sawmill_radial_1.png")
        task = BuildingUpgradeTask(
            FakeClient([frame]), self.vision, interval_sec=1, reflection=self.reflection
        )
        match = task._find_radial(frame, center=(300, 1560))
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match[0], 295, delta=8)
        self.assertAlmostEqual(match[1], 1690, delta=8)

    @unittest.skipUnless(
        Path("logs/inspect/tent_probe/frame_0.png").exists(),
        "локальный скриншот меню медицинской палатки отсутствует",
    )
    def test_medic_tent_radial_is_found(self):
        frame = cv2.imread("logs/inspect/tent_probe/frame_0.png")
        task = BuildingUpgradeTask(
            FakeClient([frame]), self.vision, interval_sec=1, reflection=self.reflection
        )
        match = task._find_radial(frame, center=(625, 1580))
        self.assertIsNotNone(match)
        # Глиф-шаблон находится раньше полного кольца и даёт центр глифа.
        self.assertAlmostEqual(match[0], 776, delta=12)
        self.assertAlmostEqual(match[1], 1714, delta=12)

    @unittest.skipUnless(
        Path("logs/inspect/sawmill650_1780_now.png").exists(),
        "локальный скриншот золотого меню лесопилки отсутствует",
    )
    def test_gold_hammer_radial_is_found(self):
        frame = cv2.imread("logs/inspect/sawmill650_1780_now.png")
        task = BuildingUpgradeTask(
            FakeClient([frame]), self.vision, interval_sec=1, reflection=self.reflection
        )
        match = task._find_radial(frame, center=(650, 1780))
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match[0], 656, delta=10)
        self.assertAlmostEqual(match[1], 1886, delta=10)

    @unittest.skipUnless(
        Path("logs/inspect/camp455_1800_now.png").exists(),
        "локальный скриншот затенённого золотого меню отсутствует",
    )
    def test_dim_gold_radial_at_bottom_edge_is_found(self):
        frame = cv2.imread("logs/inspect/camp455_1800_now.png")
        task = BuildingUpgradeTask(
            FakeClient([frame]), self.vision, interval_sec=1, reflection=self.reflection
        )
        match = task._find_radial(frame, center=(455, 1800))
        self.assertIsNotNone(match)
        self.assertAlmostEqual(match[0], 566, delta=12)
        self.assertAlmostEqual(match[1], 1954, delta=12)

    def test_template_spot_taps_matched_building(self):

        """Шаблонная точка: здание ищется на кадре, тап — со смещением tap_dy.

        Живой урок 2026-10-22: в изометрии кликабельна подошва здания,
        поэтому тап делается ниже центра найденного шаблона.
        """
        rng = np.random.default_rng(3)
        building = rng.integers(0, 256, (40, 30, 3), dtype=np.uint8).astype(np.uint8)
        self.vision.add_template("test_building", building)

        city = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        menu = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        panel = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        after = np.full((2436, 1080, 3), 100, dtype=np.uint8)

        paste(city, building, 400, 1400)
        radial = self.vision.templates_cache[BuildingUpgradeTask.RADIAL_TEMPLATE]
        # кольцо меню над подошвой: центр тапа (400, 1500); кнопка целиком
        # внутри области поиска района меню
        paste(menu, radial, 300, 1280)
        required = self.vision.templates_cache[BuildingUpgradeTask.REQUIRED_TEMPLATE]
        confirm = self.vision.templates_cache[BuildingUpgradeTask.CONFIRM_TEMPLATE]
        paste(panel, required, 540, 1455)
        paste(panel, confirm, 820, 2315)

        client = FakeClient([city, menu, panel, after])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            world=self.world,
            spots=[{"name": "wall", "template": "test_building", "tap_dy": 100}],
        )
        with mock.patch.object(
            task,
            "_read_upgrade_context",
            return_value={"name": "wall", "current_level": 4, "target_level": 5, "duration": 712},
        ):
            self.assertTrue(task.can_run())
            self.assertTrue(task.execute(city))

        self.assertEqual(client.taps, [(400, 1500), (300, 1280), (820, 2315)])
        self.assertEqual(self.reflection.task_stats("BuildingUpgrade")["successes"], 1)

    def test_template_spot_missing_is_skipped_without_taps(self):
        """Шаблон здания не найден (вне экрана после смены камеры) — кликов нет."""
        city = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        client = FakeClient([city])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            spots=[{"name": "wall", "template": "absent_building"}],
        )
        self.assertFalse(task.execute(city))
        self.assertEqual(client.taps, [])

    def test_camera_sweep_finds_building_off_screen(self):
        """Обход камерой находит здание, которого не было на стартовом кадре."""
        rng = np.random.default_rng(7)
        building = rng.integers(0, 256, (40, 30, 3), dtype=np.uint8)
        self.vision.add_template("test_sweep_building", building)

        hit = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        paste(hit, building, 700, 900)

        # Первый кадр обхода (верхний левый угол сетки) уже содержит здание.
        client = FakeSweepClient([hit])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            spots=[{"name": "forge", "template": "test_sweep_building", "tap_dy": 50}],
            sweep={"enabled": True, "cols": 3, "rows": 3, "step_px": 600, "settle_s": 0},
        )

        resolved = task._sweep_for_spots()

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved[0], "forge")
        self.assertEqual(resolved[1][0], 700)
        self.assertAlmostEqual(resolved[1][1], 950, delta=5)
        # Дошли до угла сетки двумя свайпами по осям, дальше не понадобилось.
        self.assertEqual(len(client.swipes), 2)
        self.assertEqual(client.taps, [])

    def test_camera_sweep_restores_camera_when_building_not_found(self):
        """Если обход не нашёл здание, камера возвращается теми же свайпами."""
        city = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        client = FakeSweepClient([city])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            spots=[{"name": "forge", "template": "absent_sweep_building"}],
            sweep={"enabled": True, "cols": 3, "rows": 3, "step_px": 600, "settle_s": 0},
        )

        self.assertIsNone(task._sweep_for_spots())

        # 10 свайпов вперёд (2 диагональ + 8 по сетке) и 10 назад.
        self.assertEqual(len(client.swipes), 20)
        forward, backward = client.swipes[:10], client.swipes[10:]
        for (fx1, fy1, fx2, fy2), (bx1, by1, bx2, by2) in zip(
            reversed(forward), backward
        ):
            self.assertEqual(fx2 - fx1, -(bx2 - bx1))
            self.assertEqual(fy2 - fy1, -(by2 - by1))
        self.assertEqual(client.taps, [])

    def test_execute_uses_sweep_when_no_spot_visible(self):
        """execute запускает обход, когда ни одно шаблонное здание не видно."""
        city = np.full((2436, 1080, 3), 100, dtype=np.uint8)
        client = FakeClient([city, city])
        task = BuildingUpgradeTask(
            client,
            self.vision,
            interval_sec=1,
            reflection=self.reflection,
            spots=[{"name": "forge", "template": "absent_execute_building"}],
            sweep={"enabled": True},
        )
        task.require_templates = mock.Mock(return_value=True)
        task._looks_like_upgrade_panel = mock.Mock(return_value=False)
        task.guard = None
        task._tap_radial = mock.Mock(return_value=False)

        with mock.patch.object(
            task,
            "_sweep_for_spots",
            return_value=("forge", (500, 600), (500, 600)),
        ) as sweep:
            with mock.patch.object(time, "sleep"):
                self.assertFalse(task.execute(city))

        sweep.assert_called_once()
        self.assertIn((500, 600), client.taps)


if __name__ == "__main__":
    unittest.main()
