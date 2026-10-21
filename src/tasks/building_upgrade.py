"""
Задача улучшения зданий (BuildingUpgradeTask).

Проверенный на живом устройстве сценарий:
1. В городе тапнуть по зданию (по умолчанию — Стена Замка).
2. В радиальном меню найти кнопку «Улучшить» (три молота).
3. На панели требований найти блок «Требуется».
4. Прочитать название здания, уровни ``LV.N -> LV.M`` и таймер над синей
   кнопкой (например, ``11:52``).
5. Нажать ТОЛЬКО синюю кнопку «Улучшить» в правом нижнем углу.
6. После исчезновения панели записать в WorldModel подтверждённый таймер.

Зелёная кнопка с гемами («Получить сейчас») не нажимается никогда: поиск
подтверждения ограничен правой половиной нижней полосы. Если синей кнопки
нет (не хватает ресурсов/уровня замка), задача просто выходит, а навигация
возвращает бота в город по стрелке назад.

Проверки взяты из esp32-android-agent: после каждого шага не фиксированная
пауза, а ожидание конкретного шаблона. Если панель требований не появилась
или не исчезла после подтверждения — это фиксируется как сбой, а не успех.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.ocr import fuzzy_contains, normalize_text, read_text
from src.tasks.base_task import BaseTask
from src.time_parser import parse_duration
from src.vision import Vision
from src.world_model import WorldModel

# (имя здания или None, точка в пикселях кадра)
Spot = Tuple[Optional[str], Tuple[int, int]]


class BuildingUpgradeTask(BaseTask):
    RADIAL_TEMPLATE = "btn_building_upgrade"
    CONFIRM_TEMPLATE = "btn_upgrade_confirm"
    REQUIRED_TEMPLATE = "title_upgrade_required"

    # Точки зданий в городе при камере по умолчанию. Тап по точке только
    # выделяет здание; улучшение начинается лишь после проверки шаблонов.
    DEFAULT_SPOTS: Tuple[Spot, ...] = (("wall", (639, 1500)),)

    # Нормализованные названия зданий (кириллица уже приведена к латинице).
    BUILDING_ALIASES: Tuple[Tuple[str, str], ...] = (
        # OCR часто читает кириллицу латиницей: «Стена» -> «CTeHa».
        ("stena", "wall"),
        ("cteha", "wall"),
        ("ctena", "wall"),
        ("zamok", "castle"),
        ("kazarm", "barracks"),
        ("kazap", "barracks"),
        ("ferma", "farm"),
        ("oepma", "farm"),
        ("lesopil", "sawmill"),
        ("kamenolom", "quarry"),
        ("rudnik", "iron_mine"),
        ("institut", "academy"),
        ("akadem", "academy"),
        ("gospital", "hospital"),
        ("arena", "arena"),
        ("tavern", "tavern"),
        ("kuzn", "forge"),
        ("rynok", "market"),
        ("posol", "embassy"),
        ("sobor", "cathedral"),
    )

    def __init__(
        self,
        client: AdbClient,
        vision: Vision,
        interval_sec: int = 60,
        spots: Optional[Iterable] = None,
        reflection=None,
        world: Optional[WorldModel] = None,
    ):
        super().__init__(
            "BuildingUpgrade", client, vision, interval_sec=interval_sec, reflection=reflection
        )
        self.spots: Tuple[Spot, ...] = self._parse_spots(spots)
        self.world = world
        self._spot_index = 0

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------
    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates(
            [self.RADIAL_TEMPLATE, self.CONFIRM_TEMPLATE, self.REQUIRED_TEMPLATE]
        ):
            return False

        screen = self.ensure_screen(screen)
        if screen is None:
            return False

        if self.world is not None:
            self.world.clean_expired_timers()
            if not self.world.is_builder_free():
                logger.info("[BuildingUpgrade] Все строители заняты — пропускаем.")
                return False

        # Уже на панели требований (например, после ручного открытия).
        if self._looks_like_upgrade_panel(screen):
            return self._finish_upgrade(screen)

        # Уже открыто радиальное меню здания.
        if self._tap_radial(screen):
            return self._confirm_panel(screen)

        # Город: выделяем следующее здание из списка и проверяем меню.
        building_name, spot = self.spots[self._spot_index % len(self.spots)]
        self._spot_index += 1
        logger.info(f"[BuildingUpgrade] Проверяем здание {building_name or ''} в точке {spot}.")
        self.client.tap(spot[0], spot[1])

        if self.guard is not None:
            radial = self.guard.wait_for_match(
                self.RADIAL_TEMPLATE,
                region=self._radial_region(screen),
                threshold=0.80,
                timeout=3.5,
            )
            if not radial.ok:
                self.report_failure(
                    "radial_not_found",
                    "Кнопка «Улучшить» не появилась после выбора здания",
                )
                logger.info(
                    "[BuildingUpgrade] Кнопка «Улучшить» не появилась — здание занято или недоступно."
                )
                return False
            if not self._activate_radial(radial.match, radial.after or screen):
                return False
            if self.world is not None and building_name:
                self.world.record_coord_hit(building_name, spot[0], spot[1])
            return self._confirm_panel(radial.after or screen, building_name=building_name)

        time.sleep(1.2)
        selected = self.client.get_screenshot(fresh=True)
        if selected is None:
            self.report_failure("no_frame_after_building_tap", "Нет кадра после выбора здания")
            return False
        if not self._tap_radial(selected):
            logger.info("[BuildingUpgrade] Кнопка «Улучшить» не появилась — здание занято или недоступно.")
            return False
        return self._confirm_panel(selected, building_name=building_name)

    # ------------------------------------------------------------------
    # Разбор конфигурации точек
    # ------------------------------------------------------------------
    @classmethod
    def _parse_spots(cls, raw: Optional[Iterable]) -> Tuple[Spot, ...]:
        """Поддерживает три формата конфигурации.

        * ``- [639, 1500]`` — точка без имени;
        * ``- name: wall`` + ``point: [639, 1500]`` — точка с именем;
        * ``- [wall, [639, 1500]]`` — компактная пара.
        """
        parsed: list[Spot] = []
        for item in raw or cls.DEFAULT_SPOTS:
            name: Optional[str] = None
            point = None

            if isinstance(item, dict):
                name = item.get("name")
                point = item.get("point") or item.get("xy") or item.get("coords")
            elif isinstance(item, (list, tuple)):
                if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (list, tuple)):
                    name = item[0]
                    point = item[1]
                elif len(item) >= 2 and all(isinstance(value, (int, float)) for value in item[:2]):
                    point = item[:2]

            if not point or len(point) < 2:
                continue
            parsed.append((str(name) if name else None, (int(point[0]), int(point[1]))))

        return tuple(parsed) or cls.DEFAULT_SPOTS

    # ------------------------------------------------------------------
    # Области поиска
    # ------------------------------------------------------------------
    @staticmethod
    def _radial_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.42),
            int(height * 0.50),
            int(width * 0.26),
            int(height * 0.11),
        )

    @staticmethod
    def _required_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.30),
            int(height * 0.56),
            int(width * 0.40),
            int(height * 0.08),
        )

    @staticmethod
    def _title_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.30),
            int(height * 0.04),
            int(width * 0.45),
            int(height * 0.05),
        )

    @staticmethod
    def _levels_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.38),
            int(height * 0.42),
            int(width * 0.28),
            int(height * 0.05),
        )

    @staticmethod
    def _timer_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Область таймера над синей кнопкой, правее зелёной кнопки гемов."""
        height, width = frame.shape[:2]
        return (
            int(width * 0.60),
            int(height * 0.90),
            int(width * 0.30),
            int(height * 0.06),
        )

    # ------------------------------------------------------------------
    # Шаги
    # ------------------------------------------------------------------
    def _looks_like_upgrade_panel(self, screen: np.ndarray) -> bool:
        return bool(
            self.vision.find_template(
                screen,
                self.REQUIRED_TEMPLATE,
                threshold=0.78,
                region=self._required_region(screen),
            )
        )

    def _tap_radial(self, screen: np.ndarray) -> bool:
        """Нажимает «Улучшить» в радиальном меню выбранного здания."""
        match = self.vision.find_template(
            screen,
            self.RADIAL_TEMPLATE,
            threshold=0.80,
            region=self._radial_region(screen),
        )
        if not match:
            return False
        return self._activate_radial(match, screen)

    def _activate_radial(
        self,
        match: Optional[Tuple[int, int, float]],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Проверяет границы и нажимает найденную кнопку радиального меню."""
        if not match:
            return False
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            size = getattr(self.client, "screen_size", None)
            if size:
                width, height = size
            else:
                width, height = 1080, 2436
        x, y, confidence = match
        if not (0.10 * width < x < 0.90 * width and 0.25 * height < y < 0.85 * height):
            logger.warning(f"[BuildingUpgrade] Кнопка вне игрового поля: ({x}, {y}).")
            return False
        logger.info(
            f"[BuildingUpgrade] Радиальное меню: «Улучшить» ({x}, {y}), conf={confidence:.2f}."
        )
        self.client.tap(x, y)
        return True

    def _confirm_panel(
        self,
        reference: np.ndarray,
        building_name: Optional[str] = None,
    ) -> bool:
        """Ждёт панель требований и нажимает синюю кнопку подтверждения."""
        if self.guard is not None:
            found = self.guard.wait_for_match(
                self.REQUIRED_TEMPLATE,
                region=self._required_region(reference),
                threshold=0.78,
                timeout=4.0,
            )
            if not found.ok:
                self.report_failure(
                    "upgrade_panel_not_found",
                    "Панель «Требуется» не появилась после нажатия «Улучшить»",
                )
                logger.info("[BuildingUpgrade] Панель требований не открылась — отмена.")
                return False
            panel = found.after
        else:
            time.sleep(1.8)
            panel = self.client.get_screenshot(fresh=True)

        if panel is None:
            return False
        return self._finish_upgrade(panel, building_name=building_name)

    def _finish_upgrade(
        self,
        panel: np.ndarray,
        building_name: Optional[str] = None,
    ) -> bool:
        """Читает контекст панели, нажимает синюю кнопку и обновляет таймер."""
        context = self._read_upgrade_context(panel)
        resolved_name = building_name or context.get("name")

        if not self._tap_confirm(panel):
            logger.info("[BuildingUpgrade] Синяя кнопка «Улучшить» недоступна — отмена.")
            return False

        self._record_upgrade(
            name=resolved_name,
            duration=context.get("duration"),
            current_level=context.get("current_level"),
        )
        self.mark_executed()
        return True

    # ------------------------------------------------------------------
    # OCR панели и WorldModel
    # ------------------------------------------------------------------
    def _read_upgrade_context(self, panel: np.ndarray) -> dict:
        """Читает название, уровни и длительность с панели улучшения.

        Все области маленькие, поэтому OCR быстрый (~0.4–0.7 с на область).
        Если что-то не распознано, возвращается частичный словарь: действие
        всё равно можно выполнить, но таймер мира не будет создан.
        """
        context: dict = {}

        title = read_text(
            panel,
            self._title_region(panel),
            scale=2.0,
            min_confidence=0.30,
        )
        name = self._building_name_from_text(title.text)
        if name:
            context["name"] = name

        levels = read_text(
            panel,
            self._levels_region(panel),
            scale=2.0,
            min_confidence=0.30,
        )
        parsed_levels = []
        for item in levels.items:
            match = re.search(r"L\s*V\.?\s*(\d+)", item.text, re.IGNORECASE)
            if match and item.confidence >= 0.55:
                parsed_levels.append(int(match.group(1)))
        if parsed_levels:
            context["current_level"] = parsed_levels[0]
            if len(parsed_levels) > 1:
                context["target_level"] = parsed_levels[1]

        timer = read_text(
            panel,
            self._timer_region(panel),
            scale=2.0,
            min_confidence=0.25,
        )
        duration = parse_duration(timer.text)
        if duration is None and len(timer.items) > 1:
            duration = parse_duration(" ".join(item.text for item in timer.items))
        if duration is not None:
            context["duration"] = duration

        logger.debug(
            "[BuildingUpgrade] Панель: name={}, level={}, target={}, duration={}, timer='{}'.".format(
                context.get("name"),
                context.get("current_level"),
                context.get("target_level"),
                context.get("duration"),
                timer.text,
            )
        )
        return context

    def _building_name_from_text(self, text: str) -> Optional[str]:
        normalized = normalize_text(text)
        if not normalized:
            return None
        for alias, name in self.BUILDING_ALIASES:
            # Нечёткое сравнение нужно из-за OCR: «Стена» может прийти как
            # «CTeHa» или «CTena», а «Казарма» — как «Ka3apma».
            if fuzzy_contains(normalized, [alias], threshold=0.78):
                return name
        return None

    def _record_upgrade(
        self,
        *,
        name: Optional[str],
        duration: Optional[int],
        current_level: Optional[int],
    ) -> None:
        if self.world is None:
            return
        building = name or "building"
        if not duration or duration <= 0:
            logger.warning(
                "[BuildingUpgrade] Улучшение подтверждено, но длительность не распознана — "
                "таймер WorldModel не создаём."
            )
            return
        if self.world.start_upgrade(building, duration, current_level=current_level):
            logger.info(
                f"[BuildingUpgrade] WorldModel: {building} улучшается, "
                f"осталось {duration}с."
            )
        else:
            logger.warning(
                "[BuildingUpgrade] WorldModel не принял таймер: все строители заняты."
            )

    # ------------------------------------------------------------------
    # Подтверждение
    # ------------------------------------------------------------------
    def _tap_confirm(self, screen: np.ndarray) -> bool:
        """Нажимает только синюю кнопку в правом нижнем углу панели улучшения."""
        height, width = screen.shape[:2]
        required_region = self._required_region(screen)

        required = self.vision.find_template(
            screen, self.REQUIRED_TEMPLATE, threshold=0.78, region=required_region
        )
        if not required:
            return False

        # Правая половина нижней полосы: зелёная кнопка гемов сюда не попадает.
        confirm_region = (
            int(width * 0.55),
            int(height * 0.90),
            int(width * 0.45),
            int(height * 0.10),
        )
        match = self.vision.find_template(
            screen, self.CONFIRM_TEMPLATE, threshold=0.84, region=confirm_region
        )
        if not match:
            return False

        x, y, confidence = match
        if x < 0.60 * width:
            logger.warning(
                f"[BuildingUpgrade] Подтверждение слишком слева ({x}); возможно, это кнопка гемов."
            )
            return False

        # Отсекаем неактивную (серую) кнопку: у активной синей насыщенность
        # высокая и синий канал заметно преобладает над красным.
        template = self.vision.templates_cache.get(self.CONFIRM_TEMPLATE)
        if template is not None:
            th, tw = template.shape[:2]
            x1, y1 = max(0, x - tw // 2), max(0, y - th // 2)
            roi = screen[y1 : y1 + th, x1 : x1 + tw]
            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                saturation = float(hsv[:, :, 1].mean())
                blue, _, red = [float(roi[:, :, i].mean()) for i in range(3)]
                if saturation < 60.0 or blue < red + 10.0:
                    logger.info(
                        f"[BuildingUpgrade] Кнопка неактивна (S={saturation:.0f}, B-R={blue - red:.0f}) — не нажимаем."
                    )
                    return False

        logger.success(
            f"[BuildingUpgrade] Нажимаем бесплатное «Улучшить» ({x}, {y}), conf={confidence:.2f}."
        )
        self.client.tap(x, y)

        if self.guard is not None:
            gone = self.guard.wait_until_gone(
                self.REQUIRED_TEMPLATE,
                region=required_region,
                threshold=0.78,
                timeout=5.0,
            )
            if not gone:
                self.report_failure(
                    "upgrade_not_started",
                    "Панель «Требуется» осталась после подтверждения — улучшение не началось",
                )
                logger.warning("[BuildingUpgrade] Панель не закрылась — улучшение не началось.")
                return False
            return True

        time.sleep(1.5)
        after = self.client.get_screenshot(fresh=True)
        if after is not None:
            still = self.vision.find_template(
                after, self.REQUIRED_TEMPLATE, threshold=0.78, region=required_region
            )
            if still:
                logger.warning("[BuildingUpgrade] Панель не закрылась — улучшение не началось.")
                return False
        return True
