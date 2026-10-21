"""
Задача улучшения зданий (BuildingUpgradeTask).

Проверенный на живом устройстве сценарий:
1. В городе тапнуть по зданию (по умолчанию — Стена Замка).
2. В радиальном меню найти кнопку «Улучшить» (три молотка).
3. На панели требований найти блок «Требуется».
4. Нажать ТОЛЬКО синюю кнопку «Улучшить» в правом нижнем углу.

Зелёная кнопка с гемами («Ускорить») не нажимается никогда: поиск
подтверждения ограничен правой половиной нижней полосы. Если синей
кнопки нет (не хватает ресурсов/уровня замка), задача просто выходит,
а навигация возвращает бота в город по стрелке назад.

Проверки взяты из esp32-android-agent: после каждого шага не фиксированная
пауза, а ожидание конкретного шаблона. Если панель требований не появилась
или не исчезла после подтверждения — это фиксируется как сбой, а не успех.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.tasks.base_task import BaseTask
from src.vision import Vision


class BuildingUpgradeTask(BaseTask):
    RADIAL_TEMPLATE = "btn_building_upgrade"
    CONFIRM_TEMPLATE = "btn_upgrade_confirm"
    REQUIRED_TEMPLATE = "title_upgrade_required"

    # Точки зданий в городе при камере по умолчанию. Тап по точке только
    # выделяет здание; улучшение начинается лишь после проверки шаблонов.
    DEFAULT_SPOTS: Tuple[Tuple[int, int], ...] = ((639, 1500),)

    def __init__(
        self,
        client: AdbClient,
        vision: Vision,
        interval_sec: int = 60,
        spots: Optional[Iterable[Sequence[int]]] = None,
        reflection=None,
    ):
        super().__init__(
            "BuildingUpgrade", client, vision, interval_sec=interval_sec, reflection=reflection
        )
        parsed = [tuple(int(v) for v in spot[:2]) for spot in (spots or self.DEFAULT_SPOTS)]
        self.spots: Tuple[Tuple[int, int], ...] = tuple(parsed) or self.DEFAULT_SPOTS
        self._spot_index = 0

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------
    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates(
            [self.RADIAL_TEMPLATE, self.CONFIRM_TEMPLATE, self.REQUIRED_TEMPLATE]
        ):
            return False

        if self.guard is not None and not self.guard.require_actionable(screen):
            return False

        # Уже на панели требований (например, после ручного открытия).
        if self._tap_confirm(screen):
            self.mark_executed()
            return True

        # Уже открыто радиальное меню здания.
        if self._tap_radial(screen):
            return self._confirm_panel(screen)

        # Город: выделяем следующее здание из списка и проверяем меню.
        spot = self.spots[self._spot_index % len(self.spots)]
        self._spot_index += 1
        logger.info(f"[BuildingUpgrade] Проверяем здание в точке {spot}.")
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
            return self._confirm_panel(radial.after or screen)

        time.sleep(1.2)
        selected = self.client.get_screenshot(fresh=True)
        if selected is None:
            self.report_failure("no_frame_after_building_tap", "Нет кадра после выбора здания")
            return False
        if not self._tap_radial(selected):
            logger.info("[BuildingUpgrade] Кнопка «Улучшить» не появилась — здание занято или недоступно.")
            return False
        return self._confirm_panel(selected)

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

    # ------------------------------------------------------------------
    # Шаги
    # ------------------------------------------------------------------
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
        elif self.client.screen_size:
            width, height = self.client.screen_size
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

    def _confirm_panel(self, reference: np.ndarray) -> bool:
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
        if not self._tap_confirm(panel):
            logger.info("[BuildingUpgrade] Синяя кнопка «Улучшить» недоступна — отмена.")
            return False
        self.mark_executed()
        return True

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
