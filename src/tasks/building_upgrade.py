"""
Скрипт / Задача автоматического улучшения зданий (BuildingUpgradeTask).
Выполняет полную последовательность шагов:
1. Проверка строителей (свободен ли строитель).
2. Нажатие на целевое здание (или молот строителя в HUD).
3. Нажатие на кнопку 'Улучшить' в радиальном меню.
4. Проверка требований и нажатие 'Бесплатно' / 'Улучшить'.
5. Бесплатное ускорение (если таймер < 5 минут).
6. Закрытие экрана поздравления / баннера завершения.
"""

import time
from typing import Optional, Tuple
from loguru import logger
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from src.adb_client import AdbClient
from src.tasks.base_task import BaseTask
from src.vision import Vision


class BuildingUpgradeTask(BaseTask):
    def __init__(self, client: AdbClient, vision: Vision, interval_sec: int = 15):
        super().__init__("BuildingUpgrade", client, vision, interval_sec=interval_sec)
        self.ocr = RapidOCR()
        # Известные откалиброванные координаты ключевых зданий (1080x2436)
        self.building_coords = {
            "castle": (540, 1100),
            "wall": (510, 1120),
            "barracks": (850, 1600),
            "radial_upgrade_btn": (376, 1358),
            "modal_free_btn": (304, 2339),
            "modal_upgrade_btn": (794, 2338),
            "builder_idle_hud": (75, 402),
            "quest_hammer_hud": (70, 565),
        }

    def can_run(self) -> bool:
        return super().can_run()

    def execute(self, screen: np.ndarray) -> bool:
        """
        Основная точка входа задачи улучшения.
        Возвращает True, если действие по улучшению было успешно запущено.
        """
        self.mark_executed()
        logger.info("[BuildingUpgrade] Запуск цикла проверки улучшений...")
        return self.upgrade_via_hud_or_castle(screen)

    def upgrade_via_hud_or_castle(self, screen: np.ndarray) -> bool:
        """
        Универсальный сценарий:
        Пытается улучшить здание через левую иконку свободного строителя/квеста,
        либо напрямую кликая по Замку / Стене.
        """
        # Шаг 1: Проверяем, свободен ли молот строителя на левой панели HUD
        # Иконка молота строителя находится на (75, 402)
        match_idle = self.vision.find_template(screen, "btn_builder_idle", threshold=0.65)
        if match_idle:
            bx, by, _ = match_idle
            logger.info(f"[BuildingUpgrade] Найден свободный строитель на ({bx}, {by}). Кликаем...")
            self.client.tap(bx, by)
            time.sleep(1.5)
            return self._handle_upgrade_flow()

        # Если явного молота нет, проверяем молот квестовой подсказки (70, 565)
        match_hammer = self.vision.find_template(screen, "btn_quest_hammer", threshold=0.65)
        if match_hammer:
            hx, hy, _ = match_hammer
            logger.info(f"[BuildingUpgrade] Найдена иконка рекомендуемого улучшения на ({hx}, {hy}). Кликаем...")
            self.client.tap(hx, hy)
            time.sleep(1.5)
            return self._handle_upgrade_flow()

        return False

    def upgrade_specific_building(self, building_name: str) -> bool:
        """
        Детерминированная последовательность шагов улучшения конкретного здания:
        Клик по зданию -> Клик 'Улучшить' в радиальном меню -> Клик 'Улучшить' в модалке.
        """
        coords = self.building_coords.get(building_name)
        if not coords:
            logger.error(f"[BuildingUpgrade] Неизвестное здание: {building_name}")
            return False

        logger.info(f"[BuildingUpgrade] Последовательность: улучшение '{building_name}' по координатам {coords}")

        # Шаг 1: Клик по зданию для вызова радиального меню
        self.client.tap(coords[0], coords[1])
        time.sleep(1.5)

        return self._handle_upgrade_flow()

    def _handle_upgrade_flow(self) -> bool:
        """
        Автономный пошаговый автомат обработки экранов улучшения:
        Радиальное меню -> Модалка требований -> Кнопка старта -> Ускорение.
        """
        screen = self.client.get_screenshot()
        if screen is None:
            return False

        # Шаг 2: Нажимаем кнопку 'Улучшить' в радиальном меню здания
        # Откалиброванная координата центра круглой кнопки: (376, 1358)
        radial_match = self.vision.find_template(screen, "crop_btn_building_upgrade", threshold=0.65)
        if radial_match:
            rx, ry, conf = radial_match
            logger.info(f"[BuildingUpgrade] Найдена кнопка радиального меню 'Улучшить': ({rx}, {ry}), conf={conf:.2f}")
            self.client.tap(rx, ry)
            time.sleep(1.8)
        else:
            logger.info("[BuildingUpgrade] Клик по откалиброванной кнопке 'Улучшить' радиального меню (376, 1358)...")
            self.client.tap(376, 1358)
            time.sleep(1.8)

        # Шаг 3: Модальное окно требований улучшения
        screen = self.client.get_screenshot()
        if screen is None:
            return False

        # Ищем зеленую кнопку 'Бесплатно' или синюю кнопку 'Улучшить' внизу экрана (y: 2200-2400)
        ocr_res, _ = self.ocr(screen)
        if ocr_res:
            for box, text, score in ocr_res:
                txt = text.lower()
                cx = int((box[0][0] + box[1][0]) / 2)
                cy = int((box[0][1] + box[2][1]) / 2)

                if cy > 2100:  # Нижняя панель кнопок
                    if "бесплат" in txt or "becna" in txt:
                        logger.success(f"[BuildingUpgrade] Нажата кнопка 'Бесплатно': ({cx}, {cy}) [{text}]")
                        self.client.tap(cx, cy)
                        time.sleep(2.0)
                        self._handle_post_upgrade()
                        return True
                    elif "улучш" in txt or "octpo" in txt:
                        logger.success(f"[BuildingUpgrade] Нажата кнопка 'Улучшить': ({cx}, {cy}) [{text}]")
                        self.client.tap(cx, cy)
                        time.sleep(2.0)
                        self._handle_post_upgrade()
                        return True

        # Fallback: стандартные координаты кнопок модалки
        # Правая кнопка (Улучшить): (795, 2340)
        logger.info("[BuildingUpgrade] Fallback: нажимаем правую кнопку улучшения (795, 2340)...")
        self.client.tap(795, 2340)
        time.sleep(2.0)
        self._handle_post_upgrade()
        return True

    def _handle_post_upgrade(self) -> None:
        """
        Действия после запуска стройки:
        - Бесплатное завершение (если доступно).
        - Закрытие баннеров 'Уровень повышен' / 'Улучшено'.
        """
        time.sleep(1.5)
        screen = self.client.get_screenshot()
        if screen is None:
            return

        ocr_res, _ = self.ocr(screen)
        ocr_text = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""

        # Если выскочило окно "Уровень повышен!" или "Улучшено!"
        if "повышен" in ocr_text or "улучшено" in ocr_text:
            logger.info("[BuildingUpgrade] Закрываем поздравительный баннер кликом вне окна (540, 2000)...")
            self.client.tap(540, 2000)
            time.sleep(1.5)
            # Если следом идет диалог Беллы - пропускаем
            self.client.tap(968, 160)  # Кнопка 'Пропустить'
            time.sleep(1.0)
