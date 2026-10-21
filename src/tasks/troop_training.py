"""
Скрипт / Задача автоматического найма войск (TroopTrainingTask).
Выполняет цикл:
1. Клик по Казарме (или через квестовую стрелку).
2. Выбор кнопки 'Найм' в радиальном меню.
3. Выбор типа войск (например, Лучники / Пехота).
4. Установка максимального количества.
5. Нажатие кнопки 'Нанять' (Train).
6. Бесплатное ускорение (если доступно).
"""

import time
from typing import Optional
from loguru import logger
import numpy as np

from src.adb_client import AdbClient
from src.tasks.base_task import BaseTask
from src.vision import Vision


class TroopTrainingTask(BaseTask):
    def __init__(self, client: AdbClient, vision: Vision, interval_sec: int = 120, reflection=None):
        super().__init__(
            "TroopTraining", client, vision, interval_sec=interval_sec, reflection=reflection
        )
        self.barracks_coords = (850, 1600)
        self.radial_recruit_coords = (690, 1260)
        self.tabs = {
            "infantry": (152, 329),
            "cavalry": (424, 330),
            "archers": (689, 329),
            "mages": (960, 329),
        }
        self.btn_train = (845, 2337)
        self._not_implemented_warned = False

    def can_run(self) -> bool:
        return super().can_run()

    def execute(self, screen: np.ndarray) -> bool:
        """Пока не реализовано: нет надёжных шаблонов казармы и окна найма."""
        if not self._not_implemented_warned:
            logger.warning("[TroopTraining] Задача не реализована (нужны шаблоны казармы и окна найма).")
            self._not_implemented_warned = True
        return False

    def train_archers(self, count: int = 30) -> bool:
        """
        Последовательность найма лучников:
        Клик по казарме -> Найм -> Выбор лучников -> Нанять.
        """
        logger.info(f"[TroopTraining] Запуск последовательности найма {count} лучников...")
        # 1. Клик по казарме
        self.client.tap(self.barracks_coords[0], self.barracks_coords[1])
        time.sleep(1.5)

        screen = self.client.get_screenshot()
        if screen is None:
            return False

        # 2. Нажимаем кнопку 'Найм' в радиальном меню казармы
        logger.info("[TroopTraining] Клик по кнопке радиального меню 'Найм' (690, 1260)...")
        self.client.tap(self.radial_recruit_coords[0], self.radial_recruit_coords[1])
        time.sleep(2.0)

        # 3. В окне найма выбираем вкладку Лучников (689, 329)
        logger.info("[TroopTraining] Выбор вкладки Лучников (689, 329)...")
        self.client.tap(self.tabs["archers"][0], self.tabs["archers"][1])
        time.sleep(1.2)

        # 4. Нажимаем кнопку 'Нанять' (845, 2337)
        logger.info(f"[TroopTraining] Нажатие кнопки 'Найм' ({self.btn_train})...")
        self.client.tap(self.btn_train[0], self.btn_train[1])
        time.sleep(2.0)

        # 5. Возврат на экран города
        self.client.press_back()
        time.sleep(1.2)
        return True
