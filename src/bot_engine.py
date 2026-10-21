"""
Главный движок бота (BotEngine).
Оркестрирует цикл работы: получение кадра, определение состояния,
вызов активных задач по расписанию и контроль активности игры.
"""

import time
from pathlib import Path
from typing import List
import yaml
from loguru import logger

from src.adb_client import AdbClient
from src.human_input import sleep_human
from src.tasks.alliance import AllianceTask
from src.tasks.base_task import BaseTask
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.tasks.chapter_quest import ChapterQuestTask
from src.tasks.city_harvest import CityHarvestTask
from src.tasks.popup_closer import PopupCloserTask
from src.tasks.troop_training import TroopTrainingTask
from src.vision import Vision


class BotEngine:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        dev_cfg = self.config["device"]
        self.client = AdbClient(
            adb_path=dev_cfg["adb_path"],
            ip=dev_cfg["ip"],
            port=dev_cfg["port"],
            timeout=dev_cfg.get("timeout_sec", 10),
        )

        self.vision = Vision(templates_dir="assets")
        self.tasks: List[BaseTask] = []
        self._init_tasks()
        self.running = False

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_tasks(self) -> None:
        t_cfg = self.config.get("tasks", {})

        # Задача 1: Закрытие всплывающих окон (всегда в наивысшем приоритете)
        if t_cfg.get("popup_closer", {}).get("enabled", True):
            interval = t_cfg.get("popup_closer", {}).get("interval_sec", 5)
            self.tasks.append(PopupCloserTask(self.client, self.vision, interval_sec=interval))

        # Задача 2: Помощь альянсу
        if t_cfg.get("alliance_help", {}).get("enabled", True):
            interval = t_cfg.get("alliance_help", {}).get("interval_sec", 60)
            self.tasks.append(AllianceTask(self.client, self.vision, interval_sec=interval))

        # Задача 3: Сюжетные квесты (Королевская Дорога)
        if t_cfg.get("chapter_quest", {}).get("enabled", True):
            interval = t_cfg.get("chapter_quest", {}).get("interval_sec", 15)
            self.tasks.append(ChapterQuestTask(self.client, self.vision, interval_sec=interval))

        # Задача 4: Автоматическое улучшение зданий
        if t_cfg.get("building_upgrade", {}).get("enabled", True):
            interval = t_cfg.get("building_upgrade", {}).get("interval_sec", 20)
            self.tasks.append(BuildingUpgradeTask(self.client, self.vision, interval_sec=interval))

        # Задача 5: Найм войск в казарме
        if t_cfg.get("barracks_train", {}).get("enabled", True):
            interval = t_cfg.get("barracks_train", {}).get("interval_sec", 120)
            self.tasks.append(TroopTrainingTask(self.client, self.vision, interval_sec=interval))

        # Задача 6: Сбор ресурсов в городе
        if t_cfg.get("city_harvest", {}).get("enabled", True):
            interval = t_cfg.get("city_harvest", {}).get("interval_sec", 180)
            self.tasks.append(CityHarvestTask(self.client, self.vision, interval_sec=interval))

    def start(self) -> None:
        logger.info("Запуск бота War & Order...")
        if not self.client.connect():
            logger.error("Критическая ошибка: невозможно подключиться к устройству.")
            return

        pkg = self.config["game"]["package"]
        act = self.config["game"]["activity"]

        # Проверка и запуск игры
        if not self.client.is_game_foreground(pkg):
            logger.info("Игра не запущена на переднем плане. Запускаем...")
            self.client.launch_game(pkg, act)
            sleep_human(8.0, 12.0)  # Ожидание загрузки движка Unity

        self.running = True
        logger.success("Бот успешно запущен в фоновом цикле!")

        try:
            while self.running:
                # 1. Проверяем, активна ли игра
                if not self.client.is_game_foreground(pkg):
                    logger.warning("Игра свернута или упала. Перезапуск...")
                    self.client.launch_game(pkg, act)
                    sleep_human(6.0, 10.0)
                    continue

                # 2. Захват текущего кадра
                screen = self.client.get_screenshot()
                if screen is None:
                    logger.warning("Не удалось получить кадр экрана. Пауза...")
                    time.sleep(2)
                    continue

                # 3. Выполнение задач по очереди приоритетов
                for task in self.tasks:
                    if task.can_run():
                        did_action = task.execute(screen)
                        if did_action:
                            # Если действие выполнено, обновляем снимок для следующей задачи
                            screen = self.client.get_screenshot()
                            if screen is None:
                                break

                # 4. Пауза между итерациями
                sleep_human(1.5, 3.0)

        except KeyboardInterrupt:
            logger.info("Остановка бота пользователем (Ctrl+C).")
        finally:
            self.running = False
            logger.info("Работа бота завершена.")
