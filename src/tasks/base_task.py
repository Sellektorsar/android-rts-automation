"""
Базовый класс для всех задач автоматизации (BaseTask).
Каждая рутинная задача (сбор, помощь, найм, марши) наследуется от этого класса.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.vision import Vision


class BaseTask(ABC):
    def __init__(self, name: str, client: AdbClient, vision: Vision, interval_sec: int = 60):
        self.name = name
        self.client = client
        self.vision = vision
        self.interval_sec = interval_sec
        self.last_run_time: float = 0.0
        self.enabled: bool = True

    def can_run(self) -> bool:
        """Проверка готовности задачи к выполнению по интервалу времени."""
        if not self.enabled:
            return False
        return (time.time() - self.last_run_time) >= self.interval_sec

    def mark_executed(self) -> None:
        """Обновление времени последнего выполнения."""
        self.last_run_time = time.time()

    @abstractmethod
    def execute(self, screen: np.ndarray) -> bool:
        """
        Логика выполнения задачи.
        :param screen: Текущий кадр экрана.
        :return: True, если действие успешно выполнено, иначе False.
        """
        pass
