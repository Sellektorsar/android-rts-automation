"""
Базовый класс для всех задач автоматизации (BaseTask).
Каждая рутинная задача (сбор, помощь, найм, марши) наследуется от этого класса.

Здесь же собраны общие предохранители:
* троттлинг по времени последней попытки;
* проверка свежести кадра и стабильности экрана (ActionGuard);
* запись успехов/сбоев в ReflectionLog и адаптивное увеличение интервала,
  если задача повторяет одну и ту же ошибку.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from loguru import logger

from src.action_guard import ActionGuard
from src.adb_client import AdbClient
from src.reflection import ReflectionLog
from src.vision import Vision


class BaseTask(ABC):
    def __init__(
        self,
        name: str,
        client: AdbClient,
        vision: Vision,
        interval_sec: int = 60,
        reflection: Optional[ReflectionLog] = None,
    ):
        self.name = name
        self.client = client
        self.vision = vision
        self.reflection = reflection
        self.interval_sec = interval_sec
        self.last_run_time: float = 0.0
        # Время последней попытки (успешной или нет). Без него задача,
        # которая не нашла условий, запускалась бы на каждом такте.
        self.last_attempt_time: float = 0.0
        self.enabled: bool = True
        self._missing_templates_reported: set[str] = set()
        self._success_reported_for_attempt: float = 0.0
        self._failure_reported_for_attempt: float = 0.0
        self.last_failure_reason: str = ""

        # Проверки вокруг действия не нужны в юнит-тестах, где client=None.
        self.guard: Optional[ActionGuard] = (
            ActionGuard(client, vision) if client is not None else None
        )

    def current_interval(self) -> float:
        """Интервал с учётом истории сбоев задачи.

        Идея из 4x-game-agent: неудачные попытки должны не просто
        повторяться по таймеру, а постепенно замедляться. Это особенно
        важно, когда экран изменился после обновления игры или шаблон
        устарел — бесконечные повторы только шумят в логах.
        """
        base = float(self.interval_sec)
        if self.reflection is None:
            return base

        stats = self.reflection.task_stats(self.name, window=20)
        if stats["total"] < 4 or stats["failures"] == 0:
            return base

        rate = stats["rate"]
        if rate < 0.25:
            return min(base * 3.0, base + 900.0)
        if rate < 0.50:
            return min(base * 1.75, base + 600.0)
        return base

    def can_run(self) -> bool:
        """Проверка готовности задачи к выполнению по интервалу времени."""
        if not self.enabled:
            return False
        now = time.time()
        last_activity = max(self.last_run_time, self.last_attempt_time)
        if now - last_activity < self.current_interval():
            return False
        self.last_attempt_time = now
        return True

    def require_templates(self, names: list[str]) -> bool:
        """Проверяет наличие шаблонов и один раз предупреждает об отсутствующих."""
        missing = [name for name in names if not self.vision.has_template(name)]
        for name in missing:
            if name not in self._missing_templates_reported:
                logger.warning(f"[{self.name}] Шаблон '{name}' отсутствует в assets — задача пропущена")
                self._missing_templates_reported.add(name)
        return not missing

    def mark_executed(self) -> None:
        """Обновление времени последнего выполнения и запись успеха."""
        self.last_run_time = time.time()
        self._record_success()

    def report_failure(self, reason: str, analysis: str = "") -> None:
        """Явный сбой задачи (не «условий пока нет», а реальная ошибка)."""
        self.last_failure_reason = reason
        if self.reflection is None:
            return
        if self._failure_reported_for_attempt == self.last_attempt_time:
            return
        self._failure_reported_for_attempt = self.last_attempt_time
        self.reflection.record_failure(
            task=self.name,
            error=reason,
            analysis=analysis or reason,
        )

    def _record_success(self) -> None:
        if self.reflection is None:
            return
        if self._success_reported_for_attempt == self.last_attempt_time:
            return
        self._success_reported_for_attempt = self.last_attempt_time
        self.reflection.record_success(self.name)

    @abstractmethod
    def execute(self, screen: np.ndarray) -> bool:
        """
        Логика выполнения задачи.
        :param screen: Текущий кадр экрана.
        :return: True, если действие успешно выполнено, иначе False.
        """
        pass
