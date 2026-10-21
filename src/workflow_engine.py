"""
Движок верифицированных игровых сценариев (Workflow Engine).
Основан на архитектуре 4x-game-agent.
Выполняет атомарные последовательности действий с обязательной пред- и пост-проверкой,
интегрирует модель мира (World Model), конечный автомат (FSM), модуль рефлексии (ReflectionLog)
и неблокирующий анти-детект ввод (HumanInput + AdbClient).
"""

import time
from typing import Callable, Optional, Tuple
import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.fsm import GameFSM, GameState
from src.human_input import HumanInput
from src.reflection import ReflectionLog
from src.vision import Vision
from src.world_model import WorldModel


class WorkflowEngine:
    """Управление и исполнение верифицированных макросов."""

    def __init__(
        self,
        client: AdbClient,
        vision: Vision,
        fsm: GameFSM,
        world: WorldModel,
        reflection: ReflectionLog,
    ):
        self.client = client
        self.vision = vision
        self.fsm = fsm
        self.world = world
        self.reflection = reflection
        self.stats = {"executed": 0, "succeeded": 0, "failed": 0}

    def dismiss_popup_if_present(self) -> bool:
        """Поиск и закрытие модальных окон (крестик или кнопка закрытия)."""
        screen = self.client.get_screenshot()
        if screen is None:
            return False

        # 1. Поиск крестика закрытия по шаблону
        match = self.vision.find_template(screen, "btn_close_x", threshold=0.72)
        if match:
            cx, cy, conf = match
            logger.info(f"[Workflow] Обнаружен крестик закрытия модалки: ({cx}, {cy}), conf={conf:.2f}")
            self.client.tap(cx, cy)
            time.sleep(1.2)
            return True

        # 2. Если FSM сигнализирует о модалке, можно нажать системную кнопку Back или тап вне окна
        if self.fsm.is_popup():
            logger.info("[Workflow] Закрытие модального окна через Android Back...")
            self.client.press_back()
            time.sleep(1.5)
            return True

        return False

    def wait_until_screen_stable(self, timeout: float = 4.0, poll_interval: float = 0.4) -> bool:
        """Ожидание стабилизации экрана и завершения анимаций (паттерн android-harness)."""
        start = time.time()
        prev = self.client.get_screenshot()
        while time.time() - start < timeout:
            time.sleep(poll_interval)
            curr = self.client.get_screenshot()
            if prev is not None and curr is not None and Vision.is_screen_stable(prev, curr):
                logger.debug(f"[Workflow] Экран стабилизировался за {time.time() - start:.2f}с")
                return True
            prev = curr
        return False

    def wait_for_screen_change(self, baseline: Optional[np.ndarray], timeout: float = 3.0, min_diff: float = 0.02) -> bool:
        """Проверка, привел ли клик/действие к визуальному изменению экрана (паттерн android-harness)."""
        if baseline is None:
            return True
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(0.3)
            curr = self.client.get_screenshot()
            if curr is not None and Vision.diff_score(baseline, curr) >= min_diff:
                return True
        return False

    def run_step(self, name: str, action_fn: Callable[..., bool], *args, **kwargs) -> bool:
        """Выполнение сценария с рефлексией и проверкой исхода."""
        self.stats["executed"] += 1
        logger.info(f"[Workflow] Старт сценария: {name}")

        # Проверяем недавние сбои в журнале рефлексии
        recent_fails = self.reflection.get_recent_failures(name, limit=2)
        if recent_fails:
            logger.warning(f"[Workflow] Предупреждение: у задачи {name} были недавние сбои: {recent_fails[0].get('analysis')}")

        try:
            success = action_fn(*args, **kwargs)
            if success:
                self.stats["succeeded"] += 1
                self.reflection.record_success(name)
                logger.success(f"[Workflow] Сценарий {name} успешно завершен!")
                return True
            else:
                self.stats["failed"] += 1
                self.reflection.record_failure(
                    task=name,
                    error="Action returned False",
                    analysis="Пост-условие сценария не подтвердилось на экране.",
                )
                logger.error(f"[Workflow] Сценарий {name} завершился неудачно.")
                return False
        except Exception as e:
            self.stats["failed"] += 1
            self.reflection.record_failure(
                task=name,
                error=str(e),
                analysis=f"Исключение во время исполнения: {type(e).__name__}",
            )
            logger.exception(f"[Workflow] Ошибка исполнения сценария {name}: {e}")
            return False
