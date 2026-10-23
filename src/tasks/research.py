"""""Автопилот науки: старт исследования в меню Институт."""""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from src.fsm import GameState
from src.perception import ScreenClassifier
from src.tasks.base_task import BaseTask


class ResearchTask(BaseTask):
    """Автопилот исследования (дерево технологий / Институт).

    Сценарий: находим стартовое исследование в меню науки и запускаем его
    через проверенный шаблон. Не тратит премиум и не покупает ускорения
    автоматически — это решает риск-гейт владельца.
    """

    RESEARCH_START_TEMPLATE = "btn_research_start"

    def __init__(
        self,
        client,
        vision,
        interval_sec: int = 120,
        reflection=None,
    ):
        super().__init__(
            "ResearchAuto",
            client,
            vision,
            interval_sec=interval_sec,
            reflection=reflection,
        )
        self.classifier = ScreenClassifier(vision, allow_ocr=False)

    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates([self.RESEARCH_START_TEMPLATE]):
            return False
        screen = self.ensure_screen(screen)
        if screen is None:
            return False
        state = self.classifier.classify(screen, allow_ocr=False).state
        if state != GameState.RESEARCH_MENU:
            return False
        match = self.vision.find_template(screen, self.RESEARCH_START_TEMPLATE, threshold=0.85)
        if match is None:
            return False
        x, y, confidence = int(match[0]), int(match[1]), float(match[2])
        logger.info(f"[ResearchAuto] Старт исследования ({x}, {y}), conf={confidence:.2f}.")
        self.client.tap(x, y)
        self.mark_executed()
        return True
