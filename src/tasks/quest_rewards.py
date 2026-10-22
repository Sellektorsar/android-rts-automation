"""Сбор наград ежедневных заданий (QuestRewardsTask).

Безопасный сценарий:
1. В городе найти нижнюю кнопку «Задание» по шаблону `btn_nav_quests`.
2. Открыть экран ежедневных заданий и подтвердить якорь
   `title_daily_quests`.
3. Найти зелёные кнопки «Забрать награды» (`btn_claim_green`) в правой
   колонке списка.
4. Нажать одну верхнюю кнопку, дождаться реакции и вернуться в город.

Задача не трогает кнопки покупки, золотой молот и премиальную валюту:
её шаблон — только зелёная кнопка получения бесплатной награды.
"""

from __future__ import annotations

import time

import numpy as np
from loguru import logger

from src.fsm import GameState
from src.human_input import sleep_human
from src.perception import ScreenClassifier
from src.tasks.base_task import BaseTask


class QuestRewardsTask(BaseTask):
    NAV_TEMPLATE = "btn_nav_quests"
    TITLE_TEMPLATE = "title_daily_quests"
    CLAIM_TEMPLATE = "btn_claim_green"

    def __init__(self, client, vision, interval_sec: int = 45, reflection=None):
        super().__init__(
            "QuestRewards", client, vision, interval_sec=interval_sec, reflection=reflection
        )
        # Только якоря, без OCR: классификация дешёвая и не мешает циклу.
        self.classifier = ScreenClassifier(vision, allow_ocr=False)

    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates(
            [self.NAV_TEMPLATE, self.TITLE_TEMPLATE, self.CLAIM_TEMPLATE]
        ):
            return False
        screen = self.ensure_screen(screen)
        if screen is None:
            return False

        state = self.classifier.classify(screen, allow_ocr=False).state
        opened_here = False
        if state == GameState.CITY_VIEW:
            nav = self.vision.find_template(
                screen,
                self.NAV_TEMPLATE,
                threshold=0.82,
                region=(0, int(screen.shape[0] * 0.84), screen.shape[1], int(screen.shape[0] * 0.16)),
            )
            if nav is None:
                return False
            logger.info("[QuestRewards] Открываем ежедневные задания.")
            self.client.tap(nav[0], nav[1])
            sleep_human(1.6, 2.2)
            screen = self.client.get_screenshot(fresh=True)
            if screen is None:
                return False
            opened_here = True
            state = self.classifier.classify(screen, allow_ocr=False).state

        if state != GameState.QUESTS_MENU:
            if opened_here:
                self._back_to_city()
            return False

        if not self.vision.find_template(
            screen, self.TITLE_TEMPLATE, threshold=0.85,
            region=(int(screen.shape[1] * 0.35), int(screen.shape[0] * 0.04),
                    int(screen.shape[1] * 0.30), int(screen.shape[0] * 0.06)),
        ):
            return False

        height, width = screen.shape[:2]
        matches = self.vision.find_all(
            screen,
            self.CLAIM_TEMPLATE,
            threshold=0.90,
            min_distance=70,
            region=(
                int(width * 0.72),
                int(height * 0.20),
                int(width * 0.24),
                int(height * 0.50),
            ),
            scale=0.5,
        )
        if not matches:
            logger.info("[QuestRewards] Зелёных «Забрать награды» нет.")
            self._back_to_city()
            return False

        matches.sort(key=lambda item: item[1])
        x, y, confidence = matches[0]
        logger.info(
            f"[QuestRewards] Награда найдена ({x}, {y}), conf={confidence:.2f}; "
            f"всего кнопок: {len(matches)}."
        )
        self.client.tap(x, y)
        sleep_human(1.2, 1.8)
        self._back_to_city()
        self.mark_executed()
        return True

    def _back_to_city(self) -> None:
        """До четырёх Back: тост «Уровень повышен» может съесть одно нажатие."""
        for _ in range(4):
            frame = self.client.get_screenshot(fresh=True)
            if frame is None:
                return
            state = self.classifier.classify(frame, allow_ocr=False).state
            if state == GameState.CITY_VIEW:
                return
            self.client.press_back()
            time.sleep(1.4)
