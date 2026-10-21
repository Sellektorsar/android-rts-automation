"""
Задача взаимодействия с альянсом (AllianceTask).
Автоматическая помощь («Помочь всем»), сбор подарков альянса и технологические взносы.
"""

import numpy as np
from loguru import logger

from src.human_input import sleep_human
from src.tasks.base_task import BaseTask


class AllianceTask(BaseTask):
    def __init__(self, client, vision, interval_sec: int = 60):
        super().__init__("Alliance", client, vision, interval_sec)

    def execute(self, screen: np.ndarray) -> bool:
        acted = False

        # 1. Проверяем наличие плавающей иконки помощи альянсу (рукопожатие над замком/посольством)
        help_bubble = self.vision.find_template(screen, "bubble_alliance_help", threshold=0.8)
        if help_bubble:
            cx, cy, conf = help_bubble
            logger.info(f"Найдена иконка помощи альянсу ({conf:.2f}). Нажимаем...")
            self.client.tap(cx, cy)
            sleep_human(0.8, 1.5)
            acted = True

        # 2. Если открыто окно помощи альянсу, ищем кнопку 'Помочь всем' (Help All)
        btn_help_all = self.vision.find_template(screen, "btn_alliance_help_all", threshold=0.82)
        if btn_help_all:
            cx, cy, conf = btn_help_all
            logger.info(f"Найдена кнопка 'Помочь всем' ({conf:.2f}). Нажимаем...")
            self.client.tap(cx, cy)
            sleep_human(0.8, 1.2)
            acted = True

        # 3. Кнопка сбора всех подарков альянса
        btn_claim_gifts = self.vision.find_template(screen, "btn_claim_all_gifts", threshold=0.82)
        if btn_claim_gifts:
            cx, cy, conf = btn_claim_gifts
            logger.info(f"Найдена кнопка 'Собрать все подарки' ({conf:.2f}). Забираем...")
            self.client.tap(cx, cy)
            sleep_human(0.8, 1.5)
            acted = True

        if acted:
            self.mark_executed()
        return acted
