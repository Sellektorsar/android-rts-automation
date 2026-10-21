"""
Задача закрытия всплывающих окон, акций и баннеров (PopupCloser).
"""

import numpy as np
from loguru import logger

from src.human_input import sleep_human
from src.tasks.base_task import BaseTask


class PopupCloserTask(BaseTask):
    def __init__(self, client, vision, interval_sec: int = 5):
        super().__init__("PopupCloser", client, vision, interval_sec)
        # Имена шаблонов кнопок закрытия
        self.close_button_templates = [
            "btn_close_x",
            "btn_close_round",
            "btn_close_gold",
        ]

    def execute(self, screen: np.ndarray) -> bool:
        closed_any = False
        for tmpl in self.close_button_templates:
            match = self.vision.find_template(screen, tmpl, threshold=0.82)
            if match:
                cx, cy, conf = match
                logger.info(f"Обнаружено всплывающее окно ({tmpl}, точность {conf:.2f}). Закрываем...")
                self.client.tap(cx, cy)
                sleep_human(0.6, 1.2)
                closed_any = True
                break

        if closed_any:
            self.mark_executed()
        return closed_any
