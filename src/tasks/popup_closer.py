"""
Задача закрытия всплывающих окон, акций и баннеров (PopupCloser).
"""

import numpy as np
from loguru import logger

from src.human_input import sleep_human
from src.tasks.base_task import BaseTask


class PopupCloserTask(BaseTask):
    def __init__(self, client, vision, interval_sec: int = 5, reflection=None):
        super().__init__("PopupCloser", client, vision, interval_sec, reflection=reflection)
        # Имена шаблонов кнопок закрытия
        self.close_button_templates = [
            "btn_close_x",
            "btn_close_round",
            "btn_close_gold",
        ]

    def execute(self, screen: np.ndarray) -> bool:
        # Работаем с теми шаблонами, которые уже есть: отсутствие одного
        # крестика не должно отключать всю задачу.
        available = [name for name in self.close_button_templates if self.vision.has_template(name)]
        for name in self.close_button_templates:
            if name not in available and name not in self._missing_templates_reported:
                logger.warning(f"[{self.name}] Шаблон '{name}' отсутствует в assets — пропускаем")
                self._missing_templates_reported.add(name)
        if not available:
            return False

        match = self.vision.find_best(screen, available, threshold=0.82)
        if not match:
            return False

        cx, cy, conf, tmpl = match
        logger.info(f"Обнаружено всплывающее окно ({tmpl}, точность {conf:.2f}). Закрываем...")
        self.client.tap(cx, cy)
        sleep_human(0.6, 1.2)
        self.mark_executed()
        return True
