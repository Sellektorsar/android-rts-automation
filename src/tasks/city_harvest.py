"""
Задача сбора ресурсов внутри замка (CityHarvestTask).
Поиск и нажатие на плавающие значки ресурсов над фермами,
лесопилками, каменоломнями и железными рудниками.
"""

import numpy as np
from loguru import logger

from src.human_input import sleep_human
from src.tasks.base_task import BaseTask


class CityHarvestTask(BaseTask):
    def __init__(self, client, vision, interval_sec: int = 180):
        super().__init__("CityHarvest", client, vision, interval_sec)
        self.resource_templates = [
            "bubble_grain",
            "bubble_wood",
            "bubble_stone",
            "bubble_iron",
            "bubble_wishing_well",
        ]

    def execute(self, screen: np.ndarray) -> bool:
        collected = False

        for res_tmpl in self.resource_templates:
            bubbles = self.vision.find_all(screen, res_tmpl, threshold=0.70, min_distance=35)
            if bubbles:
                logger.info(f"Найдено {len(bubbles)} значков сбора '{res_tmpl}'")
                for cx, cy, conf in bubbles:
                    self.client.tap(cx, cy)
                    sleep_human(0.3, 0.7)
                    collected = True

        if collected:
            self.mark_executed()
            sleep_human(1.0, 1.8)

        return collected
