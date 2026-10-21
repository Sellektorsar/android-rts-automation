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
    def __init__(self, client, vision, interval_sec: int = 180, reflection=None):
        super().__init__("CityHarvest", client, vision, interval_sec, reflection=reflection)
        self.resource_templates = [
            "bubble_grain",
            "bubble_wood",
            "bubble_stone",
            "bubble_iron",
            "bubble_wishing_well",
        ]

    def execute(self, screen: np.ndarray) -> bool:
        available = [name for name in self.resource_templates if self.vision.has_template(name)]
        missing = [name for name in self.resource_templates if name not in available]
        for name in missing:
            if name not in self._missing_templates_reported:
                logger.warning(f"[{self.name}] Шаблон '{name}' отсутствует в assets — этот ресурс не собирается")
                self._missing_templates_reported.add(name)
        if not available:
            return False

        height, width = screen.shape[:2]
        # Игровое поле: исключаем верхний HUD и нижнюю навигацию, чтобы
        # случайное совпадение не привело к тапу по интерфейсу.
        playfield = (
            int(width * 0.03),
            int(height * 0.12),
            int(width * 0.94),
            int(height * 0.76),
        )

        collected = False
        taps = 0

        for res_tmpl in available:
            bubbles = self.vision.find_all(
                screen, res_tmpl, threshold=0.78, min_distance=40, region=playfield
            )
            if bubbles:
                logger.info(f"Найдено {len(bubbles)} значков сбора '{res_tmpl}'")
                for cx, cy, conf in bubbles:
                    self.client.tap(cx, cy)
                    sleep_human(0.3, 0.7)
                    collected = True
                    taps += 1
                    if taps >= 12:
                        break
            if taps >= 12:
                break

        if collected:
            self.mark_executed()
            sleep_human(1.0, 1.8)

        return collected
