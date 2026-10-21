"""
Сбор подарков торговца в городе (CityGiftsTask).

Над повозкой торговца периодически появляется подарок с зелёной галочкой.
Тап по нему открывает баннер «Вы получили награду!». Баннер закрывается
тапом по свободному полю ниже окна — это проверено на живом устройстве.

Задача ищет только конкретный шаблон подарка в игровом поле, поэтому
не трогает квестовые иконки и кнопки оплаты.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from src.human_input import sleep_human
from src.tasks.base_task import BaseTask


class CityGiftsTask(BaseTask):
    GIFT_TEMPLATE = "icon_gift_ready"
    REWARD_TEMPLATE = "banner_reward_received"

    def __init__(self, client, vision, interval_sec: int = 900, reflection=None):
        super().__init__(
            "CityGifts", client, vision, interval_sec=interval_sec, reflection=reflection
        )

    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates([self.GIFT_TEMPLATE]):
            return False
        if self.guard is not None and not self.guard.require_actionable(screen):
            return False

        height, width = screen.shape[:2]
        # Только игровое поле: без верхнего HUD и нижней навигации.
        region = (
            int(width * 0.03),
            int(height * 0.18),
            int(width * 0.94),
            int(height * 0.70),
        )
        match = self.vision.find_template(
            screen, self.GIFT_TEMPLATE, threshold=0.82, region=region
        )
        if not match:
            return False

        x, y, confidence = match
        logger.info(f"[CityGifts] Подарок торговца найден ({x}, {y}), conf={confidence:.2f}.")
        self.client.tap(x, y)

        if self.guard is not None:
            # Ждём именно баннер награды, а не просто «что-то изменилось».
            if self.vision.has_template(self.REWARD_TEMPLATE):
                reward = self.guard.wait_for_match(
                    self.REWARD_TEMPLATE, threshold=0.80, timeout=4.0
                )
                if reward.ok:
                    logger.info("[CityGifts] Награда получена, закрываем баннер тапом по полю.")
                    self.client.tap(int(width * 0.50), int(height * 0.82))
                    sleep_human(0.5, 0.9)
                    self.mark_executed()
                    return True

            # Баннер мог быть другим: подтверждаем хотя бы реакцию экрана.
            after, changed = self.guard.wait_for_change(
                screen, region=region, timeout=3.0
            )
            if not changed:
                self.report_failure(
                    "gift_no_reaction",
                    "После тапа по подарку экран не изменился",
                )
                logger.warning("[CityGifts] Подарок не отреагировал на тап — пропускаем.")
                return False

            logger.info("[CityGifts] Экран изменился, закрываем возможный баннер тапом по полю.")
            self.client.tap(int(width * 0.50), int(height * 0.82))
            sleep_human(0.5, 0.9)
            self.mark_executed()
            return True

        sleep_human(1.2, 1.8)
        after = self.client.get_screenshot(fresh=True)
        if after is None:
            return False

        if self.vision.has_template(self.REWARD_TEMPLATE) and self.vision.find_template(
            after, self.REWARD_TEMPLATE, threshold=0.80
        ):
            logger.info("[CityGifts] Награда получена, закрываем баннер тапом по полю.")
            self.client.tap(int(width * 0.50), int(height * 0.82))
            sleep_human(0.5, 0.9)

        self.mark_executed()
        return True
