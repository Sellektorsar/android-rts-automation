"""
Задача прохождения сюжетных квестов (Королевская Дорога).

Цикл:
1. Проверка баннера квеста в левом нижнем углу HUD.
2. Сбор готовых наград.
3. Открытие Королевской Дороги по иконке короны.
4. Нажатие 'Вперёд' у первого активного задания.
5. Сбор награды за главу.

Задача использует общий OCR-слой и не создаёт собственный экземпляр
RapidOCR (иначе в памяти висело несколько моделей).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.ocr import read_text
from src.tasks.base_task import BaseTask
from src.vision import Vision


class ChapterQuestTask(BaseTask):
    def __init__(self, client: AdbClient, vision: Vision, interval_sec: int = 20, reflection=None):
        super().__init__(
            "ChapterQuest", client, vision, interval_sec=interval_sec, reflection=reflection
        )
        self.crown_coords = (75, 1960)
        self.banner_tap_coords = (250, 1955)

    def execute(self, screen: np.ndarray) -> bool:
        logger.info("[ChapterQuest] Проверка статуса сюжетных заданий...")
        height, _ = screen.shape[:2]
        ocr = read_text(screen, scale=0.5)

        quest_text = ""
        is_completed = False
        is_chapter_finished = False

        for item in ocr.items:
            text = item.text.lower()
            cy = item.y
            cx = item.x
            if int(height * 0.77) <= cy <= int(height * 0.83) and cx < int(screen.shape[1] * 0.56):
                quest_text += " " + text
                if any(marker in text for marker in ("1/1", "2/2", "3/3", "4/4", "5/5", "заберите", "награды")):
                    is_completed = True
                if "награды главы" in text or "заберите награды" in text:
                    is_chapter_finished = True

        if is_completed or is_chapter_finished:
            logger.success(f"[ChapterQuest] Награда готова к сбору: '{quest_text.strip()}'")
            self.client.tap(*self.banner_tap_coords)
            time.sleep(2.0)
            self._dismiss_reward_popup()
            self.mark_executed()
            return True

        logger.info("[ChapterQuest] Открываем Королевскую Дорогу (корона)...")
        self.client.tap(*self.crown_coords)
        time.sleep(2.5)
        acted = self._handle_royal_road_menu()
        if acted:
            self.mark_executed()
        return acted

    def _handle_royal_road_menu(self) -> bool:
        screen = self.client.get_screenshot(fresh=True)
        if screen is None:
            return False

        ocr = read_text(screen, scale=0.5)
        if not ocr.items:
            self.client.press_back()
            return False

        # 1. Финальная кнопка 'Забрать награды' внизу.
        for item in ocr.items:
            text = item.text.lower()
            if "забрать" in text and "наград" in text and item.y > int(screen.shape[0] * 0.76):
                logger.success(f"[ChapterQuest] Кнопка награды главы: ({item.x}, {item.y})")
                self.client.tap(item.x, item.y)
                time.sleep(2.0)
                self._dismiss_reward_popup()
                return True

        # 2. Золотая кнопка 'Вперёд' у первого незавершённого квеста.
        match = self.vision.find_template(screen, "btn_vpered_active", threshold=0.75)
        if match:
            vx, vy, confidence = match
            logger.info(f"[ChapterQuest] 'Вперёд' по шаблону: ({vx}, {vy}), conf={confidence:.2f}")
            self.client.tap(vx, vy)
            time.sleep(2.0)
            return True

        # 3. Резерв через OCR.
        for item in ocr.items:
            text = item.text.lower()
            if "вперед" in text:
                if int(screen.shape[1] * 0.72) <= item.x <= int(screen.shape[1] * 0.92) and int(screen.shape[0] * 0.33) <= item.y <= int(screen.shape[0] * 0.68):
                    logger.info(f"[ChapterQuest] 'Вперёд' по OCR: ({item.x}, {item.y}) [{item.text}]")
                    self.client.tap(item.x, item.y)
                    time.sleep(2.0)
                    return True

        logger.info("[ChapterQuest] Активных кнопок не найдено, закрываем окно.")
        self.client.press_back()
        return False

    def _dismiss_reward_popup(self) -> None:
        time.sleep(1.2)
        screen = self.client.get_screenshot(fresh=True)
        if screen is None:
            return

        ocr = read_text(screen, scale=0.5)
        text = ocr.text.lower()
        if any(word in text for word in ("награда", "повышен", "задание")):
            logger.info("[ChapterQuest] Закрываем окно награды (540, 1800)...")
            self.client.tap(540, 1800)
            time.sleep(1.5)
            self.client.tap(968, 160)
            time.sleep(1.0)
