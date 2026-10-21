"""
Скрипт / Задача автоматического прохождения сюжетных квестов (Королевская Дорога).
Выполняет цикл:
1. Мониторинг баннера квеста в левом нижнем углу HUD (y ~ 1945).
2. Сбор готовых наград (если квест выполнен: 1/1, 2/2, 3/3 или 'Заберите награды').
3. Открытие меню Королевской Дороги через иконку короны (75, 1960).
4. Автоматическое нажатие кнопки 'Вперед' у первого активного задания.
5. Сбор финальной награды за всю главу ('Забрать награды главы').
"""

import time
from typing import Optional
from loguru import logger
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from src.adb_client import AdbClient
from src.tasks.base_task import BaseTask
from src.vision import Vision


class ChapterQuestTask(BaseTask):
    def __init__(self, client: AdbClient, vision: Vision, interval_sec: int = 20):
        super().__init__("ChapterQuest", client, vision, interval_sec=interval_sec)
        self.ocr = RapidOCR()
        self.crown_coords = (75, 1960)
        self.banner_tap_coords = (250, 1955)

    def can_run(self) -> bool:
        return super().can_run()

    def execute(self, screen: np.ndarray) -> bool:
        """
        Проверка и выполнение сюжетных квестов.
        """
        self.mark_executed()
        logger.info("[ChapterQuest] Проверка статуса сюжетных заданий...")
        ocr_res, _ = self.ocr(screen)
        if not ocr_res:
            return False

        # Шаг 1: Проверяем текст в зоне квестового баннера (x: 100-600, y: 1920-2000)
        quest_text = ""
        is_completed = False
        is_chapter_finished = False

        for box, text, score in ocr_res:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)

            if 1900 <= cy <= 2010 and cx < 600:
                quest_text += " " + txt
                if any(x in txt for x in ["(1/1)", "(2/2)", "(3/3)", "(4/4)", "(5/5)", "заберите", "награды"]):
                    is_completed = True
                if "награды главы" in txt or "заберите награды" in txt:
                    is_chapter_finished = True

        # Шаг 2: Если награда готова к сбору - кликаем по баннеру
        if is_completed or is_chapter_finished:
            logger.success(f"[ChapterQuest] Награда за задание готова к сбору! Текст: '{quest_text.strip()}'. Кликаем баннер...")
            self.client.tap(self.banner_tap_coords[0], self.banner_tap_coords[1])
            time.sleep(2.0)
            self._dismiss_reward_popup()
            return True

        # Шаг 3: Если квест еще не завершен, открываем меню Королевской Дороги
        logger.info("[ChapterQuest] Открываем меню Королевской Дороги (кликом по короне)...")
        self.client.tap(self.crown_coords[0], self.crown_coords[1])
        time.sleep(2.5)

        return self._handle_royal_road_menu()

    def _handle_royal_road_menu(self) -> bool:
        """
        Обработка окна списка квестов главы:
        Ищет первую активную кнопку 'Вперед' или 'Забрать награды главы'.
        """
        screen = self.client.get_screenshot()
        if screen is None:
            return False

        ocr_res, _ = self.ocr(screen)
        if not ocr_res:
            self.client.press_back()
            return False

        # 1. Проверяем, доступна ли финальная кнопка 'Забрать награды' внизу (y ~ 1975)
        for box, text, score in ocr_res:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)

            if "забрать" in txt and "наград" in txt and cy > 1850:
                logger.success(f"[ChapterQuest] Обнаружена кнопка сбора награды за главу: ({cx}, {cy})")
                self.client.tap(cx, cy)
                time.sleep(2.0)
                self._dismiss_reward_popup()
                return True

        # 2. Ищем золотую кнопку 'Вперед' у первого незавершенного квеста
        match_vpered = self.vision.find_template(screen, "btn_vpered_active", threshold=0.75)
        if match_vpered:
            vx, vy, conf = match_vpered
            logger.info(f"[ChapterQuest] Нажата активная кнопка 'Вперед' по шаблону: ({vx}, {vy}), conf={conf:.2f}")
            self.client.tap(vx, vy)
            time.sleep(2.0)
            return True

        # Fallback через OCR: ищем слово 'Вперед' в диапазоне x: 800-950
        for box, text, score in ocr_res:
            txt = text.lower()
            cx = int((box[0][0] + box[1][0]) / 2)
            cy = int((box[0][1] + box[2][1]) / 2)

            if "вперед" in txt or "bnepe" in txt:
                if 800 <= cx <= 980 and 850 <= cy <= 1650:
                    logger.info(f"[ChapterQuest] Нажата кнопка 'Вперед' по OCR: ({cx}, {cy}) [{text}]")
                    self.client.tap(cx, cy)
                    time.sleep(2.0)
                    return True

        logger.info("[ChapterQuest] Активных кнопок не найдено, закрываем окно назад.")
        self.client.press_back()
        return False

    def _dismiss_reward_popup(self) -> None:
        """Закрытие всплывающего окна полученных ресурсов/наград."""
        time.sleep(1.2)
        screen = self.client.get_screenshot()
        if screen is None:
            return

        ocr_res, _ = self.ocr(screen)
        ocr_text = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""

        if "награда" in ocr_text or "повышен" in ocr_text or "задание" in ocr_text:
            logger.info("[ChapterQuest] Закрываем всплывающее окно награды кликом (540, 1800)...")
            self.client.tap(540, 1800)
            time.sleep(1.5)
            # Дополнительный клик для пропуска диалога Беллы
            self.client.tap(968, 160)
            time.sleep(1.0)
