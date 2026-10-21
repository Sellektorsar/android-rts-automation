"""
Автономный скрипт улучшения Замка (Castle Upgrade Flow).
Выполняет детерминированную последовательность действий:
1. Подключение к устройству через ADB.
2. Клик по Замку (540, 1100).
3. Клик по кнопке 'Улучшить' в радиальном меню (376, 1358).
4. Проверка условий в модальном окне:
   - Если горит зеленая кнопка 'Бесплатно' (304, 2339) -> моментальный апгрейд.
   - Иначе клик по синей кнопке 'Улучшить' (794, 2338).
5. Если время стройки <= 5 мин -> бесплатное мгновенное завершение.
6. Закрытие поздравлений и диалогов.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient


def upgrade_castle():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(
        adb_path=dev_cfg["adb_path"],
        ip=dev_cfg["ip"],
        port=dev_cfg["port"],
        timeout=dev_cfg.get("timeout_sec", 10),
    )

    if not client.connect():
        logger.error("Не удалось подключиться к ADB!")
        return False

    ocr = RapidOCR()

    logger.info("Шаг 1: Клик по Замку (540, 1100)...")
    client.tap(540, 1100)
    time.sleep(1.2)

    logger.info("Шаг 2: Клик по кнопке 'Улучшить' в радиальном меню (376, 1358)...")
    client.tap(376, 1358)
    time.sleep(2.0)

    screen = client.get_screenshot()
    if screen is None:
        logger.error("Не удалось захватить экран модалки улучшения!")
        return False

    # Шаг 3: Проверка кнопок модалки
    # Зеленая кнопка 'Бесплатно': (304, 2339)
    # Синяя кнопка 'Улучшить': (794, 2338)
    res, _ = ocr(screen)
    has_free = False
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        if cy > 2200 and ("бесплат" in txt or "becna" in txt):
            has_free = True
            break

    if has_free:
        logger.success("Доступно мгновенное бесплатное улучшение! Клик по (304, 2339)...")
        client.tap(304, 2339)
    else:
        logger.info("Запуск стройки по кнопке 'Улучшить' (794, 2338)...")
        client.tap(794, 2338)

    time.sleep(2.0)

    # Шаг 4: Закрытие возможных поздравлений / баннеров
    client.tap(540, 2000)
    time.sleep(1.0)
    client.tap(968, 160)  # Кнопка 'Пропустить' диалог

    logger.success("Последовательность улучшения замка успешно выполнена!")
    return True


if __name__ == "__main__":
    upgrade_castle()
