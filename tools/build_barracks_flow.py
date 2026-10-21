"""
Автономный скрипт постройки второй Казармы (Build Second Barracks Flow).
Полный цикл:
1. Проверка Города (City View).
2. Переход к квесту 'Построить 2 Казармы 1 ур.' через Королевскую Дорогу (кнопка 'Вперед').
3. Клик по подсвеченному участку строительства в центре экрана (540, 1200).
4. Выбор 'Казарма' в меню строительства (через OCR / координаты).
5. Нажатие кнопки 'Бесплатно' (304, 2339) или 'Построить' (794, 2338).
6. Бесплатное мгновенное ускорение (если таймер < 5 минут).
7. Сбор награды за квест на баннере HUD (250, 1955).
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
from src.vision import Vision


def build_second_barracks() -> bool:
    config_path = Path("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
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

    vision = Vision()
    ocr = RapidOCR()

    logger.info("=== ЗАПУСК ПОСТРОЙКИ ВТОРОЙ КАЗАРМЫ ===")

    # 1. Проверяем Город
    screen = client.get_screenshot()
    if screen is None:
        return False

    # 2. Открываем Королевскую Дорогу (75, 1960)
    logger.info("Открываем Королевскую Дорогу (75, 1960)...")
    client.tap(75, 1960)
    time.sleep(2.5)

    screen = client.get_screenshot()
    ocr_res, _ = ocr(screen)
    
    # Ищем строку с квестом Казарм
    vpered_target = None
    for box, text, score in ocr_res if ocr_res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if any(w in txt for w in ["казарм", "казармы", "построить 2"]):
            logger.info(f"Найден квест Казарм: '{text}' на высоте y={cy}")
            # Кнопка 'Вперед' находится справа на той же высоте (x ~ 870)
            vpered_target = (870, cy)
            break

    if vpered_target:
        logger.info(f"Нажимаем 'Вперед' у квеста казарм: {vpered_target}...")
        client.tap(vpered_target[0], vpered_target[1])
        time.sleep(2.5)
    else:
        # Fallback: нажимаем первую доступную кнопку 'Вперед' (x: 800-950, y: 800-1400)
        logger.info("Ищем любую кнопку 'Вперед'...")
        for box, text, score in ocr_res if ocr_res else []:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)
            if "вперед" in txt or "bnepe" in txt:
                logger.info(f"Нажимаем 'Вперед' по координатам ({cx}, {cy})...")
                client.tap(cx, cy)
                time.sleep(2.5)
                break

    # 3. Камера центрировалась на свободном участке строительства
    logger.info("Камера навелась на участок. Кликаем по центру участка (540, 1200)...")
    client.tap(540, 1200)
    time.sleep(2.0)

    # 4. Проверяем меню строительства
    screen = client.get_screenshot()
    ocr_res, _ = ocr(screen)
    barracks_btn = None
    for box, text, score in ocr_res if ocr_res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if "казарма" in txt:
            barracks_btn = (cx, cy)
            logger.info(f"Найдена карточка 'Казарма' в меню: ({cx}, {cy})")
            break

    if barracks_btn:
        logger.info(f"Выбираем 'Казарма' ({barracks_btn[0]}, {barracks_btn[1]})...")
        client.tap(barracks_btn[0], barracks_btn[1])
        time.sleep(2.0)
        screen = client.get_screenshot()

    # 5. В окне постройки нажимаем 'Бесплатно' (304, 2339) или 'Построить' (794, 2338)
    logger.info("Нажимаем кнопку постройки/бесплатно (304, 2339)...")
    client.tap(304, 2339)
    time.sleep(1.0)
    client.tap(794, 2338)
    time.sleep(2.5)

    # 6. Проверяем бесплатное ускорение
    logger.info("Проверяем бесплатное ускорение постройки...")
    client.tap(540, 1200)
    time.sleep(1.0)
    client.tap(304, 2339)
    time.sleep(2.0)

    # 7. Забираем награду за выполненный квест
    logger.info("Забираем награду за постройку казарм...")
    client.tap(250, 1955)
    time.sleep(2.0)
    # Закрываем экран поздравления
    client.tap(540, 1800)
    time.sleep(1.0)

    logger.success("Постройка второй Казармы завершена!")
    return True


if __name__ == "__main__":
    build_second_barracks()
