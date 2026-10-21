"""
Автономный скрипт сбора и выполнения заданий Королевской Дороги (Chapter Quest Flow).
1. Проверка наличия готовой награды на баннере (250, 1955).
2. Сбор награды и закрытие поздравительного окна.
3. Открытие Королевской Дороги (75, 1960).
4. Нажатие активной кнопки 'Вперед' первого незавершенного квеста.
5. Сбор финальной награды главы (543, 1995), когда все задания выполнены.
"""

import sys
import time
from pathlib import Path
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient


def complete_chapter_step():
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
    screen = client.get_screenshot()
    if screen is None:
        return False

    res, _ = ocr(screen)
    banner_done = False
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if 1920 <= cy <= 2000 and cx < 600:
            if any(x in txt for x in ["(1/1)", "(2/2)", "(3/3)", "(4/4)", "(5/5)", "(30/30)", "заберите", "награды"]):
                banner_done = True
                break

    if banner_done:
        logger.success("Обнаружена готовая награда на баннере! Клик по (250, 1955)...")
        client.tap(250, 1955)
        time.sleep(1.8)
        client.tap(540, 1800)  # Закрыть попап награды
        time.sleep(1.0)
        return True

    logger.info("Открытие меню Королевской Дороги (75, 1960)...")
    client.tap(75, 1960)
    time.sleep(2.0)

    screen = client.get_screenshot()
    res, _ = ocr(screen)

    # Проверка кнопки 'Забрать награды главы' внизу
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if "забрать" in txt and "наград" in txt and cy > 1850:
            logger.success(f"Обнаружена кнопка награды за главу: ({cx}, {cy})! Забираем...")
            client.tap(cx, cy)
            time.sleep(2.0)
            client.tap(540, 1800)
            return True

    # Ищем первую кнопку 'Вперед'
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if ("вперед" in txt or "bnepe" in txt) and cx > 750 and 800 <= cy <= 1600:
            logger.info(f"Нажатие первой кнопки 'Вперед' ({cx}, {cy})...")
            client.tap(cx, cy)
            time.sleep(2.0)
            return True

    client.press_back()
    return False


if __name__ == "__main__":
    complete_chapter_step()
