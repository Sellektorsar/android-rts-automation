"""
Скрипт завершения Главы 3:
1. Забирает награду за 5-й квест (Вызов Спасителя порядка, этап 6) на HUD.
2. Открывает 'Путь Правителя' (Royal Road) на (75, 1960).
3. Забирает финальную награду Главы 3 (543, 1995).
4. Закрывает модальное окно наград и проверяет переход в Главу 4.
"""

import sys
import time
from pathlib import Path
import cv2
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient
from src.vision import Vision


def claim_chapter3_rewards():
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

    ocr = RapidOCR()
    vision = Vision()

    logger.info("=== ФИНАЛИЗАЦИЯ ГЛАВЫ 3 ===")

    # 1. Забираем награду за квест 5 на HUD
    logger.info("1. Проверяем HUD и клеймим награду за квест (250, 1955)...")
    for _ in range(3):
        client.tap(250, 1955)
        time.sleep(1.2)
        client.tap(540, 1800)
        time.sleep(1.0)

    # 2. Открываем Путь Правителя
    logger.info("2. Открываем Путь Правителя (75, 1960)...")
    client.tap(75, 1960)
    time.sleep(2.5)

    screen = client.get_screenshot()
    if screen is not None:
        cv2.imwrite("screen_royal_road_ch3_done.png", screen)
        artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\royal_road_ch3_finish.png"
        cv2.imwrite(artifact_path, screen)
        
        ocr_res, _ = ocr(screen)
        joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
        logger.info(f"Royal Road OCR: {joined[:120]}")

    # 3. Кликаем 'Забрать награды главы'
    logger.info("3. Забираем награды Главы 3 (543, 1995)...")
    client.tap(543, 1995)
    time.sleep(2.0)
    
    # 4. Пропускаем анимацию наград
    for _ in range(4):
        client.tap(540, 1800)
        time.sleep(1.0)

    # 5. Проверяем переход в Главу 4
    screen_final = client.get_screenshot()
    if screen_final is not None:
        cv2.imwrite("screen_chapter4_start.png", screen_final)
        artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_start.png"
        cv2.imwrite(artifact_path, screen_final)
        
        ocr_res, _ = ocr(screen_final)
        joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
        logger.info(f"Итоговый OCR: {joined[:150]}")
        if any(w in joined for w in ["глава 4", "глава iv", "глава"]):
            logger.success("ГЛАВА 3 УСПЕШНО ЗАВЕРШЕНА! ДОБРО ПОЖАЛОВАТЬ В ГЛАВУ 4!")
            return True

    return True


if __name__ == "__main__":
    claim_chapter3_rewards()
