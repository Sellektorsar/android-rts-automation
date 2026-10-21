"""
Скрипт для прохождения боевого этапа кампании / обучения.
Управляет джойстиком по круговой траектории (кайт), собирает кристаллы опыта,
выбирает навыки при повышении уровня и нажимает подтверждение при победе.
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


def main():
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
        return

    vision = Vision()
    ocr = RapidOCR()

    logger.info("Старт боевого цикла (круговой кайт)...")
    start_time = time.time()
    max_duration_sec = 180

    # Сбалансированный круговой кайт вокруг центра джойстика (540, 1980)
    # Позволяет не врезаться в стены арены и равномерно уклоняться от монстров
    move_cycle = [
        (540, 1980, 540, 1720, 1100),  # Вверх
        (540, 1980, 780, 1850, 1100),  # Вверх-вправо
        (540, 1980, 780, 2100, 1100),  # Вниз-вправо
        (540, 1980, 540, 2240, 1100),  # Вниз
        (540, 1980, 300, 2100, 1100),  # Вниз-влево
        (540, 1980, 300, 1850, 1100),  # Вверх-влево
    ]
    step_idx = 0

    while time.time() - start_time < max_duration_sec:
        screen = client.get_screenshot()
        if screen is None:
            time.sleep(1)
            continue

        cv2.imwrite("current_screen.png", screen)

        # 0. Периодически сворачиваем шторку уведомлений
        if step_idx % 8 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

        # 1. Проверяем экран выбора навыка
        if vision.find_template(screen, "title_skill_choice", threshold=0.70):
            logger.info("Обнаружен экран выбора навыка!")
            match_btn = vision.find_template(screen, "btn_green_confirm", threshold=0.70)
            if match_btn:
                cx, cy, conf = match_btn
                logger.info(f"Нажимаем 'Подтвердить' выбор навыка: ({cx}, {cy}), conf={conf:.2f}")
                client.tap(cx, cy)
                time.sleep(1.2)
                continue
            else:
                logger.info("Выбираем карту навыка (500, 1100)...")
                client.tap(500, 1100)
                time.sleep(0.6)
                continue

        # 2. Проверяем наличие отдельной кнопки 'Подтвердить'
        match = vision.find_template(screen, "btn_green_confirm", threshold=0.75)
        if match:
            cx, cy, conf = match
            logger.info(f"Обнаружена зеленая кнопка подтверждения: ({cx}, {cy}), conf={conf:.2f}")
            client.tap(cx, cy)
            time.sleep(1.5)
            continue

        # 3. Быстрая проверка OCR на победу или поражение
        ocr_res, _ = ocr(screen)
        ocr_texts = [item[1].lower() for item in ocr_res] if ocr_res else []
        joined_ocr = " ".join(ocr_texts)

        if any(w in joined_ocr for w in ["победа", "побед", "нажмите, чтобы", "victory", "100%"]):
            logger.success("Обнаружен экран победы! Кликаем для продолжения...")
            client.tap(540, 1850)
            time.sleep(2.0)
            client.tap(540, 2000)
            time.sleep(2.0)
            continue

        if any(w in joined_ocr for w in ["проиграна", "битва проиграна"]):
            logger.warning("Битва проиграна, нажимаем Повторить...")
            client.tap(875, 1418)  # Кнопка Повторить
            time.sleep(3.0)
            continue

        # 4. Проверяем диалог Беллы
        dialog_sample = screen[2150:2300, 400:680]
        d_mean = dialog_sample.mean(axis=(0, 1))
        if d_mean[2] > 170 and d_mean[1] > 150 and d_mean[0] < 150:
            logger.info("Обнаружен диалог! Кликаем для продолжения (540, 2250)...")
            client.tap(540, 2250)
            time.sleep(1.2)
            continue

        # 5. Проверяем возвращение в Город или на карту Кампании
        if (
            vision.find_template(screen, "btn_builder_idle", threshold=0.65)
            or vision.find_template(screen, "btn_quest_hammer", threshold=0.65)
            or vision.find_template(screen, "btn_world_map", threshold=0.65)
        ):
            logger.success("Мы вернулись в Город (City View)! Боевой этап успешно завершен.")
            break

        if any(w in joined_ocr for w in ["спаситель порядка", "испытания доблести", "просмотр навыка"]):
            if "этап 4" not in joined_ocr:
                logger.success("Мы на карте Кампании! Этап пройден.")
                break

        # 6. Круговой кайт
        x1, y1, x2, y2, dur = move_cycle[step_idx % len(move_cycle)]
        logger.debug(f"Боевой кайт #{step_idx}: ({x1},{y1}) -> ({x2},{y2}) [{dur}ms]")
        client.swipe(x1, y1, x2, y2, duration_ms=dur)
        step_idx += 1
        time.sleep(0.3)

    logger.info("Боевой цикл завершен.")


if __name__ == "__main__":
    main()
