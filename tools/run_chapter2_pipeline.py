"""
Главный конвейер завершения Главы 2 (Chapter 2 Completion Pipeline).
Запускает строгую последовательность шагов:
1. Этап 5 Кампании (Спаситель порядка) -> победа и сбор лута.
2. Сбор награды за квест Этапа 5 на баннере HUD.
3. Постройка второй Казармы 1 ур. через Королевскую Дорогу.
4. Бесплатное ускорение постройки.
5. Сбор награды за постройку Казарм.
6. Открытие Королевской Дороги и получение финальной награды за всю Главу 2.
"""

import sys
import time
from pathlib import Path
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient
from src.vision import Vision
from tools.complete_campaign_stage import run_campaign_stage
from tools.build_barracks_flow import build_second_barracks
from tools.complete_chapter_quest import complete_chapter_step


def claim_hud_banner_reward(client: AdbClient, ocr: RapidOCR) -> bool:
    """Проверяет и собирает выполненную награду с баннера квеста в левом нижнем углу HUD."""
    logger.info("Проверка готовых наград на баннере HUD (250, 1955)...")
    for _ in range(3):
        screen = client.get_screenshot()
        if screen is None:
            time.sleep(1.0)
            continue
        
        ocr_res, _ = ocr(screen)
        has_reward = False
        for box, text, score in ocr_res if ocr_res else []:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)
            if 1900 <= cy <= 2010 and cx < 600:
                if any(w in txt for w in ["(1/1)", "(2/2)", "(30/30)", "заберите", "награды"]):
                    has_reward = True
                    break

        if has_reward:
            logger.success("Найдена готовая награда! Забираем кликом по баннеру (250, 1955)...")
            client.tap(250, 1955)
            time.sleep(2.0)
            # Закрываем модальное окно наград
            client.tap(540, 1800)
            time.sleep(1.0)
            client.tap(968, 160)
            time.sleep(1.0)
            return True
        else:
            time.sleep(1.0)
    return False


def claim_final_chapter_reward(client: AdbClient, ocr: RapidOCR) -> bool:
    """Открывает Королевскую Дорогу и забирает главную награду главы."""
    logger.info("Открываем Королевскую Дорогу для сбора финальной награды главы...")
    client.tap(75, 1960)
    time.sleep(2.5)

    screen = client.get_screenshot()
    ocr_res, _ = ocr(screen)
    
    claimed = False
    for box, text, score in ocr_res if ocr_res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if "забрать" in txt and "наград" in txt and cy > 1800:
            logger.success(f"Обнаружена финальная кнопка награды главы: ({cx}, {cy}). Забираем!")
            client.tap(cx, cy)
            time.sleep(2.5)
            # Закрываем модальное окно наград
            client.tap(540, 1800)
            time.sleep(1.5)
            claimed = True
            break

    if not claimed:
        # Пробуем нажать по стандартным координатам финальной кнопки (543, 1995)
        logger.info("Пробуем забрать финальную награду по координатам (543, 1995)...")
        client.tap(543, 1995)
        time.sleep(2.0)
        client.tap(540, 1800)
        time.sleep(1.5)

    client.press_back()
    time.sleep(1.5)
    return True


def run_pipeline():
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

    logger.info("==========================================")
    logger.info("   КОНВЕЙЕР ЗАВЕРШЕНИЯ ГЛАВЫ 2 (CHAPTER 2)")
    logger.info("==========================================")

    # 1. Проверяем и собираем готовые награды на баннере HUD
    logger.info("--- ШАГ 1: Проверка готовых наград на баннере HUD ---")
    claim_hud_banner_reward(client, ocr)

    # 2. Прохождение Этапа 5 Кампании (Спаситель порядка)
    logger.info("--- ШАГ 2: Запуск и прохождение Этапа 5 Кампании ---")
    stage_res = run_campaign_stage(5)
    if stage_res:
        logger.success("Этап 5 Кампании успешно завершен!")
    else:
        logger.warning("Этап 5 завершился с предупреждением, продолжаем сбор...")
    time.sleep(2.5)

    # 3. Сбор награды за этап 5 на баннере HUD
    logger.info("--- ШАГ 3: Сбор награды за Этап 5 на баннере HUD ---")
    claim_hud_banner_reward(client, ocr)
    time.sleep(2.0)

    # 4. Сбор финальной награды Главы 2 в Королевской Дороге
    logger.info("--- ШАГ 4: Сбор финальной награды всей Главы 2 ---")
    claim_final_chapter_reward(client, ocr)

    logger.success("==========================================")
    logger.success("   ГЛАВА 2 (РАЗВИТИЕ ТЕХНОЛОГИЙ) ЗАВЕРШЕНА!")
    logger.success("==========================================")
    return True


if __name__ == "__main__":
    run_pipeline()
