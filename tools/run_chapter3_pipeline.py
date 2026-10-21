"""
Главный конвейер завершения Главы 3 (Chapter 3 Completion Pipeline).
Проходит финальный квест Главы 3:
1. Этап 6 Кампании (Спаситель порядка).
2. Сбор награды за Этап 6 на баннере HUD.
3. Сбор финальной награды за Главу 3 в Королевской Дороге.
4. Открытие Главы 4.
"""

import sys
import time
from pathlib import Path
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient
from tools.complete_campaign_stage import find_green_button


def run_stage_6(client: AdbClient, ocr: RapidOCR) -> bool:
    logger.info("=== ПЕРЕХОД И ПРОХОЖДЕНИЕ ЭТАПА 6 КАМПАНИИ ===")
    
    # 1. Если Королевская Дорога открыта, жмем 'Вперед' у Этапа 6 (871, 948)
    screen = client.get_screenshot()
    ocr_res, _ = ocr(screen)
    joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
    
    if "дорога" in joined or "глава 3" in joined or "этап 6" in joined:
        logger.info("Нажимаем 'Вперед' напротив Этапа 6 (871, 948)...")
        client.tap(871, 948)
        time.sleep(2.5)

    # 2. Если мы в Городе - тапаем по зданию Кампании в центре (540, 1200)
    screen = client.get_screenshot()
    ocr_res, _ = ocr(screen)
    joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
    
    if "замок" in joined or "рюкзак" in joined or "событие" in joined:
        logger.info("Открываем карту Кампании кликом по (540, 1200)...")
        client.tap(540, 1200)
        time.sleep(2.5)

    # 3. На карте Кампании кликаем по узлу Этапа 6 (760, 1180)
    logger.info("Кликаем по узлу Этапа 6 (760, 1180)...")
    client.tap(760, 1180)
    time.sleep(2.5)

    # 4. В окне брифинга нажимаем зеленую кнопку 'Испытание' (542, 1879)
    screen = client.get_screenshot()
    btn = find_green_button(screen)
    btn_pos = btn if btn else (542, 1879)
    logger.info(f"Нажимаем 'Испытание' по координатам {btn_pos}...")
    client.tap(btn_pos[0], btn_pos[1])
    time.sleep(3.5)

    # 5. Боевой цикл: непрерывный кайт джойстиком с авто-выбором навыков
    logger.info("Бой Этапа 6 начался! Активируем непрерывный круговой кайт...")
    battle_start = time.time()
    kite_moves = [
        (540, 1980, 540, 1680, 1000),  # Вверх
        (540, 1980, 840, 1980, 1000),  # Вправо
        (540, 1980, 540, 2280, 1000),  # Вниз
        (540, 1980, 240, 1980, 1000),  # Влево
    ]

    cycle = 0
    # Кайтим 75-80 секунд
    while time.time() - battle_start < 80:
        for x1, y1, x2, y2, dur in kite_moves:
            client.swipe(x1, y1, x2, y2, duration_ms=dur)
            time.sleep(0.05)
        cycle += 1
        elapsed = time.time() - battle_start
        logger.info(f"Бой Этапа 6: цикл #{cycle}, прошло {elapsed:.1f}с...")
        
        # Снятие паузы выбора навыка (если появился)
        client.tap(537, 1004)
        time.sleep(0.15)
        client.tap(541, 1744)
        time.sleep(0.15)
        
        if cycle % 3 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

    logger.info("Время боя истекло, ожидаем экран Победы...")
    time.sleep(3.0)

    # 6. Закрываем экран победы
    for _ in range(5):
        client.tap(540, 1826)
        time.sleep(1.2)
        client.tap(540, 1850)
        time.sleep(1.2)

    # 7. Возврат в Город
    logger.info("Возврат в Город (City View)...")
    for _ in range(4):
        screen = client.get_screenshot()
        ocr_res, _ = ocr(screen)
        joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
        if "замок" in joined or "рюкзак" in joined or "событие" in joined:
            logger.success("Успешно вернулись в Город!")
            return True
        client.tap(75, 120)
        time.sleep(1.5)
        client.press_back()
        time.sleep(1.5)

    return True


def claim_chapter3_rewards(client: AdbClient, ocr: RapidOCR):
    # 1. Сбор награды за Этап 6 на баннере HUD
    logger.info("Сбор награды за Этап 6 на баннере HUD (250, 1955)...")
    client.tap(250, 1955)
    time.sleep(2.0)
    client.tap(540, 1800)
    time.sleep(1.5)

    # 2. Открытие Королевской Дороги
    logger.info("Открытие Королевской Дороги (75, 1960)...")
    client.tap(75, 1960)
    time.sleep(2.5)

    # 3. Забрать главную награду главы (543, 1995)
    logger.info("Сбор финальной награды Главы 3 (543, 1995)...")
    client.tap(543, 1995)
    time.sleep(2.5)
    client.tap(540, 1800)
    time.sleep(1.5)
    client.tap(540, 1800)
    time.sleep(1.5)

    logger.success("ГЛАВА 3 ПОЛНОСТЬЮ ЗАВЕРШЕНА!")


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
        return False

    ocr = RapidOCR()
    run_stage_6(client, ocr)
    claim_chapter3_rewards(client, ocr)
    return True


if __name__ == "__main__":
    run_pipeline()
