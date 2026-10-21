"""
Автономный скрипт прохождения боевого этапа Кампании (Спаситель порядка).
Полный цикл:
1. Запуск боя из окна брифинга (зеленая кнопка 'Испытание' на 543, 1885) или через карту Кампании.
   Для Этапа 6 узел на карте находится на (822, 1284).
2. Загрузка боевой локации.
3. Непрерывный круговой кайт джойстиком вокруг центра (540, 1980).
4. Автоматический выбор карт навыков при level-up: (537, 1004) + Подтвердить (541, 1744).
5. Обработка экрана победы (Victory / 100% / Нажмите, чтобы продолжить) и возврат в Город.
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


def find_green_button(screen: np.ndarray) -> tuple[int, int] | None:
    """Поиск зеленой кнопки 'Испытание' / 'Подтвердить' по цвету HSV."""
    hsv = cv2.cvtColor(screen[1600:2050, 200:880], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 60), (85, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 140 and h > 45:
            cx = 200 + x + w // 2
            cy = 1600 + y + h // 2
            return (cx, cy)
    return None


def run_campaign_stage(stage_num: int = 6) -> bool:
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

    logger.info(f"=== ЗАПУСК ЭТАПА {stage_num} КАМПАНИИ ===")

    # Координаты узлов на карте Кампании
    stage_nodes = {
        1: (293, 2051),
        2: (279, 1710),
        3: (818, 1673),
        4: (463, 1563),
        5: (830, 1605),
        6: (822, 1284),
    }

    # Шаг 1: Проверяем текущий экран
    screen = client.get_screenshot()
    if screen is None:
        logger.error("Не удалось получить скриншот!")
        return False

    btn_pos = find_green_button(screen)
    if btn_pos:
        logger.info(f"Обнаружена кнопка 'Испытание': {btn_pos}. Запускаем бой...")
        client.tap(btn_pos[0], btn_pos[1])
        time.sleep(4.0)
    else:
        # Проверяем, может мы на карте Кампании
        ocr_res, _ = ocr(screen)
        ocr_text = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
        if "спаситель" in ocr_text or "доблести" in ocr_text:
            node_x, node_y = stage_nodes.get(stage_num, (822, 1284))
            logger.info(f"Мы на карте Кампании. Кликаем по узлу этапа {stage_num} ({node_x}, {node_y})...")
            client.tap(node_x, node_y)
            time.sleep(2.0)
            screen = client.get_screenshot()
            btn_pos = find_green_button(screen)
            if btn_pos:
                client.tap(btn_pos[0], btn_pos[1])
            else:
                client.tap(543, 1885)
            time.sleep(4.0)
        else:
            logger.warning("Кнопка 'Испытание' не найдена, пробуем кликнуть по (543, 1885)...")
            client.tap(543, 1885)
            time.sleep(4.0)

    # Шаг 2: Боевой цикл (активный круговой кайт)
    logger.info("Бой начался! Активируем непрерывный круговой кайт...")
    battle_start = time.time()
    max_duration = 110  # секунд максимум для боя с боссом
    
    # 4-направленный круговой кайт вокруг джойстика (540, 1980)
    kite_moves = [
        (540, 1980, 540, 1680, 1100),  # Вверх
        (540, 1980, 840, 1980, 1100),  # Вправо
        (540, 1980, 540, 2280, 1100),  # Вниз
        (540, 1980, 240, 1980, 1100),  # Влево
    ]
    
    cycle_count = 0
    in_active_battle = True

    while in_active_battle and (time.time() - battle_start < max_duration):
        for x1, y1, x2, y2, dur in kite_moves:
            client.swipe(x1, y1, x2, y2, duration_ms=dur)
            time.sleep(0.05)
        
        cycle_count += 1
        elapsed = time.time() - battle_start
        logger.info(f"Боевой кайт: цикл #{cycle_count}, прошло {elapsed:.1f}s...")
        
        # Снятие паузы / выбор навыка при level-up
        # Тапаем карту навыка (537, 1004) и зеленую кнопку подтверждения (541, 1744)
        client.tap(537, 1004)
        time.sleep(0.25)
        client.tap(541, 1744)
        time.sleep(0.25)
        
        # Сворачиваем статусбар каждые 3 цикла
        if cycle_count % 3 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

        # После 40 секунд начинаем проверять, не закончился ли бой
        if elapsed > 40:
            screen = client.get_screenshot()
            if screen is not None:
                # Быстрая проверка на экран победы по тексту
                ocr_res, _ = ocr(screen)
                joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""
                
                if any(w in joined for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                    logger.success("ОБНАРУЖЕН ЭКРАН ПОБЕДЫ!")
                    in_active_battle = False
                    break
                
                if any(w in joined for w in ["проиграна", "битва проиграна"]):
                    logger.warning("Битва проиграна, жмем 'Повторить'...")
                    client.tap(875, 1418)
                    time.sleep(3.0)
                    battle_start = time.time()
                    continue

                if any(w in joined for w in ["спаситель порядка", "просмотр навыка"]):
                    logger.success("Мы вернулись на карту Кампании!")
                    in_active_battle = False
                    break

    logger.info("Боевая фаза завершена. Обрабатываем экран победы и сбор наград...")
    time.sleep(2.0)

    # Шаг 3: Сбор наград и закрытие экрана победы
    for _ in range(6):
        client.tap(540, 1850)
        time.sleep(1.2)
        client.tap(540, 2000)
        time.sleep(1.0)

    # Шаг 4: Возврат в Город (City View)
    logger.info("Возврат в Город (City View)...")
    for attempt in range(5):
        screen = client.get_screenshot()
        if screen is None:
            time.sleep(1.0)
            continue

        ocr_res, _ = ocr(screen)
        joined = " ".join([item[1].lower() for item in ocr_res]) if ocr_res else ""

        # Проверка возврата в Город
        if (
            vision.find_template(screen, "btn_builder_idle", threshold=0.65)
            or vision.find_template(screen, "btn_quest_hammer", threshold=0.65)
            or vision.find_template(screen, "btn_world_map", threshold=0.65)
            or "замок" in joined
            or "событие" in joined
        ):
            logger.success("Успешно вернулись в Город (City View)!")
            return True

        # Если на карте Кампании - кликаем стрелку 'Назад' (75, 120)
        if "спаситель" in joined or "просмотр" in joined or "доблести" in joined:
            logger.info(f"Попытка #{attempt + 1}: Нажимаем 'Назад' на карте Кампании (75, 120)...")
            client.tap(75, 120)
            time.sleep(2.5)
        else:
            client.tap(540, 1850)
            time.sleep(1.5)

    return True


if __name__ == "__main__":
    stage = 6
    if len(sys.argv) > 1:
        stage = int(sys.argv[1])
    run_campaign_stage(stage)
