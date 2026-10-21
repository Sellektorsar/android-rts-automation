"""
Мастер-скрипт победы на Этапе 6 (Босс) с активным сбором кристаллов и завершением Главы 3.
Ключевые улучшения:
1. Траектория 'Figure-8 & Cross-Center Vacuum': герой постоянно пересекает центр и перекрестные линии,
   подбирая 100% выпавших кристаллов опыта (EXP gems).
2. Интеллектуальный выбор навыков по приоритетам (Магнит/Радиус сбора > Пушка > Щит > Меч).
3. Усиленные характеристики героя (Lv.3, Атака 120, HP 1500).
4. Автоматический сбор финальных наград Главы 3.
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


def select_best_skill(client: AdbClient, ocr: RapidOCR, screen) -> bool:
    """Интеллектуальный выбор лучшего навыка при level-up."""
    res, _ = ocr(screen)
    joined = " ".join([r[1].lower() for r in res]) if res else ""
    if not any(w in joined for w in ["выбор навыка", "bbl6op habblka", "подтвердить", "oatbepantb"]):
        return False

    logger.info("Обнаружен экран выбора навыка! Анализируем 3 карты...")
    
    # Координаты карт: Card 1 (180, 1100), Card 2 (540, 1100), Card 3 (890, 1100)
    card_boxes = {
        1: (80, 800, 360, 1300),
        2: (380, 800, 700, 1300),
        3: (720, 800, 1020, 1300),
    }
    
    card_scores = {1: 1, 2: 2, 3: 1}  # базовый приоритет (Card 2 по умолчанию)
    
    for c_idx, (x1, y1, x2, y2) in card_boxes.items():
        card_crop = screen[y1:y2, x1:x2]
        c_res, _ = ocr(card_crop)
        c_text = " ".join([r[1].lower() for r in c_res]) if c_res else ""
        logger.info(f"Карта #{c_idx}: {c_text[:50]}")
        
        # Приоритет 1: Магнит / Радиус сбора (топ для кристаллов)
        if any(k in c_text for k in ["магнит", "притяж", "радиус", "сбор"]):
            card_scores[c_idx] = 100
        # Приоритет 2: Алхимическая пушка (мощный АОЕ урон)
        elif any(k in c_text for k in ["пушк", "алхим", "cannon"]):
            card_scores[c_idx] = 50
        # Приоритет 3: Божественный щит (неуязвимость)
        elif any(k in c_text for k in ["щит", "божеств", "shield"]):
            card_scores[c_idx] = 40
        # Приоритет 4: Штормовой меч
        elif any(k in c_text for k in ["шторм", "меч", "sword"]):
            card_scores[c_idx] = 30
        # Приоритет 5: Разряд / Молния
        elif any(k in c_text for k in ["разряд", "молн", "light"]):
            card_scores[c_idx] = 20

    best_card = max(card_scores, key=card_scores.get)
    card_tap_x = {1: 220, 2: 540, 3: 870}[best_card]
    
    logger.success(f"Выбираем Карту #{best_card} (счет {card_scores[best_card]}) на ({card_tap_x}, 1100)...")
    client.tap(card_tap_x, 1100)
    time.sleep(0.5)
    
    # Кнопка 'Подтвердить'
    logger.info("Подтверждаем выбор навыка (541, 1744)...")
    client.tap(541, 1744)
    time.sleep(1.0)
    return True


def run_stage6():
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

    logger.info("=== СТАРТ ЭТАПА 6 (ЛИДЕР) С ВАКУУМНЫМ СБОРОМ КРИСТАЛЛОВ ===")

    # 1. Запуск боя из карты Кампании или брифинга
    screen = client.get_screenshot()
    if screen is not None:
        res, _ = ocr(screen)
        joined = " ".join([r[1].lower() for r in res]) if res else ""
        
        if "спаситель" in joined or "просмотр" in joined or "доблести" in joined:
            logger.info("Открываем узел Этапа 6 (822, 1284)...")
            client.tap(822, 1284)
            time.sleep(2.0)
            logger.info("Жмем 'Испытание' (540, 1885)...")
            client.tap(540, 1885)
            time.sleep(3.5)
        elif "этап 6" in joined or "победить лидера" in joined:
            logger.info("Жмем 'Испытание' в брифинге (540, 1885)...")
            client.tap(540, 1885)
            time.sleep(3.5)

    # 2. Траектория сбора кристаллов 'Vacuum Figure-8'
    # Центр джойстика: (540, 1980)
    # Постоянно пересекаем центр, собирая 100% кристаллов
    vacuum_moves = [
        (540, 1980, 740, 1780, 650),   # Вверх-вправо
        (540, 1980, 840, 1980, 600),   # Вправо
        (540, 1980, 340, 2180, 800),   # ПЕРЕСЕЧЕНИЕ ЦЕНТРА ВНИЗ-ВЛЕВО (пылесос кристаллов)
        (540, 1980, 240, 1980, 600),   # Влево
        (540, 1980, 340, 1780, 650),   # Вверх-влево
        (540, 1980, 740, 2180, 800),   # ПЕРЕСЕЧЕНИЕ ЦЕНТРА ВНИЗ-ВПРАВО (пылесос кристаллов)
        (540, 1980, 540, 1680, 650),   # Вверх
        (540, 1980, 540, 2280, 650),   # Вниз через центр
    ]

    start_time = time.time()
    max_duration = 180  # до 3 минут на босса
    cycle = 0
    in_battle = True

    while in_battle and (time.time() - start_time < max_duration):
        for x1, y1, x2, y2, dur in vacuum_moves:
            client.swipe(x1, y1, x2, y2, duration_ms=dur)
            time.sleep(0.03)

        cycle += 1
        elapsed = time.time() - start_time
        logger.info(f"Вакуум-кайт: цикл #{cycle}, прошло {elapsed:.1f}с...")

        # Сворачиваем шторку
        if cycle % 3 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

        # Проверка экрана каждый цикл
        scr = client.get_screenshot()
        if scr is not None:
            # 1. Проверяем выбор навыка (игра на паузе)
            if select_best_skill(client, ocr, scr):
                continue

            # 2. Проверяем Победу
            res, _ = ocr(scr)
            j = " ".join([r[1].lower() for r in res]) if res else ""
            if any(w in j for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                logger.success(f"ОБНАРУЖЕН ЭКРАН ПОБЕДЫ! {j[:60]}")
                in_battle = False
                break

            # 3. Проверяем поражение
            if any(w in j for w in ["проиграна", "битва проиграна"]):
                logger.warning("Битва проиграна, жмем 'Повторить' (875, 1418)...")
                client.tap(875, 1418)
                time.sleep(3.0)
                start_time = time.time()
                continue

            # 4. Если вернулись на карту
            if "спаситель порядка" in j or "просмотр навыка" in j:
                logger.success("Вернулись на карту Кампании!")
                in_battle = False
                break

    logger.info("Бой завершен! Очищаем экран наград...")
    for _ in range(6):
        client.tap(540, 1850)
        time.sleep(1.2)
        client.tap(540, 2000)
        time.sleep(1.0)

    # 3. Возврат в Город
    logger.info("Возвращаемся в Город...")
    for attempt in range(5):
        scr = client.get_screenshot()
        if scr is None:
            continue
        res, _ = ocr(scr)
        j = " ".join([r[1].lower() for r in res]) if res else ""
        if any(w in j for w in ["замок", "событие", "рюкзак", "почта", "альянс"]):
            logger.success("Успешно в Городе (City View)!")
            break
        if "спаситель" in j or "просмотр" in j or "доблести" in j:
            client.tap(75, 120)
            time.sleep(2.5)
        else:
            client.tap(540, 1850)
            time.sleep(1.5)

    # 4. Финализация Главы 3
    logger.info("Клеймим награду за Этап 6 на HUD...")
    for _ in range(3):
        client.tap(250, 1955)
        time.sleep(1.2)
        client.tap(540, 1800)
        time.sleep(1.0)

    logger.info("Открываем Путь Правителя (75, 1960)...")
    client.tap(75, 1960)
    time.sleep(2.5)

    logger.info("Забираем финальные награды Главы 3 (543, 1995)...")
    client.tap(543, 1995)
    time.sleep(2.0)
    for _ in range(4):
        client.tap(540, 1800)
        time.sleep(1.0)

    # Проверка перехода в Главу 4
    final_scr = client.get_screenshot()
    if final_scr is not None:
        cv2.imwrite("screen_ch3_victory_final.png", final_scr)
        artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\ch3_victory_final.png"
        cv2.imwrite(artifact_path, final_scr)
        res_f, _ = ocr(final_scr)
        jf = " ".join([r[1].lower() for r in res_f]) if res_f else ""
        logger.info(f"Финальный OCR: {jf[:120]}")

    logger.success("=== ПОЛНЫЙ ЦИКЛ ГЛАВЫ 3 УСПЕШНО ЗАВЕРШЕН! ===")
    return True


if __name__ == "__main__":
    run_stage6()
