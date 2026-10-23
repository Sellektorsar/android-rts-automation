"""
Универсальный пайплайн прохождения этапа Кампании через Королевскую Дорогу.
Полный цикл:
1. Навигация: Город -> Королевская Дорога -> кнопка 'Вперед' у нужного квеста.
2. Переход на карту Кампании и нажатие зеленой кнопки 'Испытание'.
3. Бой с использованием Vacuum Figure-8 траектории + интеллектуальный выбор навыков.
4. Сбор наград победы.
5. Возврат в Город и сбор награды с HUD баннера.
6. Сбор финальной награды главы (если все квесты выполнены).

Использование:
    python tools/run_campaign_via_road.py          # по умолчанию этап 7
    python tools/run_campaign_via_road.py --stage 8
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient
from tools.complete_campaign_stage import find_green_button


# ──────────────────────────────────────────────
# Интеллектуальный выбор навыка (из run_stage6_master.py)
# ──────────────────────────────────────────────
def select_best_skill(client: AdbClient, ocr: RapidOCR, screen) -> bool:
    """Выбирает лучший навык при level-up. Возвращает True, если экран навыков был обработан."""
    res, _ = ocr(screen)
    joined = " ".join([r[1].lower() for r in res]) if res else ""
    if not any(w in joined for w in ["выбор навыка", "bbl6op habblka", "подтвердить", "oatbepantb"]):
        return False

    logger.info("🎴 Обнаружен экран выбора навыка! Анализируем карты...")

    card_boxes = {
        1: (80, 800, 360, 1300),
        2: (380, 800, 700, 1300),
        3: (720, 800, 1020, 1300),
    }

    card_scores = {1: 1, 2: 2, 3: 1}

    for c_idx, (x1, y1, x2, y2) in card_boxes.items():
        card_crop = screen[y1:y2, x1:x2]
        c_res, _ = ocr(card_crop)
        c_text = " ".join([r[1].lower() for r in c_res]) if c_res else ""
        logger.info(f"  Карта #{c_idx}: {c_text[:60]}")

        # Приоритет навыков по урону по боссу и сбору кристаллов:
        if any(k in c_text for k in ["дракон", "apakoh", "dragon"]):
            card_scores[c_idx] = 100
        elif any(k in c_text for k in ["пушк", "xmm", "алхим", "cannon"]):
            card_scores[c_idx] = 95
        elif any(k in c_text for k in ["жар-птиц", "kap-ntw", "птиц", "bird"]):
            card_scores[c_idx] = 90
        elif any(k in c_text for k in ["молн", "moahn", "цепь", "разряд"]):
            card_scores[c_idx] = 85
        elif any(k in c_text for k in ["магнит", "притяж", "радиус", "сбор", "magnet"]):
            card_scores[c_idx] = 80
        elif any(k in c_text for k in ["щит", "божеств", "shield"]):
            card_scores[c_idx] = 75
        elif any(k in c_text for k in ["лед", "neaf", "frost"]):
            card_scores[c_idx] = 70
        else:
            card_scores[c_idx] = 10

    best_card = max(card_scores, key=card_scores.get)
    card_tap_x = {1: 220, 2: 540, 3: 870}[best_card]

    logger.success(f"  → Выбираем Карту #{best_card} (счёт {card_scores[best_card]})")
    client.tap(card_tap_x, 1100)
    time.sleep(0.5)
    client.tap(541, 1744)  # Подтвердить
    time.sleep(1.0)
    return True


# ──────────────────────────────────────────────
# Проверка текущего состояния экрана (OCR)
# ──────────────────────────────────────────────
def detect_screen(client: AdbClient, ocr: RapidOCR) -> tuple[str, str]:
    """Возвращает (тип_экрана, ocr_текст)."""
    screen = client.get_screenshot()
    if screen is None:
        return "unknown", ""
    res, _ = ocr(screen)
    joined = " ".join([r[1].lower() for r in res]) if res else ""

    # Проверяем попап «Выйти из игры?»
    if "выйти" in joined or "bblitu" in joined or "w3 wrpbl" in joined:
        logger.info("Обнаружен попап 'Выйти из игры?', жмём Нет...")
        client.tap(765, 1338)
        time.sleep(1.5)
        return "city", joined

    # City View — нижняя навигация (OCR часто даёт латинские аналоги кириллицы)
    city_markers = ["замок", "событие", "рюкзак", "почта", "альянс", "задание", "адванс",
                    "co6bitm", "co6bitwm", "co66itw", "piok3ak", "3aaahme", "3aaahue",
                    "oyta", "aabahc", "0aopae", "oadopae"]
    if any(w in joined for w in city_markers):
        return "city", joined

    # Карта Кампании «Спаситель порядка»
    campaign_markers = ["спаситель", "cnacmteab", "cnacuteab", "просмотр навыка", "pocmotphabblka",
                        "доблести", "a06aectm", "ao6aecth", "укрепление", "ykpennehhe", "ykpennehue",
                        "ykpenen", "repo3ma", "repow3ma"]
    if any(w in joined for w in campaign_markers):
        return "campaign_map", joined

    if any(w in joined for w in ["королевская дорога", "kopolebcka", "kopolebckaa", "забрать награды", "3abpatb",
                                  "3abpатb", "глава", "rnaba", "fnaba"]):
        return "royal_road", joined

    if any(w in joined for w in ["испытание", "wcnbitahm", "wcnbitahma", "wcnbitahmg"]):
        return "briefing_or_map", joined

    if any(w in joined for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
        return "victory", joined

    if any(w in joined for w in ["проиграна", "битва проиграна"]):
        return "defeat", joined
    return "unknown", joined


# ──────────────────────────────────────────────
# Шаг 1: Навигация в Город
# ──────────────────────────────────────────────
def ensure_city_view(client: AdbClient, ocr: RapidOCR) -> bool:
    """Гарантирует что мы в City View. Максимум 6 попыток."""
    for attempt in range(6):
        stype, text = detect_screen(client, ocr)
        if stype == "city":
            logger.success("✅ Мы в Городе (City View)")
            return True
        if stype in ["campaign_map", "briefing_or_map"]:
            logger.info(f"  Попытка {attempt+1}: на карте Кампании, жмём Назад...")
            client.tap(75, 120)
            time.sleep(2.0)
        elif stype == "royal_road":
            logger.info(f"  Попытка {attempt+1}: в Королевской Дороге, жмём Back...")
            client.press_back()
            time.sleep(1.5)
        else:
            logger.info(f"  Попытка {attempt+1}: неизвестный экран, жмём Back...")
            client.press_back()
            time.sleep(1.5)
    return False


# ──────────────────────────────────────────────
# Шаг 2: Открыть Королевскую Дорогу и нажать «Вперёд»
# ──────────────────────────────────────────────
def navigate_via_royal_road(client: AdbClient, ocr: RapidOCR, stage_num: int = 7) -> bool:
    """Открывает Королевскую Дорогу и нажимает кнопку 'Вперёд' у квеста этапа кампании."""
    logger.info("📜 Открываем Королевскую Дорогу (75, 1960)...")
    client.tap(75, 1960)
    time.sleep(2.5)

    screen = client.get_screenshot()
    if screen is None:
        return False

    res, _ = ocr(screen)
    # 1. Поиск строки квеста кампании с нужным этапом
    stage_str = str(stage_num)
    target_cy = None
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        if ("этап" in txt or "3tan" in txt or "спасител" in txt or "cnacmte" in txt):
            if stage_str in txt:
                target_cy = cy
                logger.info(f"Найден квест этапа {stage_num}: '{text}' на высоте y={cy}")
                break
            elif target_cy is None:
                target_cy = cy

    if target_cy:
        logger.success(f"🔘 Нажимаем 'Вперёд' для этапа {stage_num} (870, {target_cy})...")
        client.tap(870, target_cy)
        time.sleep(3.0)
        return True

    # 2. Ищем любую кнопку «Вперёд» у незавершённого квеста (справа, cx > 750)
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if ("вперед" in txt or "bnepe" in txt) and cx > 750 and 800 <= cy <= 1500:
            logger.success(f"🔘 Нажимаем 'Вперёд' ({cx}, {cy})...")
            client.tap(cx, cy)
            time.sleep(3.0)
            return True

    # Запасной вариант — координата квеста этапа 8 (870, 1252)
    logger.warning("Кнопка 'Вперёд' не найдена OCR, нажимаем по координатам (870, 1252)...")
    client.tap(870, 1252)
    time.sleep(3.0)
    return True


# ──────────────────────────────────────────────
# Шаг 3: Запуск боя (Испытание)
# ──────────────────────────────────────────────
def start_battle(client: AdbClient, ocr: RapidOCR) -> bool:
    """Находит и нажимает зелёную кнопку 'Испытание' на карте Кампании."""
    for attempt in range(3):
        screen = client.get_screenshot()
        if screen is None:
            time.sleep(1.0)
            continue

        # Пробуем найти зелёную кнопку через HSV
        btn = find_green_button(screen)
        if btn:
            logger.success(f"🟢 Зелёная кнопка 'Испытание' найдена: {btn}")
            client.tap(btn[0], btn[1])
            time.sleep(4.0)
            return True

        # Проверяем, может мы на карте — нажимаем справа-внизу «Испытание доблести»
        res, _ = ocr(screen)
        joined = " ".join([r[1].lower() for r in res]) if res else ""
        if "wcnbitahm" in joined or "испытание" in joined or "доблести" in joined:
            logger.info(f"  Попытка {attempt+1}: нажимаем 'Испытание доблести' (969, 1976)...")
            client.tap(969, 1976)
            time.sleep(3.0)

            # Теперь должен открыться брифинг, ищем зелёную кнопку
            screen2 = client.get_screenshot()
            if screen2 is not None:
                btn2 = find_green_button(screen2)
                if btn2:
                    logger.success(f"🟢 Кнопка 'Испытание' в брифинге: {btn2}")
                    client.tap(btn2[0], btn2[1])
                    time.sleep(4.0)
                    return True
            # Фоллбэк
            client.tap(543, 1885)
            time.sleep(4.0)
            return True

    # Финальный фоллбэк
    logger.warning("Не удалось найти кнопку, жмём по координатам (543, 1885)...")
    client.tap(543, 1885)
    time.sleep(4.0)
    return True


# ──────────────────────────────────────────────
# Шаг 4: Боевой цикл — Vacuum Figure-8 + выбор навыков
# ──────────────────────────────────────────────
def run_battle(client: AdbClient, ocr: RapidOCR, max_duration: int = 180) -> str:
    """
    Запускает боевой цикл Figure-8 Vacuum Kite.
    Возвращает: 'victory', 'defeat', 'timeout'.
    """
    logger.info("⚔️ БОЙ НАЧАЛСЯ! Vacuum Figure-8 Kite активирован...")

    vacuum_moves = [
        (540, 1980, 740, 1780, 650),   # Вверх-вправо
        (540, 1980, 840, 1980, 600),   # Вправо
        (540, 1980, 340, 2180, 800),   # Пересечение центра вниз-влево
        (540, 1980, 240, 1980, 600),   # Влево
        (540, 1980, 340, 1780, 650),   # Вверх-влево
        (540, 1980, 740, 2180, 800),   # Пересечение центра вниз-вправо
        (540, 1980, 540, 1680, 650),   # Вверх
        (540, 1980, 540, 2280, 650),   # Вниз через центр
    ]

    start_time = time.time()
    cycle = 0
    result = "timeout"

    while time.time() - start_time < max_duration:
        # Выполняем один полный цикл Figure-8
        for x1, y1, x2, y2, dur in vacuum_moves:
            client.swipe(x1, y1, x2, y2, duration_ms=dur)
            time.sleep(0.03)

        cycle += 1
        elapsed = time.time() - start_time
        logger.info(f"  Вакуум-кайт: цикл #{cycle}, прошло {elapsed:.0f}с")

        # Сворачиваем шторку каждые 3 цикла
        if cycle % 3 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

        # Проверяем экран
        scr = client.get_screenshot()
        if scr is None:
            continue

        # 1. Проверяем выбор навыка
        if select_best_skill(client, ocr, scr):
            continue

        # 2. Проверяем состояние через OCR
        res, _ = ocr(scr)
        j = " ".join([r[1].lower() for r in res]) if res else ""

        if any(w in j for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
            logger.success(f"🏆 ПОБЕДА! (цикл #{cycle}, {elapsed:.0f}с)")
            result = "victory"
            break

        if any(w in j for w in ["проиграна", "битва проиграна"]):
            logger.warning("💀 Поражение! Нажимаем 'Повторить' (875, 1418)...")
            client.tap(875, 1418)
            time.sleep(3.0)
            start_time = time.time()
            cycle = 0
            continue

        if "спаситель порядка" in j or "просмотр навыка" in j:
            logger.success("Бой завершён, мы на карте Кампании.")
            result = "victory"
            break

    return result


# ──────────────────────────────────────────────
# Шаг 5: Сбор наград после боя
# ──────────────────────────────────────────────
def collect_victory_rewards(client: AdbClient, ocr: RapidOCR):
    """Закрывает экраны победы и лута."""
    logger.info("🎁 Собираем награды победы...")
    for _ in range(8):
        client.tap(540, 1850)
        time.sleep(1.0)
        client.tap(540, 2000)
        time.sleep(0.8)


# ──────────────────────────────────────────────
# Шаг 6: Возврат в Город и сбор награды HUD
# ──────────────────────────────────────────────
def return_to_city_and_claim(client: AdbClient, ocr: RapidOCR):
    """Возвращается в Город и собирает награду с HUD баннера."""
    logger.info("🏙️ Возвращаемся в Город...")
    if not ensure_city_view(client, ocr):
        logger.warning("Не удалось вернуться в Город автоматически.")

    # Клеймим награду на HUD баннере
    logger.info("📋 Проверяем HUD баннер (250, 1955)...")
    for _ in range(3):
        client.tap(250, 1955)
        time.sleep(1.5)
        client.tap(540, 1800)
        time.sleep(1.0)


# ──────────────────────────────────────────────
# Шаг 7: Проверяем финальную награду главы
# ──────────────────────────────────────────────
def try_claim_chapter_reward(client: AdbClient, ocr: RapidOCR):
    """Открывает Королевскую Дорогу и пробует забрать финальную награду главы."""
    logger.info("👑 Проверяем финальную награду главы...")
    client.tap(75, 1960)
    time.sleep(2.5)

    screen = client.get_screenshot()
    if screen is None:
        return

    res, _ = ocr(screen)
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if ("забрать" in txt or "3abpatb" in txt) and ("наград" in txt or "harpa" in txt) and cy > 1800:
            logger.success(f"🏅 Финальная награда главы найдена! ({cx}, {cy})")
            client.tap(cx, cy)
            time.sleep(2.5)
            for _ in range(4):
                client.tap(540, 1800)
                time.sleep(1.0)
            client.press_back()
            time.sleep(1.5)
            return

    logger.info("Финальная награда главы пока недоступна.")
    client.press_back()
    time.sleep(1.5)


# ──────────────────────────────────────────────
# ГЛАВНЫЙ ПАЙПЛАЙН
# ──────────────────────────────────────────────
def run_campaign_pipeline(stage_num: int = 7):
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
        logger.error("❌ Не удалось подключиться к ADB!")
        return False

    ocr = RapidOCR()

    logger.info("=" * 50)
    logger.info(f"  🚀 ЗАПУСК ПАЙПЛАЙНА ЭТАПА {stage_num} КАМПАНИИ")
    logger.info("=" * 50)

    # ── Шаг 1: Гарантируем City View ──
    if not ensure_city_view(client, ocr):
        logger.error("Не удалось перейти в City View!")
        return False

    # ── Шаг 2: Королевская Дорога → Вперёд ──
    navigate_via_royal_road(client, ocr, stage_num)

    # ── Шаг 3: Запуск боя ──
    start_battle(client, ocr)

    # ── Шаг 4: Боевой цикл ──
    battle_result = run_battle(client, ocr, max_duration=180)

    # ── Шаг 5: Сбор наград ──
    collect_victory_rewards(client, ocr)

    # ── Шаг 6: Возврат в Город + HUD ──
    return_to_city_and_claim(client, ocr)

    # ── Шаг 7: Финальная награда главы ──
    try_claim_chapter_reward(client, ocr)

    logger.success("=" * 50)
    logger.success(f"  ✅ ЭТАП {stage_num} — ПАЙПЛАЙН ЗАВЕРШЁН ({battle_result})")
    logger.success("=" * 50)
    return battle_result == "victory"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Универсальный пайплайн прохождения этапа кампании")
    parser.add_argument("--stage", type=int, default=7, help="Номер этапа (по умолчанию 7)")
    args = parser.parse_args()
    run_campaign_pipeline(args.stage)
