"""
Диагностический скрипт боя Этапа 8 с визуальным логированием и надежным выбором навыков.
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
from tools.complete_campaign_stage import find_green_button
from tools.run_campaign_via_road import collect_victory_rewards, return_to_city_and_claim, try_claim_chapter_reward

CONFIRM_TMPL = cv2.imread("assets/templates/btn_skill_confirm.png")
TITLE_TMPL = cv2.imread("assets/templates/title_skill_choice.png")


def check_skill_screen(screen, ocr) -> tuple[bool, str]:
    """Проверяет наличие экрана выбора навыка через шаблоны и OCR."""
    if screen is None:
        return False, ""

    # 1. Проверка по шаблону кнопки Подтвердить
    if CONFIRM_TMPL is not None:
        res = cv2.matchTemplate(screen, CONFIRM_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_pt = cv2.minMaxLoc(res)
        if max_v >= 0.75:
            return True, f"template_confirm ({max_v:.3f} at {max_pt})"

    # 2. Проверка по шаблону заголовка выбора навыка
    if TITLE_TMPL is not None:
        res = cv2.matchTemplate(screen, TITLE_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_pt = cv2.minMaxLoc(res)
        if max_v >= 0.72:
            return True, f"template_title ({max_v:.3f} at {max_pt})"

    # 3. OCR проверка
    res, _ = ocr(screen[300:700, :])
    j = " ".join([r[1].lower() for r in res]) if res else ""
    if any(w in j for w in ["выбор", "навык", "bbl6op", "habblka", "подтвердить"]):
        return True, f"ocr_title: {j[:40]}"

    return False, ""


def select_skill(client: AdbClient, ocr: RapidOCR, screen):
    """Выбирает лучший навык: Карта 2 по умолчанию, или Магнит/Пушка."""
    logger.info("🎴 ОБНАРУЖЕН ЭКРАН ВЫБОРА НАВЫКА! Анализ карт...")
    card_boxes = {
        1: (80, 800, 360, 1300),
        2: (380, 800, 700, 1300),
        3: (720, 800, 1020, 1300),
    }
    card_scores = {1: 1, 2: 5, 3: 1}  # Карта 2 (центр) имеет приоритет

    for c_idx, (x1, y1, x2, y2) in card_boxes.items():
        card_crop = screen[y1:y2, x1:x2]
        c_res, _ = ocr(card_crop)
        c_text = " ".join([r[1].lower() for r in c_res]) if c_res else ""
        logger.info(f"  Карта #{c_idx}: {c_text[:50]}")

        if any(k in c_text for k in ["магнит", "притяж", "радиус", "сбор", "magnet"]):
            card_scores[c_idx] = 100
        elif any(k in c_text for k in ["пушк", "алхим", "cannon"]):
            card_scores[c_idx] = 60
        elif any(k in c_text for k in ["щит", "божеств", "shield"]):
            card_scores[c_idx] = 50
        elif any(k in c_text for k in ["шторм", "меч", "sword"]):
            card_scores[c_idx] = 40
        elif any(k in c_text for k in ["разряд", "молн", "light"]):
            card_scores[c_idx] = 30

    best_card = max(card_scores, key=card_scores.get)
    card_tap_x = {1: 220, 2: 540, 3: 870}[best_card]

    logger.success(f"  → Выбираем Карту #{best_card} (счёт {card_scores[best_card]})...")
    client.tap(card_tap_x, 1100)
    time.sleep(0.6)
    client.tap(541, 1744)  # Кнопка 'Подтвердить'
    time.sleep(1.0)


def run():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(dev_cfg["adb_path"], dev_cfg["ip"], dev_cfg["port"], timeout=10)
    if not client.connect():
        logger.error("ADB fail")
        return

    ocr = RapidOCR()
    logger.info("=== ЗАПУСК ЭТАПА 8 ЧЕРЕЗ HUD КВЕСТ ===")

    # 1. Кликаем по HUD квесту этапа 8 (360, 1956)
    logger.info("Тапаем квест этапа 8 на баннере (360, 1956)...")
    client.tap(360, 1956)
    time.sleep(2.5)

    # 2. Проверяем брифинг или карту
    screen = client.get_screenshot()
    btn = find_green_button(screen) if screen is not None else None
    if not btn:
        logger.info("Тапаем узел этапа 8 (853, 1152)...")
        client.tap(853, 1152)
        time.sleep(2.0)
        screen = client.get_screenshot()
        btn = find_green_button(screen) if screen is not None else None

    if btn:
        logger.success(f"🟢 Зелёная кнопка найдена: {btn}. Нажимаем...")
        client.tap(btn[0], btn[1])
        time.sleep(4.0)
    else:
        logger.warning("Фоллбэк тапа кнопки 'Испытание' (544, 1840)...")
        client.tap(544, 1840)
        time.sleep(4.0)

    # 3. Боевые движения с активным перемещением
    # Центр джойстика: (540, 1980)
    moves = [
        (540, 1980, 800, 1750, 700),   # Вверх-вправо
        (540, 1980, 850, 1980, 650),   # Вправо
        (540, 1980, 300, 2200, 850),   # Сквозной проход через центр вниз-влево
        (540, 1980, 230, 1980, 650),   # Влево
        (540, 1980, 300, 1750, 700),   # Вверх-влево
        (540, 1980, 780, 2200, 850),   # Сквозной проход через центр вниз-вправо
        (540, 1980, 540, 1650, 700),   # Вверх
        (540, 1980, 540, 2300, 700),   # Вниз через центр
    ]

    start_time = time.time()
    max_duration = 240
    step = 0
    snapshot_idx = 0
    last_snap_time = start_time
    battle_result = "timeout"

    logger.info("⚔️ БОЙ НАЧАЛСЯ!")

    while time.time() - start_time < max_duration:
        m = moves[step % len(moves)]
        client.swipe(m[0], m[1], m[2], m[3], duration_ms=m[4])
        step += 1
        time.sleep(0.02)

        # Каждые 2 свайпа проверяем экран
        if step % 2 == 0:
            scr = client.get_screenshot()
            if scr is None:
                continue

            # Периодический снапшот каждые 12 секунд для контроля
            if time.time() - last_snap_time >= 12.0:
                cv2.imwrite(f"logs/inspect/combat_snap_{snapshot_idx}.png", scr)
                snapshot_idx += 1
                last_snap_time = time.time()

            # 1. Проверяем выбор навыка
            is_skill, reason = check_skill_screen(scr, ocr)
            if is_skill:
                logger.info(f"⚡ Экран навыков обнаружен: {reason}")
                select_skill(client, ocr, scr)
                continue

            # 2. Проверяем победу / поражение / возврат
            res, _ = ocr(scr)
            j = " ".join([r[1].lower() for r in res]) if res else ""

            if any(w in j for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                logger.success(f"🏆 ПОБЕДА! ({time.time() - start_time:.0f}с)")
                cv2.imwrite("logs/inspect/combat_victory.png", scr)
                battle_result = "victory"
                break

            if any(w in j for w in ["проиграна", "битва проиграна"]):
                logger.warning(f"💀 Поражение ({time.time() - start_time:.0f}с)! Рестарт...")
                cv2.imwrite("logs/inspect/combat_defeat.png", scr)
                client.tap(875, 1418)  # Повторить
                time.sleep(3.0)
                start_time = time.time()
                step = 0
                continue

            if "спаситель порядка" in j or "просмотр навыка" in j:
                logger.success("Бой завершён, возврат на карту Кампании.")
                battle_result = "victory"
                break

        if step % 8 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Бой: шаг #{step}, прошло {elapsed:.0f}с...")
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

    # Сбор наград
    collect_victory_rewards(client, ocr)
    # Возврат в город
    return_to_city_and_claim(client, ocr)
    # Сбор финальной награды главы
    try_claim_chapter_reward(client, ocr)

    logger.success(f"=== ЗАВЕРШЕНИЕ ЭТАПА 8: {battle_result} ===")
    return battle_result == "victory"


if __name__ == "__main__":
    run()
