"""
Тестовый пошаговый вход в бой Этапа 8 с верификацией каждого экрана.
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
from tools.complete_campaign_stage import find_green_button

PAUSE_TMPL = cv2.imread("assets/templates/battle_pause.png")
PAUSE_FAST_TMPL = cv2.imread("assets/templates/battle_pause_fast.png")


def is_in_battle(screen) -> bool:
    if screen is None or PAUSE_TMPL is None:
        return False
    res = cv2.matchTemplate(screen, PAUSE_TMPL, cv2.TM_CCOEFF_NORMED)
    _, max_v, _, max_pt = cv2.minMaxLoc(res)
    if max_v >= 0.70:
        logger.success(f"⚔️ BATTLE_PAUSE обнаружен! max_val={max_v:.3f} at {max_pt}")
        return True
    if PAUSE_FAST_TMPL is not None:
        res2 = cv2.matchTemplate(screen, PAUSE_FAST_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v2, _, max_pt2 = cv2.minMaxLoc(res2)
        if max_v2 >= 0.70:
            logger.success(f"⚔️ BATTLE_PAUSE_FAST обнаружен! max_val={max_v2:.3f} at {max_pt2}")
            return True
    return False


def test_enter():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(dev_cfg["adb_path"], dev_cfg["ip"], dev_cfg["port"], timeout=10)
    if not client.connect():
        return

    ocr = RapidOCR()
    logger.info("=== ШАГ 1: Открываем Королевскую Дорогу (75, 1960) ===")
    client.tap(75, 1960)
    time.sleep(2.5)

    scr1 = client.get_screenshot()
    cv2.imwrite("logs/inspect/step1_rr.png", scr1)

    logger.info("=== ШАГ 2: Ищем квест Этапа 8 и жмём 'Вперёд' ===")
    res, _ = ocr(scr1)
    target_vpered = (870, 1100)
    for box, txt, score in (res or []):
        t = txt.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        if ("этап 8" in t or "3tan 8" in t or "спасител" in t) and cy > 800:
            logger.info(f"Найден квест: '{txt}' на y={cy}")
            target_vpered = (870, cy + 25)
            break

    logger.info(f"Тапаем 'Вперёд' {target_vpered}...")
    client.tap(target_vpered[0], target_vpered[1])
    time.sleep(3.0)

    scr2 = client.get_screenshot()
    cv2.imwrite("logs/inspect/step2_centered.png", scr2)

    logger.info("=== ШАГ 3: Тапаем центрированный Портал (540, 1200) ===")
    client.tap(540, 1200)
    time.sleep(3.0)

    scr3 = client.get_screenshot()
    cv2.imwrite("logs/inspect/step3_campaign.png", scr3)
    res3, _ = ocr(scr3)
    j3 = " ".join([r[1].lower() for r in res3]) if res3 else ""
    logger.info(f"Текст карты кампании: {j3[:80]}")

    logger.info("=== ШАГ 4: Тапаем узел Этапа 8 (853, 1152) ===")
    client.tap(853, 1152)
    time.sleep(2.5)

    scr4 = client.get_screenshot()
    cv2.imwrite("logs/inspect/step4_briefing.png", scr4)
    btn = find_green_button(scr4)
    logger.info(f"Зелёная кнопка в брифинге: {btn}")

    if btn:
        logger.success(f"=== ШАГ 5: Нажимаем 'Испытание' {btn} ===")
        client.tap(btn[0], btn[1])
    else:
        logger.warning("Тапаем фоллбэк (544, 1840)...")
        client.tap(544, 1840)

    logger.info("=== ШАГ 6: Ожидание входа в бой (проверка battle_pause) ===")
    in_combat = False
    for attempt in range(10):
        time.sleep(1.0)
        scr_c = client.get_screenshot()
        if is_in_battle(scr_c):
            in_combat = True
            cv2.imwrite("logs/inspect/confirmed_battle_screen.png", scr_c)
            logger.success(f"🎉 БОЙ ВЕРИФИЦИРОВАН на попытке {attempt+1}!")
            break

    if not in_combat:
        scr_fail = client.get_screenshot()
        cv2.imwrite("logs/inspect/combat_enter_failed.png", scr_fail)
        res_f, _ = ocr(scr_fail)
        jf = " ".join([r[1].lower() for r in res_f]) if res_f else ""
        logger.error(f"Не удалось подтвердить вход в бой! Экран: {jf[:100]}")
    else:
        logger.success("БОЙ ПОДТВЕРЖДЁН! Теперь можно запускать управление джойстиком!")


if __name__ == "__main__":
    test_enter()
