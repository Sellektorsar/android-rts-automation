"""
Окончательный скрипт прохождения Этапа 8 Кампании:
- Старт с карты Кампании (узел 853, 1152 -> Испытание) либо подхват активного боя.
- Обработка сундуков сокровищ мини-боссов (Найти сокровища -> Собрать 540, 1939).
- Обработка выбора навыков (Дракон 100 > Пушка 95 > Жар-птица 90 > Молния 85 > Щит 80).
- Вакуумный кайт Figure-8.
- Верификация победы и сбор финальной награды Главы 5.
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
from tools.run_campaign_via_road import select_best_skill, collect_victory_rewards, return_to_city_and_claim, try_claim_chapter_reward

PAUSE_TMPL = cv2.imread("assets/templates/battle_pause.png")
PAUSE_FAST_TMPL = cv2.imread("assets/templates/battle_pause_fast.png")
CONFIRM_TMPL = cv2.imread("assets/templates/btn_skill_confirm.png")
TITLE_TMPL = cv2.imread("assets/templates/title_skill_choice.png")


def is_in_battle(screen) -> bool:
    if screen is None:
        return False
    if PAUSE_TMPL is not None:
        res = cv2.matchTemplate(screen, PAUSE_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        if max_v >= 0.68:
            return True
    if PAUSE_FAST_TMPL is not None:
        res2 = cv2.matchTemplate(screen, PAUSE_FAST_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v2, _, _ = cv2.minMaxLoc(res2)
        if max_v2 >= 0.68:
            return True
    return False


def check_skill_screen(screen, ocr) -> bool:
    if screen is None:
        return False
    if CONFIRM_TMPL is not None:
        res = cv2.matchTemplate(screen, CONFIRM_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        if max_v >= 0.72:
            return True
    if TITLE_TMPL is not None:
        res = cv2.matchTemplate(screen, TITLE_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        if max_v >= 0.70:
            return True
    res_ocr, _ = ocr(screen[300:700, :])
    j = " ".join([r[1].lower() for r in res_ocr]) if res_ocr else ""
    return any(w in j for w in ["выбор", "навык", "bbl6op", "подтвердить"])


def run_stage8_definitive():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(dev_cfg["adb_path"], dev_cfg["ip"], dev_cfg["port"], timeout=10)
    if not client.connect():
        return False

    ocr = RapidOCR()
    logger.info("=" * 65)
    logger.info("  🚀 DEFINITIVE STAGE 8 VICTORY RUN")
    logger.info("=" * 65)

    scr_init = client.get_screenshot()
    in_combat = is_in_battle(scr_init)

    if in_combat:
        logger.success("⚔️ Уже находимся в бою! Подхватываем активное управление...")
    else:
        # 1. Открываем узел Этапа 8 (853, 1152)
        logger.info("Тапаем узел Этапа 8 (853, 1152)...")
        client.tap(853, 1152)
        time.sleep(2.5)

        # 2. Нажимаем кнопку 'Испытание' в брифинге
        screen = client.get_screenshot()
        btn = find_green_button(screen) if screen is not None else None
        if btn:
            logger.success(f"🟢 Зелёная кнопка найдена: {btn}. Нажимаем...")
            client.tap(btn[0], btn[1])
        else:
            logger.warning("Тапаем (542, 1842)...")
            client.tap(542, 1842)
        time.sleep(2.0)

        # 3. Проверяем попап подтверждения
        screen = client.get_screenshot()
        res_pop, _ = ocr(screen)
        j_pop = " ".join([r[1].lower() for r in res_pop]) if res_pop else ""
        if "увеличить силу" in j_pop or "укреплению" in j_pop or "ykpenne" in j_pop or "опыт" in j_pop:
            logger.info("Попап предупреждения: жмём правую кнопку 'Испытание' (764, 1338)...")
            client.tap(764, 1338)
            time.sleep(2.5)

        # 4. Верификация входа в бой
        logger.info("Верификация входа в бой...")
        for attempt in range(10):
            time.sleep(1.0)
            scr_c = client.get_screenshot()
            if is_in_battle(scr_c):
                in_combat = True
                logger.success(f"🎉 ВХОД В БОЙ ПОДТВЕРЖДЁН на попытке {attempt+1}!")
                break
            if attempt % 3 == 0:
                client.tap(764, 1338)

    if not in_combat:
        logger.error("❌ Не удалось подтвердить бой!")
        return False

    # 5. Движения Vacuum Figure-8
    moves = [
        (540, 1980, 800, 1720, 750),   # Вверх-вправо
        (540, 1980, 850, 1980, 650),   # Вправо
        (540, 1980, 270, 2220, 900),   # Диагональный вакуум вниз-влево
        (540, 1980, 220, 1980, 650),   # Влево
        (540, 1980, 270, 1720, 750),   # Вверх-влево
        (540, 1980, 800, 2220, 900),   # Диагональный вакуум вниз-вправо
        (540, 1980, 540, 1620, 750),   # Вверх
        (540, 1980, 540, 2320, 750),   # Вниз через центр
    ]

    start_time = time.time()
    max_duration = 720
    step = 0
    battle_result = "timeout"

    logger.info("⚔️ БОЙ НАЧАЛСЯ! Vacuum Figure-8 активен...")

    while time.time() - start_time < max_duration:
        m = moves[step % len(moves)]
        client.swipe(m[0], m[1], m[2], m[3], duration_ms=m[4])
        step += 1
        time.sleep(0.02)

        if step % 2 == 0:
            scr = client.get_screenshot()
            if scr is None:
                continue

            res_s, _ = ocr(scr)
            j_s = " ".join([r[1].lower() for r in res_s]) if res_s else ""

            # А: Проверка сундука 'Найти сокровища' / 'Собрать'
            if any(w in j_s for w in ["сокровища", "cokpob", "собрать", "co6patb"]):
                logger.success("🎁 СУНДУК СОКРОВИЩ! Жмём 'Собрать' (540, 1939)...")
                client.tap(540, 1939)
                time.sleep(0.8)
                client.tap(540, 1800)
                time.sleep(0.5)
                continue

            # Б: Проверка выбора навыков при level-up
            if check_skill_screen(scr, ocr):
                logger.info("⚡ ВЫБОР НАВЫКА!")
                select_best_skill(client, ocr, scr)
                continue

            # В: Проверка ПОБЕДЫ над боссом
            if any(w in j_s for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                logger.success(f"🏆 ПОБЕДА НАД ЛИДЕРОМ ЭТАПА 8! ({time.time() - start_time:.0f}с)")
                cv2.imwrite("logs/inspect/stage8_final_victory.png", scr)
                battle_result = "victory"
                break

            # Г: Поражение
            if any(w in j_s for w in ["битва проиграна", "проиграна", "gutba"]):
                logger.warning("💀 Поражение. Нажимаем 'Повторить' (875, 1419)...")
                client.tap(875, 1419)
                time.sleep(3.0)
                start_time = time.time()
                step = 0
                continue

            # Д: Возврат на карту кампании
            if "спаситель порядка" in j_s or "просмотр навыка" in j_s:
                logger.success("Бой завершён, возврат на карту Кампании.")
                battle_result = "victory"
                break

        if step % 8 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Бой: шаг #{step}, прошло {elapsed:.0f}с...")
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

    # 6. Сбор наград победы
    collect_victory_rewards(client, ocr)

    # 7. Возврат в Город и сбор награды квеста с HUD
    return_to_city_and_claim(client, ocr)

    # 8. Забираем финальную награду Главы 5 в Королевской Дороге!
    try_claim_chapter_reward(client, ocr)

    logger.success("=" * 65)
    logger.success(f"  ✅ ИТОГ ПРОХОЖДЕНИЯ ЭТАПА 8: {battle_result}")
    logger.success("=" * 65)
    return battle_result == "victory"


if __name__ == "__main__":
    run_stage8_definitive()
