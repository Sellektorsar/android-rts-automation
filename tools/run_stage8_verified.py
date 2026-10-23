"""
Окончательное прохождение Этапа 8 Кампании с прокачанным героем (Lv.5, +50% ATK, +50% HP).
Последовательность:
1. Выход из 'Укрепления героизма' по стрелке (105, 170) на карту кампании.
2. Клик по узлу этапа 8 (853, 1152).
3. Клик по 'Испытание' в брифинге (542, 1842).
4. Если вылезет попап предупреждения — клик по правой кнопке 'Испытание' (764, 1338).
5. Верификация входа в бой по шаблону battle_pause.
6. Боевой цикл Vacuum Figure-8 с интеллектуальным выбором навыков (Магнит > Пушка > Щит > Меч).
7. Сбор наград победы, возврат в Город и сбор финальной награды Главы 5!
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
        _, max_v, _, max_pt = cv2.minMaxLoc(res)
        if max_v >= 0.68:
            return True
    if PAUSE_FAST_TMPL is not None:
        res2 = cv2.matchTemplate(screen, PAUSE_FAST_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v2, _, _ = cv2.minMaxLoc(res2)
        if max_v2 >= 0.68:
            return True
    return False


def check_and_handle_skill(client: AdbClient, ocr: RapidOCR, screen) -> bool:
    if screen is None:
        return False
    # Быстрый темплейт-матч на кнопку подтверждения
    has_skill_screen = False
    if CONFIRM_TMPL is not None:
        res = cv2.matchTemplate(screen, CONFIRM_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        if max_v >= 0.72:
            has_skill_screen = True
    if not has_skill_screen and TITLE_TMPL is not None:
        res = cv2.matchTemplate(screen, TITLE_TMPL, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, _ = cv2.minMaxLoc(res)
        if max_v >= 0.70:
            has_skill_screen = True

    if not has_skill_screen:
        res_ocr, _ = ocr(screen[300:700, :])
        j = " ".join([r[1].lower() for r in res_ocr]) if res_ocr else ""
        if any(w in j for w in ["выбор", "навык", "bbl6op", "подтвердить"]):
            has_skill_screen = True

    if has_skill_screen:
        logger.info("⚡ Экран выбора навыка активен! Выбираем навык...")
        select_best_skill(client, ocr, screen)
        return True
    return False


def run_stage8_verified():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(dev_cfg["adb_path"], dev_cfg["ip"], dev_cfg["port"], timeout=10)
    if not client.connect():
        return False

    ocr = RapidOCR()
    logger.info("=" * 60)
    logger.info("  🚀 ЗАПУСК ВЕРИФИЦИРОВАННОГО БОЯ ЭТАПА 8 (ГЕРОЙ LV.5)")
    logger.info("=" * 60)

    # 1. Если мы на экране Укрепления героизма — жмем стрелку назад
    screen = client.get_screenshot()
    res, _ = ocr(screen)
    j = " ".join([r[1].lower() for r in res]) if res else ""
    if "укрепление" in j or "ykpenne" in j or "lv.5" in j or "развивать" in j:
        logger.info("Возвращаемся на карту кампании (стрелка 105, 170)...")
        client.tap(105, 170)
        time.sleep(2.0)

    # 2. Тапаем узел этапа 8 (853, 1152)
    logger.info("Тапаем узел Этапа 8 (853, 1152)...")
    client.tap(853, 1152)
    time.sleep(2.5)

    # 3. Нажимаем кнопку 'Испытание' в брифинге
    screen = client.get_screenshot()
    btn = find_green_button(screen) if screen is not None else None
    if btn:
        logger.success(f"🟢 Зелёная кнопка 'Испытание' найдена: {btn}")
        client.tap(btn[0], btn[1])
    else:
        logger.warning("Тапаем 'Испытание' по координатам (542, 1842)...")
        client.tap(542, 1842)
    time.sleep(2.0)

    # 4. Проверяем, не появился ли попап с предупреждением
    screen = client.get_screenshot()
    res_pop, _ = ocr(screen)
    j_pop = " ".join([r[1].lower() for r in res_pop]) if res_pop else ""
    if "увеличить силу" in j_pop or "опыт" in j_pop or "укреплению" in j_pop or "ykpenne" in j_pop:
        logger.info("Обнаружен попап подтверждения! Жмём правую кнопку 'Испытание' (764, 1338)...")
        client.tap(764, 1338)
        time.sleep(2.5)

    # 5. Верифицируем вход в бой
    logger.info("Ожидание входа в бой (проверка battle_pause)...")
    in_combat = False
    for attempt in range(12):
        time.sleep(1.0)
        scr_c = client.get_screenshot()
        if is_in_battle(scr_c):
            in_combat = True
            logger.success(f"🎉 ВХОД В БОЙ ПОДТВЕРЖДЁН на попытке {attempt+1}!")
            cv2.imwrite("logs/inspect/battle_started_confirmed.png", scr_c)
            break
        # Если попап всё ещё висит, жмём (764, 1338)
        if attempt % 3 == 0:
            client.tap(764, 1338)

    if not in_combat:
        logger.error("❌ Не удалось подтвердить вход в бой!")
        return False

    # 6. БОЕВОЙ ЦИКЛ: Vacuum Figure-8 + Deep Crystal Sweep
    moves = [
        (540, 1980, 800, 1720, 750),   # Вверх-вправо
        (540, 1980, 850, 1980, 650),   # Вправо
        (540, 1980, 270, 2220, 900),   # Диагональный вакуум через центр вниз-влево
        (540, 1980, 220, 1980, 650),   # Влево
        (540, 1980, 270, 1720, 750),   # Вверх-влево
        (540, 1980, 800, 2220, 900),   # Диагональный вакуум через центр вниз-вправо
        (540, 1980, 540, 1620, 750),   # Вверх
        (540, 1980, 540, 2320, 750),   # Вниз через центр
    ]

    start_time = time.time()
    max_duration = 300
    step = 0
    battle_result = "timeout"

    logger.info("⚔️ БОЙ В РАЗГАРЕ! Vacuum Kite активирован...")

    while time.time() - start_time < max_duration:
        m = moves[step % len(moves)]
        client.swipe(m[0], m[1], m[2], m[3], duration_ms=m[4])
        step += 1
        time.sleep(0.02)

        # Каждые 2 движения проверяем экран
        if step % 2 == 0:
            scr = client.get_screenshot()
            if scr is None:
                continue

            # Проверка навыков
            if check_and_handle_skill(client, ocr, scr):
                continue

            # Проверка победы / поражения
            res_s, _ = ocr(scr)
            j_s = " ".join([r[1].lower() for r in res_s]) if res_s else ""

            if any(w in j_s for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                logger.success(f"🏆 ПОБЕДА НАД БОССОМ ЭТАПА 8! ({time.time() - start_time:.0f}с)")
                cv2.imwrite("logs/inspect/stage8_victory_screen.png", scr)
                battle_result = "victory"
                break

            if any(w in j_s for w in ["проиграна", "битва проиграна"]):
                logger.warning("💀 Поражение! Жмём Повторить (875, 1418)...")
                client.tap(875, 1418)
                time.sleep(3.0)
                start_time = time.time()
                step = 0
                continue

            if "спаситель порядка" in j_s or "просмотр навыка" in j_s:
                logger.success("Бой завершён, возврат на карту Кампании.")
                battle_result = "victory"
                break

        if step % 8 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Бой: шаг #{step}, прошло {elapsed:.0f}с...")
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

    # 7. Сбор наград победы
    collect_victory_rewards(client, ocr)

    # 8. Возврат в Город и сбор награды квеста с HUD баннера
    return_to_city_and_claim(client, ocr)

    # 9. Сбор финальной награды Главы 5 в Королевской Дороге!
    try_claim_chapter_reward(client, ocr)

    logger.success("=" * 60)
    logger.success(f"  ✅ ПАЙПЛАЙН ЭТАПА 8 ЗАВЕРШЁН: {battle_result}")
    logger.success("=" * 60)
    return battle_result == "victory"


if __name__ == "__main__":
    run_stage8_verified()
