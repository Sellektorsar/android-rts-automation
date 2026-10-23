"""
Master Stage 8 Combat Engine:
1. Перезапуск боя по кнопке 'Повторить' (875, 1419) прямо с текущего экрана поражения.
2. Обработка всех экранов паузы во время боя:
   - Экран level-up навыков (анализ карт с приоритетом Удар дракона / Химическая пушка / Жар-птица).
   - Экран 'Найти сокровища' (сундук за мини-босса): мгновенный тап 'Собрать' (540, 1939).
3. Тактика 'Orbit Boss Kite' (круговой кайт вокруг босса на средней дистанции для концентрации всего ДПС на Лидере + вакуумный сбор кристаллов).
4. Фиксация победы над Лидером, сбор наград и финальный сбор Главы 5!
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

CONFIRM_TMPL = cv2.imread("assets/templates/btn_skill_confirm.png")
TITLE_TMPL = cv2.imread("assets/templates/title_skill_choice.png")


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


def master_stage8_win():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(dev_cfg["adb_path"], dev_cfg["ip"], dev_cfg["port"], timeout=10)
    if not client.connect():
        return False

    ocr = RapidOCR()
    logger.info("=" * 65)
    logger.info("  🏆 MASTER STAGE 8 WINNER PIPELINE ACTIVATED")
    logger.info("=" * 65)

    # 1. Проверяем текущий экран
    screen = client.get_screenshot()
    res, _ = ocr(screen)
    j = " ".join([r[1].lower() for r in res]) if res else ""

    if "битва проиграна" in j or "проиграна" in j or "повторить" in j:
        logger.info("Нажимаем 'Повторить' (875, 1419) для мгновенного рестарта боя...")
        client.tap(875, 1419)
        time.sleep(3.5)
    else:
        # Если не на экране поражения, ищем зелёную кнопку 'Испытание'
        btn = find_green_button(screen)
        if btn:
            logger.info(f"Нажимаем 'Испытание' {btn}...")
            client.tap(btn[0], btn[1])
            time.sleep(2.0)
            # Проверяем попап подтверждения
            scr_p = client.get_screenshot()
            res_p, _ = ocr(scr_p)
            jp = " ".join([r[1].lower() for r in res_p]) if res_p else ""
            if "увеличить силу" in jp or "укреплению" in jp:
                client.tap(764, 1338)
            time.sleep(3.0)

    # 2. Тактический круговой кайт вокруг босса (Orbit Boss Kite)
    # Плавные круговые дуги с возвратом в центр для сбора кристаллов:
    # Центр джойстика (540, 1980)
    orbit_moves = [
        (540, 1980, 720, 1800, 600),   # Вправо-вверх
        (540, 1980, 780, 1980, 550),   # Вправо
        (540, 1980, 720, 2160, 600),   # Вправо-вниз
        (540, 1980, 540, 2220, 550),   # Вниз
        (540, 1980, 360, 2160, 600),   # Влево-вниз
        (540, 1980, 300, 1980, 550),   # Влево
        (540, 1980, 360, 1800, 600),   # Влево-вверх
        (540, 1980, 540, 1740, 550),   # Вверх
        # Перекрестный вакуум через центр:
        (540, 1980, 260, 2240, 750),   # Диагональ вниз-влево
        (540, 1980, 820, 1720, 750),   # Диагональ вверх-вправо
    ]

    start_time = time.time()
    max_duration = 360  # До 6 минут реального времени с учетом пауз
    step = 0
    battle_result = "timeout"

    logger.info("⚔️ БОЙ НАЧАЛСЯ! Orbit Boss Kite активен...")

    while time.time() - start_time < max_duration:
        m = orbit_moves[step % len(orbit_moves)]
        client.swipe(m[0], m[1], m[2], m[3], duration_ms=m[4])
        step += 1
        time.sleep(0.02)

        # Каждые 2 свайпа проверяем события экрана
        if step % 2 == 0:
            scr = client.get_screenshot()
            if scr is None:
                continue

            res_s, _ = ocr(scr)
            j_s = " ".join([r[1].lower() for r in res_s]) if res_s else ""

            # 1. Проверяем сундук с сокровищами 'Найти сокровища' / 'Собрать'
            if any(w in j_s for w in ["сокровища", "cokpob", "собрать", "co6patb"]):
                logger.success("🎁 ОБНАРУЖЕН СУНДУК СОКРОВИЩ! Жмём 'Собрать' (540, 1939)...")
                client.tap(540, 1939)
                time.sleep(1.0)
                client.tap(540, 1800)
                time.sleep(0.8)
                continue

            # 2. Проверяем выбор навыка при level-up
            if check_skill_screen(scr, ocr):
                logger.info("⚡ ВЫБОР НАВЫКА!")
                select_best_skill(client, ocr, scr)
                continue

            # 3. Проверяем ПОБЕДУ над боссом
            if any(w in j_s for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                logger.success(f"🏆 ПОБЕДА НАД ЛИДЕРОМ ЭТАПА 8! ({time.time() - start_time:.0f}с)")
                cv2.imwrite("logs/inspect/stage8_absolute_victory.png", scr)
                battle_result = "victory"
                break

            # 4. Проверяем поражение
            if any(w in j_s for w in ["битва проиграна", "проиграна"]):
                logger.warning("💀 Поражение. Нажимаем 'Повторить' (875, 1419)...")
                client.tap(875, 1419)
                time.sleep(3.0)
                start_time = time.time()
                step = 0
                continue

            # 5. Возврат на карту кампании (после победы)
            if "спаситель порядка" in j_s or "просмотр навыка" in j_s:
                logger.success("Бой завершён, возврат на карту Кампании.")
                battle_result = "victory"
                break

        if step % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  Orbit-кайт: шаг #{step}, прошло {elapsed:.0f}с...")
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

    # Сбор наград
    collect_victory_rewards(client, ocr)

    # Возврат в Город и сбор квеста на HUD
    return_to_city_and_claim(client, ocr)

    # Забираем финальную награду Главы 5 в Королевской Дороге!
    try_claim_chapter_reward(client, ocr)

    logger.success("=" * 65)
    logger.success(f"  ✅ ИТОГ ПРОХОЖДЕНИЯ ЭТАПА 8: {battle_result}")
    logger.success("=" * 65)
    return battle_result == "victory"


if __name__ == "__main__":
    master_stage8_win()
