"""
Оптимизированный боевой скрипт для Этапа 8 Кампании (Savior of Order Stage 8).
Включает:
1. Запуск боя из открытого брифинга (или карты) по кнопке 'Испытание'.
2. Широкий вакуумный сбор кристаллов по всей арене (Wide Arena Vacuum Cycle).
3. Быстрая проверка выбора навыка (каждые 2 свайпа, без 12-секундных задержек).
4. Приоритет навыков: Магнит (сбор 100% кристаллов) > Пушка > Щит > Меч > Молния.
5. Фиксация победы, сбор наград и возврат в Город.
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


def run_stage8():
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
        logger.error("❌ Не удалось подключиться к ADB!")
        return False

    ocr = RapidOCR()
    logger.info("=" * 55)
    logger.info("  🚀 СТАРТ БОЯ ЭТАПА 8 (BOSS ATTACK) — VACUUM COMBAT")
    logger.info("=" * 55)

    # 1. Проверяем наличие зелёной кнопки 'Испытание' на текущем экране
    screen = client.get_screenshot()
    btn = find_green_button(screen) if screen is not None else None
    if btn:
        logger.success(f"🟢 Нажимаем 'Испытание' в брифинге: {btn}")
        client.tap(btn[0], btn[1])
        time.sleep(4.0)
    else:
        # Если брифинг закрыт, тапаем узел этапа 8 (853, 1152)
        logger.info("Тапаем узел этапа 8 (853, 1152)...")
        client.tap(853, 1152)
        time.sleep(2.0)
        screen = client.get_screenshot()
        btn = find_green_button(screen) if screen is not None else None
        if btn:
            logger.success(f"🟢 Нажимаем 'Испытание' в брифинге: {btn}")
            client.tap(btn[0], btn[1])
            time.sleep(4.0)
        else:
            client.tap(544, 1840)
            time.sleep(4.0)

    # 2. Боевой цикл с глубоким сбором кристаллов по всей арене
    # Центр виртуального джойстика: (540, 1980)
    # Широкие свайпы с удержанием 900-1100мс перемещают героя к краям и обратно через центр
    wide_arena_moves = [
        # Восьмерка с широким охватом:
        (540, 1980, 800, 1700, 950),   # Вверх-вправо на край
        (540, 1980, 850, 1980, 850),   # Вправо
        (540, 1980, 250, 2250, 1100),  # Сквозной диагональный проход через центр вниз-влево
        (540, 1980, 200, 1980, 850),   # Влево на край
        (540, 1980, 250, 1700, 950),   # Вверх-влево на край
        (540, 1980, 800, 2250, 1100),  # Сквозной диагональный проход через центр вниз-вправо
        (540, 1980, 540, 1600, 950),   # Вверх на север арены
        (540, 1980, 540, 2350, 1000),  # Вниз на юг арены через центр
    ]

    start_time = time.time()
    max_duration = 240  # Босс может потребовать до 4 минут
    cycle = 0
    battle_result = "timeout"

    logger.info("⚔️ БОЙ НАЧАЛСЯ! Активирован Wide Arena Vacuum...")

    while time.time() - start_time < max_duration:
        # Выполняем движения парами и сразу проверяем выбор навыка
        for i in range(0, len(wide_arena_moves), 2):
            m1 = wide_arena_moves[i]
            client.swipe(m1[0], m1[1], m1[2], m1[3], duration_ms=m1[4])
            time.sleep(0.02)

            if i + 1 < len(wide_arena_moves):
                m2 = wide_arena_moves[i + 1]
                client.swipe(m2[0], m2[1], m2[2], m2[3], duration_ms=m2[4])
                time.sleep(0.02)

            # Быстрая проверка экрана каждые 2 свайпа:
            scr = client.get_screenshot()
            if scr is None:
                continue

            # 1. Проверяем выбор навыка (пауза)
            if select_best_skill(client, ocr, scr):
                logger.info("⚡ Навык выбран! Продолжаем кайт...")
                continue

            # 2. Проверяем победу или поражение
            res, _ = ocr(scr)
            joined = " ".join([r[1].lower() for r in res]) if res else ""

            if any(w in joined for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы продолжить"]):
                logger.success(f"🏆 ПОБЕДА НАД БОССОМ ЭТАПА 8! ({time.time() - start_time:.0f}с)")
                battle_result = "victory"
                break

            if any(w in joined for w in ["проиграна", "битва проиграна"]):
                logger.warning("💀 Поражение! Нажимаем 'Повторить' (875, 1418)...")
                client.tap(875, 1418)
                time.sleep(3.0)
                start_time = time.time()
                break

            if "спаситель порядка" in joined or "просмотр навыка" in joined:
                logger.success("Бой завершён, возврат на карту Кампании.")
                battle_result = "victory"
                break

        if battle_result in ["victory", "defeat"]:
            if battle_result == "victory":
                break
            else:
                battle_result = "timeout"
                continue

        cycle += 1
        elapsed = time.time() - start_time
        logger.info(f"  Кристалл-вакуум: цикл #{cycle}, прошло {elapsed:.0f}с")

        # Сворачиваем шторку уведомлений
        if cycle % 2 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

    # Сбор наград победы
    collect_victory_rewards(client, ocr)

    # Возврат в Город и клейм с HUD
    return_to_city_and_claim(client, ocr)

    # Проверка финальной награды главы в Королевской Дороге
    try_claim_chapter_reward(client, ocr)

    logger.success("=" * 55)
    logger.success(f"  ✅ ЗАВЕРШЕНИЕ ЭТАПА 8: {battle_result}")
    logger.success("=" * 55)
    return battle_result == "victory"


if __name__ == "__main__":
    run_stage8()
