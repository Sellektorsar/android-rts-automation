"""
Автономный скрипт для прохождения боя Этапа 6 (Победить Лидера) и финализации Главы 3.
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


def main():
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

    logger.info("=== ЗАПУСК БОЕВОЙ ФАЗЫ ЭТАПА 6 (ЛИДЕР) ===")

    # Проверяем текущий экран
    screen = client.get_screenshot()
    if screen is not None:
        ocr_res, _ = ocr(screen)
        joined = " ".join([r[1].lower() for r in ocr_res]) if ocr_res else ""
        logger.info(f"Начальный экран: {joined[:100]}")

        # Если случайно на карте Кампании - открываем Этап 6 (822, 1284) и жмем 'Испытание'
        if "спаситель" in joined or "просмотр" in joined or "доблести" in joined:
            logger.info("На карте Кампании. Открываем Этап 6 (822, 1284)...")
            client.tap(822, 1284)
            time.sleep(2.0)
            client.tap(540, 1885)
            time.sleep(3.0)
        elif "этап 6" in joined or "нападение босса" in joined:
            logger.info("В окне брифинга Этапа 6. Жмем 'Испытание' (540, 1885)...")
            client.tap(540, 1885)
            time.sleep(3.0)

    # Круговой кайт вокруг джойстика (540, 1980)
    kite_moves = [
        (540, 1980, 540, 1680, 950),  # Вверх
        (540, 1980, 840, 1980, 950),  # Вправо
        (540, 1980, 540, 2280, 950),  # Вниз
        (540, 1980, 240, 1980, 950),  # Влево
    ]

    start_time = time.time()
    max_duration = 150  # секунд максимум
    cycle = 0
    in_battle = True

    while in_battle and (time.time() - start_time < max_duration):
        # 1. Выполняем круг кайта
        for x1, y1, x2, y2, dur in kite_moves:
            client.swipe(x1, y1, x2, y2, duration_ms=dur)
            time.sleep(0.04)

        cycle += 1
        elapsed = time.time() - start_time
        logger.info(f"Бой Этап 6: цикл #{cycle}, прошло {elapsed:.1f}с...")

        # 2. Снятие паузы / выбор карты навыков
        client.tap(537, 1004)
        time.sleep(0.2)
        client.tap(541, 1744)
        time.sleep(0.2)
        client.tap(540, 1850)
        time.sleep(0.1)

        # 3. Сворачиваем шторку
        if cycle % 3 == 0:
            client._run_cmd(["shell", "cmd", "statusbar", "collapse"])

        # 4. Проверяем состояние каждые 2 цикла
        if cycle % 2 == 0:
            scr = client.get_screenshot()
            if scr is not None:
                res, _ = ocr(scr)
                j = " ".join([r[1].lower() for r in res]) if res else ""

                if any(w in j for w in ["победа", "побед", "victory", "100%", "завершено", "нажмите, чтобы"]):
                    logger.success(f"ОБНАРУЖЕН ЭКРАН ПОБЕДЫ! Текст: {j[:80]}")
                    in_battle = False
                    break

                if any(w in j for w in ["проиграна", "битва проиграна"]):
                    logger.warning("Битва проиграна, жмем 'Повторить' (875, 1418)...")
                    client.tap(875, 1418)
                    time.sleep(3.0)
                    start_time = time.time()
                    continue

                if "спаситель порядка" in j or "просмотр навыка" in j:
                    logger.success("Мы вернулись на карту Кампании!")
                    in_battle = False
                    break

    logger.info("Боевая фаза окончена. Очищаем экран наград победы...")
    for _ in range(6):
        client.tap(540, 1850)
        time.sleep(1.2)
        client.tap(540, 2000)
        time.sleep(1.0)

    # Возврат в Город
    logger.info("Возвращаемся в Город...")
    for attempt in range(5):
        scr = client.get_screenshot()
        if scr is None:
            continue
        res, _ = ocr(scr)
        j = " ".join([r[1].lower() for r in res]) if res else ""

        if any(w in j for w in ["замок", "событие", "рюкзак", "почта", "альянс"]):
            logger.success("Мы в Городе (City View)!")
            break

        if "спаситель" in j or "просмотр" in j or "доблести" in j:
            logger.info("Нажимаем стрелку 'Назад' на карте Кампании (75, 120)...")
            client.tap(75, 120)
            time.sleep(2.5)
        else:
            client.tap(540, 1850)
            time.sleep(1.5)

    # Сбор наград квеста 5 и закрытие Главы 3
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

    final_screen = client.get_screenshot()
    if final_screen is not None:
        cv2.imwrite("screen_chapter3_completed.png", final_screen)
        artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter3_completed.png"
        cv2.imwrite(artifact_path, final_screen)
        res_f, _ = ocr(final_screen)
        jf = " ".join([r[1].lower() for r in res_f]) if res_f else ""
        logger.info(f"Финальный экран: {jf[:120]}")

    logger.success("=== ГЛАВА 3 ПОЛНОСТЬЮ ЗАВЕРШЕНА! ===")
    return True


if __name__ == "__main__":
    main()
