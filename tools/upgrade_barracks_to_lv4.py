"""
Автономный скрипт прокачки Казармы с 1 по 4 уровень.
Поскольку время улучшения на этих уровнях <= 5 минут,
каждое улучшение завершается мгновенно и бесплатно.
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


def run_upgrade_to_level_4():
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
    logger.info("=== ЗАПУСК ЦИКЛА УЛУЧШЕНИЯ КАЗАРМЫ ДО 4 УРОВНЯ ===")

    for step in range(1, 5):
        logger.info(f"--- Шаг цикла #{step} ---")
        screen = client.get_screenshot()
        if screen is None:
            continue

        res, _ = ocr(screen)
        joined = " ".join([r[1].lower() for r in res]) if res else ""

        # 1. Если уже открыто модальное окно улучшения
        if "требуется" in joined or "tpebyetca" in joined or "бесплатно" in joined or "becna" in joined or "улучшить" in joined:
            logger.info("Обнаружено открытое окно улучшения здания!")
            # Ищем кнопку 'Бесплатно' внизу
            has_free = False
            for box, text, score in res if res else []:
                txt = text.lower()
                cy = int((box[0][1] + box[2][1]) / 2)
                cx = int((box[0][0] + box[1][0]) / 2)
                if cy > 2250 and ("бесплат" in txt or "becna" in txt):
                    logger.success(f"Нажимаем 'Бесплатно' в модалке ({cx}, {cy})...")
                    client.tap(cx, cy)
                    has_free = True
                    break
            
            if not has_free:
                logger.info("Нажимаем 'Улучшить' в модалке (794, 2338)...")
                client.tap(794, 2338)

            time.sleep(2.0)

            # Бесплатное ускорение на здании если появилось
            for _ in range(3):
                client.tap(540, 1000)
                time.sleep(0.5)

        # 2. Если в городе
        # Закрываем возможные всплывашки
        client.tap(540, 2000)
        time.sleep(0.5)

        # Центрируем на казарме
        logger.info("Клик по казарме (540, 1200)...")
        client.tap(540, 1200)
        time.sleep(1.5)

        # Если над казармой была награда/солдаты, кликаем еще раз
        scr_check = client.get_screenshot()
        res_c, _ = ocr(scr_check)
        joined_c = " ".join([r[1].lower() for r in res_c]) if res_c else ""
        if "награду" in joined_c or "получили" in joined_c:
            logger.info("Собрали солдат/награду, открываем казарму снова...")
            client.tap(540, 1200)
            time.sleep(1.5)

        # Клик по иконке 'Улучшить' (530, 1282)
        logger.info("Клик по кнопке 'Улучшить' в радиальном меню (530, 1282)...")
        client.tap(530, 1282)
        time.sleep(2.0)

        # Проверяем модалку
        modal_scr = client.get_screenshot()
        res_m, _ = ocr(modal_scr)
        m_joined = " ".join([r[1].lower() for r in res_m]) if res_m else ""

        # Если достигли максимального уровня для текущего замка или 4 ур.
        if "lv.4" in m_joined and "lv.5" in m_joined:
            logger.success("Казарма достигла 4 уровня!")
            client.press_back()
            break

        # Нажимаем улучшить/бесплатно
        has_f = False
        for box, text, score in res_m if res_m else []:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)
            if cy > 2250 and ("бесплат" in txt or "becna" in txt):
                logger.success(f"Нажимаем 'Бесплатно' ({cx}, {cy})...")
                client.tap(cx, cy)
                has_f = True
                break

        if not has_f and ("улучшить" in m_joined or "yayy" in m_joined):
            logger.info("Нажимаем 'Улучшить' (794, 2338)...")
            client.tap(794, 2338)

        time.sleep(2.0)
        # Проверяем ускорение на здании
        for _ in range(3):
            client.tap(540, 1000)
            time.sleep(0.5)

    # Завершение: проверяем клейм квеста
    logger.info("Проверяем клейм награды на баннере HUD...")
    for _ in range(3):
        client.tap(250, 1955)
        time.sleep(1.2)
        client.tap(540, 1800)
        time.sleep(0.8)

    final_scr = client.get_screenshot()
    if final_scr is not None:
        cv2.imwrite("screen_after_barracks_all_upgrades.png", final_scr)
        artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\after_barracks_all_upgrades.png"
        cv2.imwrite(artifact_path, final_scr)
        res_f, _ = ocr(final_scr)
        logger.info("=== HUD ПОСЛЕ АПГРЕЙДОВ КАЗАРМЫ ===")
        for r in (res_f or []):
            cy = int(sum(p[1] for p in r[0])/4)
            if 1800 < cy < 2100:
                logger.info(f"[{float(r[2]):.2f}]: {r[1]}")

    logger.success("Цикл прокачки казармы успешно завершен!")
    return True


if __name__ == "__main__":
    run_upgrade_to_level_4()
