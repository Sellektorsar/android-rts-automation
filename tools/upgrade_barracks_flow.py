"""
Скрипт выполнения Квеста 2 Главы 4: Начать улучшение Казармы до 4 ур.
1. Проверка текущего экрана. Если радиальное меню открыто -> сразу жмем 'Улучшить'.
2. Если меню не открыто -> центрирование через квест (350, 1956), клик по зданию (540, 1200).
3. Динамический поиск кнопки 'Улучшить' через OCR.
4. В модальном окне: запуск улучшения ('Бесплатно' 304, 2339 или 'Улучшить' 794, 2338).
5. Мгновенное бесплатное ускорение стройки (< 5 мин).
6. Клейм награды на баннере HUD (350, 1956) или (250, 1955).
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


def upgrade_barracks_to_level_4():
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

    logger.info("1. Анализируем текущее состояние экрана...")
    screen = client.get_screenshot()
    if screen is None:
        logger.error("Не удалось получить скриншот!")
        return False

    res, _ = ocr(screen)
    upgrade_btn = None

    # Проверяем, есть ли уже кнопка 'Улучшить' на экране
    for box, text, score in res if res else []:
        txt = text.lower()
        cy = int((box[0][1] + box[2][1]) / 2)
        cx = int((box[0][0] + box[1][0]) / 2)
        if 1150 <= cy <= 1500 and any(w in txt for w in ["улучш", "yayy", "ymyw", "ymyy"]):
            upgrade_btn = (cx, cy)
            logger.info(f"Найдена кнопка 'Улучшить' в радиальном меню: ({cx}, {cy}) '{text}'")
            break

    if not upgrade_btn:
        logger.info("Радиальное меню не открыто. Центрируем камеру через баннер квеста (350, 1956)...")
        client.tap(350, 1956)
        time.sleep(2.0)

        logger.info("Клик по зданию Казармы в центре экрана (540, 1200)...")
        client.tap(540, 1200)
        time.sleep(1.5)

        screen = client.get_screenshot()
        res, _ = ocr(screen)
        for box, text, score in res if res else []:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)
            if 1150 <= cy <= 1500 and any(w in txt for w in ["улучш", "yayy", "ymyw", "ymyy"]):
                upgrade_btn = (cx, cy)
                logger.info(f"Найдена кнопка 'Улучшить' после клика по зданию: ({cx}, {cy}) '{text}'")
                break

    if upgrade_btn:
        logger.info(f"Клик по кнопке 'Улучшить' ({upgrade_btn[0]}, {upgrade_btn[1]})...")
        client.tap(upgrade_btn[0], upgrade_btn[1])
    else:
        logger.warning("Кнопка 'Улучшить' не найдена через OCR, используем известные координаты (539, 1338)...")
        client.tap(539, 1338)

    time.sleep(2.0)

    # Модальное окно улучшения
    modal_screen = client.get_screenshot()
    if modal_screen is not None:
        artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_upgrade_modal.png"
        cv2.imwrite(artifact_path, modal_screen)
        
        res_m, _ = ocr(modal_screen)
        has_free = False
        upgrade_action_btn = (794, 2338) # по умолчанию синяя кнопка 'Улучшить'
        
        for box, text, score in res_m if res_m else []:
            txt = text.lower()
            cy = int((box[0][1] + box[2][1]) / 2)
            cx = int((box[0][0] + box[1][0]) / 2)
            if cy > 2200:
                if any(w in txt for w in ["бесплат", "becna", "beon"]):
                    has_free = True
                    upgrade_action_btn = (cx, cy)
                    break
                elif any(w in txt for w in ["улучш", "yayyw"]):
                    upgrade_action_btn = (cx, cy)

        if has_free:
            logger.success(f"Доступно мгновенное улучшение! Клик по ({upgrade_action_btn[0]}, {upgrade_action_btn[1]})...")
            client.tap(upgrade_action_btn[0], upgrade_action_btn[1])
        else:
            logger.info(f"Запуск улучшения Казармы по кнопке ({upgrade_action_btn[0]}, {upgrade_action_btn[1]})...")
            client.tap(upgrade_action_btn[0], upgrade_action_btn[1])

    time.sleep(2.5)

    # Проверяем мгновенное бесплатное ускорение стройки на здании
    logger.info("Проверяем мгновенное бесплатное ускорение стройки...")
    for _ in range(3):
        client.tap(540, 1000)
        time.sleep(0.5)

    # Закрываем возможные окна / диалоги
    client.tap(540, 2000)
    time.sleep(0.8)
    client.tap(968, 160)
    time.sleep(0.8)

    # Клеймим награду за квест 2 на баннере HUD
    logger.info("Клеймим награду за квест на баннере HUD...")
    for _ in range(2):
        client.tap(250, 1955)
        time.sleep(1.5)
        client.tap(540, 1800)
        time.sleep(0.8)

    final_screen = client.get_screenshot()
    if final_screen is not None:
        artifact_path2 = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\after_barracks_quest.png"
        cv2.imwrite(artifact_path2, final_screen)
        res_f, _ = ocr(final_screen)
        logger.info("=== HUD ПОСЛЕ ВЫПОЛНЕНИЯ КВЕСТА КАЗАРМЫ ===")
        for r in (res_f or []):
            cy = int(sum(p[1] for p in r[0])/4)
            if 1800 < cy < 2100:
                logger.info(f"[{float(r[2]):.2f}]: {r[1]}")

    logger.success("Квест улучшения Казармы до 4 ур. завершен!")
    return True


if __name__ == "__main__":
    upgrade_barracks_to_level_4()

