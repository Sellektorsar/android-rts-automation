"""
Автономный скрипт найма войск в Казарме (Troop Training Flow).
Выполняет пошаговую последовательность:
1. Клик по Казарме (540, 1200) или (850, 1600).
2. Клик по иконке 'Найм' в радиальном меню (690, 1260).
3. Выбор типа войск:
   - 'infantry' (152, 329)
   - 'cavalry'  (424, 330)
   - 'archers'  (689, 329)
   - 'mages'    (960, 329)
4. Клик по кнопке 'Найм' (845, 2337).
5. Возврат в Город по кнопке Назад (65, 165).
"""

import sys
import time
from pathlib import Path
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient


def train_troops(unit_type: str = "archers"):
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
        logger.error("Не удалось подключиться к ADB!")
        return False

    tabs = {
        "infantry": (152, 329),
        "cavalry": (424, 330),
        "archers": (689, 329),
        "mages": (960, 329),
    }

    tab_coords = tabs.get(unit_type, tabs["archers"])

    logger.info("Шаг 1: Клик по Казарме (540, 1200)...")
    client.tap(540, 1200)
    time.sleep(1.2)

    logger.info("Шаг 2: Клик по 'Найм' в радиальном меню (690, 1260)...")
    client.tap(690, 1260)
    time.sleep(1.8)

    logger.info(f"Шаг 3: Выбор вкладки '{unit_type}' ({tab_coords})...")
    client.tap(tab_coords[0], tab_coords[1])
    time.sleep(1.0)

    logger.info("Шаг 4: Нажатие кнопки 'Найм' (845, 2337)...")
    client.tap(845, 2337)
    time.sleep(2.0)

    logger.info("Шаг 5: Возврат на экран города...")
    client.tap(65, 165)
    time.sleep(1.2)

    logger.success(f"Последовательность найма '{unit_type}' успешно выполнена!")
    return True


if __name__ == "__main__":
    unit = sys.argv[1] if len(sys.argv) > 1 else "archers"
    train_troops(unit)
