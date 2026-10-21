"""
Служебный скрипт для снятия актуального скриншота с устройства
и проверки связи с игрой.
"""

import sys
from pathlib import Path
import cv2
import yaml
from loguru import logger

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient


def main():
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("Файл config.yaml не найден!")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    game_cfg = config["game"]

    client = AdbClient(
        adb_path=dev_cfg["adb_path"],
        ip=dev_cfg["ip"],
        port=dev_cfg["port"],
        timeout=dev_cfg.get("timeout_sec", 10),
    )

    if not client.connect():
        logger.error("Не удалось подключиться к телефону!")
        return

    is_game = client.is_game_foreground(game_cfg["package"])
    logger.info(f"Статус игры ({game_cfg['package']}): {'АКТИВНА на экране' if is_game else 'СВЕРНУТА / НЕ ЗАПУЩЕНА'}")

    screen = client.get_screenshot()
    if screen is not None:
        out_file = "current_screen.png"
        cv2.imwrite(out_file, screen)
        logger.success(f"Скриншот успешно сохранен в файл: {out_file} (Размер: {screen.shape[1]}x{screen.shape[0]})")
    else:
        logger.error("Не удалось получить скриншот экрана.")


if __name__ == "__main__":
    main()
