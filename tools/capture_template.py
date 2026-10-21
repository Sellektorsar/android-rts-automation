"""
Утилита для захвата и нарезки шаблонов UI (кнопок, иконок) с экрана устройства.

Пример использования:
    python tools/capture_template.py --name btn_close_x --box 1000 50 60 60
    python tools/capture_template.py --screenshot-only
"""

import argparse
import sys
from pathlib import Path
import cv2
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient


def main():
    parser = argparse.ArgumentParser(description="Захват шаблонов кнопок и иконок")
    parser.add_argument("--screenshot-only", action="store_true", help="Только сохранить снимок экрана")
    parser.add_argument("--name", type=str, help="Имя шаблона (без .png)")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="Координаты прямоугольника: x y w h")
    parser.add_argument("--out-dir", type=str, default="assets", help="Папка для сохранения")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dev_cfg = config["device"]
    client = AdbClient(
        adb_path=dev_cfg["adb_path"],
        ip=dev_cfg["ip"],
        port=dev_cfg["port"],
        timeout=10,
    )

    if not client.connect():
        logger.error("Не удалось подключиться к телефону!")
        return

    screen = client.get_screenshot()
    if screen is None:
        logger.error("Ошибка получения скриншота")
        return

    # Сохраняем полный кадр для удобства анализа
    cv2.imwrite("screen_full.png", screen)
    logger.info(f"Полный снимок сохранен в 'screen_full.png' ({screen.shape[1]}x{screen.shape[0]})")

    if args.screenshot_only or not args.name or not args.box:
        logger.info("Для вырезки шаблона используйте: --name <имя> --box <x> <y> <w> <h>")
        return

    x, y, w, h = args.box
    crop = screen[y : y + h, x : x + w]

    out_path = Path(args.out_dir) / f"{args.name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    logger.success(f"Шаблон '{args.name}' успешно сохранен: {out_path} ({w}x{h} px)")


if __name__ == "__main__":
    main()
