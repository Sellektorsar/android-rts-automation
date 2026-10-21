"""
Утилита для выполнения одного или нескольких действий (tap, swipe, wait)
и автоматического обновления current_screen.png.
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.adb_client import AdbClient


def main():
    parser = argparse.ArgumentParser(description="Step action runner")
    parser.add_argument("--tap", nargs=2, type=int, help="Tap coordinates: x y")
    parser.add_argument("--swipe", nargs=4, type=int, help="Swipe: x1 y1 x2 y2")
    parser.add_argument("--duration", type=int, default=350, help="Swipe duration in ms")
    parser.add_argument("--back", action="store_true", help="Press Android back key")
    parser.add_argument("--wait", type=float, default=2.0, help="Wait time in seconds after action")
    parser.add_argument("--repeat-tap", type=int, default=1, help="Repeat tap N times with small delay")
    parser.add_argument("--out", type=str, default="current_screen.png", help="Output screenshot path")
    args = parser.parse_args()

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
        sys.exit(1)

    if args.tap:
        x, y = args.tap
        for i in range(args.repeat_tap):
            client.tap(x, y)
            if i < args.repeat_tap - 1:
                time.sleep(0.5)

    if args.swipe:
        x1, y1, x2, y2 = args.swipe
        client.swipe(x1, y1, x2, y2, duration_ms=args.duration)

    if args.back:
        client.press_back()

    if args.wait > 0:
        time.sleep(args.wait)

    screen = client.get_screenshot()
    if screen is not None:
        cv2.imwrite(args.out, screen)
        logger.success(f"Экран обновлен: {args.out}")
    else:
        logger.error("Не удалось сделать скриншот!")


if __name__ == "__main__":
    main()
