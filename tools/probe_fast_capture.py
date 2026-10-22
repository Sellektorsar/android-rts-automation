"""
Проверка быстрого потока кадров (`screenrecord` + `ffmpeg`).

Скрипт только читает кадры и сохраняет один снимок. Он не нажимает кнопки
и не запускает бой.

Использование:
    python tools/probe_fast_capture.py --frames 30
    python tools/probe_fast_capture.py --frames 60 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.fast_capture import FastCapture
from src.perception import ScreenClassifier
from src.vision import Vision


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка быстрого потока кадров")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", default="logs/inspect/fast_capture.png")
    parser.add_argument("--no-game-check", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = config["device"]
    client = AdbClient(
        adb_path=device["adb_path"],
        ip=device["ip"],
        port=device["port"],
        timeout=device.get("timeout_sec", 10),
    )
    if not client.connect(allow_discovery=True):
        print("Не удалось подключиться к ADB")
        return 2
    client.wake_screen()
    client.collapse_statusbar()
    if not args.no_game_check and not client.is_game_foreground(
        config["game"]["package"]
    ):
        print("Игра не на переднем плане — поток не запускаем.")
        return 3

    # После mDNS-поиска AdbClient мог переключиться на другой порт.
    serial = client.device_addr
    capture = FastCapture(device["adb_path"], serial)
    if not capture.start():
        print("FastCapture недоступен.")
        return 4

    started = time.time()
    frames = []
    try:
        while len(frames) < max(1, args.frames):
            frame = capture.read(timeout_sec=3.0)
            if frame is None:
                break
            frames.append(frame)
    finally:
        capture.stop()

    elapsed = time.time() - started
    result = {
        "frames": len(frames),
        "elapsed_sec": round(elapsed, 3),
        "fps": round(len(frames) / elapsed, 2) if elapsed > 0 else 0.0,
        "shape": list(frames[-1].shape) if frames else None,
        "saved_to": None,
        "screen": None,
        "confidence": None,
    }
    if frames:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save, frames[-1])
        result["saved_to"] = args.save

        # Классификатор рассчитан на 1080x2436; для быстрого кадра
        # масштабируем обратно, чтобы проверить, что картинка валидна.
        height, width = frames[-1].shape[:2]
        if (width, height) != (1080, 2436):
            scaled = cv2.resize(frames[-1], (1080, 2436))
        else:
            scaled = frames[-1]
        vision = Vision(templates_dir="assets")
        classifier = ScreenClassifier(vision, ocr_scale=0.5)
        classification = classifier.classify(scaled)
        result["screen"] = classification.state
        result["confidence"] = round(classification.confidence, 3)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"Кадров: {result['frames']}, время: {result['elapsed_sec']}с, "
            f"FPS: {result['fps']}, экран: {result['screen']} "
            f"({result['confidence']})"
        )
        if result["saved_to"]:
            print(f"Снимок: {result['saved_to']}")
    return 0 if frames else 5


if __name__ == "__main__":
    raise SystemExit(main())
