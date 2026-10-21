"""
Инструмент калибровки восприятия.

Делает один свежий скриншот, классифицирует экран, читает HUD и
сохраняет размеченный кадр. Ничего не нажимает.

Использование:
    python tools/inspect_state.py
    python tools/inspect_state.py --json
    python tools/inspect_state.py --no-ocr --no-hud
    python tools/inspect_state.py --port 39519 --save logs/inspect/city.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.perception import HudParser, ScreenClassifier
from src.vision import Vision


def annotate(frame: np.ndarray, state: str, confidence: float, source: str, hud: dict) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    cv2.putText(
        canvas,
        f"{state}  conf={confidence:.2f}  src={source}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    hud_text = "  ".join(f"{key}={value}" for key, value in hud.items() if value is not None)
    cv2.putText(
        canvas,
        hud_text[:110],
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Снимок и классификация экрана War & Order")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-ocr", action="store_true", help="Только якорные шаблоны")
    parser.add_argument("--no-hud", action="store_true", help="Не читать HUD")
    parser.add_argument("--json", action="store_true", help="Вывести JSON")
    parser.add_argument("--save", default="logs/inspect/last.png")
    parser.add_argument(
        "--force", action="store_true",
        help="Разрешить снимок, даже если игра не на переднем плане (не сохраняет кадр)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = config["device"]
    client = AdbClient(
        adb_path=device["adb_path"],
        ip=args.ip or device["ip"],
        port=args.port or device["port"],
        timeout=device.get("timeout_sec", 10),
    )

    if not client.connect(allow_discovery=True):
        logger.error("Не удалось подключиться к устройству")
        return 1

    client.wake_screen()
    client.collapse_statusbar()

    # Приватность: если на экране другое приложение, не читаем и не сохраняем
    # кадр. Калибровочные снимки не должны содержать чужие данные.
    game_foreground = client.is_game_foreground(config["game"]["package"])
    if not game_foreground:
        if not args.force:
            logger.error(
                "Игра не на переднем плане. Снимок не делается и не сохраняется. "
                "Откройте игру или используйте --force для диагностики (без сохранения)."
            )
            return 2
        logger.warning("Игра не на переднем плане (--force): кадр не будет сохранён.")

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        logger.error("Не удалось получить скриншот")
        return 1
    if float(frame.mean()) < 2.0:
        logger.error("Кадр чёрный — экран выключен или заблокирован.")
        return 3

    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision,
        ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5),
        allow_ocr=not args.no_ocr,
        match_scale=config.get("perception", {}).get("match_scale", 0.5),
        city_nav_min_brightness=config.get("perception", {}).get("city_nav_min_brightness", 45.0),
    )

    classification = classifier.classify(frame)
    hud_state = None if args.no_hud else HudParser().parse(frame)
    hud = hud_state.as_dict() if hud_state else {}

    result = {
        "screen": classification.state,
        "confidence": round(classification.confidence, 3),
        "source": classification.source,
        "evidence": classification.evidence,
        "ocr_elapsed": round(classification.ocr_elapsed, 3),
        "ocr_text": classification.ocr_text,
        "near_misses": classification.near_misses,
        "hud": hud,
        "hud_raw": hud_state.raw if hud_state else {},
        "frame_size": [frame.shape[1], frame.shape[0]],
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.info(f"Экран: {result['screen']} (источник={result['source']}, conf={result['confidence']})")
        if result["evidence"]:
            logger.info(f"Доказательства: {result['evidence']}")
        if result["ocr_text"]:
            logger.info(f"OCR: {result['ocr_text'][:200]}")
        if result["near_misses"]:
            logger.info(f"Ближайшие совпадения: {result['near_misses']}")
        logger.info(f"HUD: {hud} (raw={result['hud_raw']})")

    save_path = Path(args.save)
    if not game_foreground:
        logger.warning("Кадр не сохранён: игра не на переднем плане.")
        return 0
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotate(frame, result["screen"], result["confidence"], result["source"], hud))
    logger.info(f"Размеченный кадр сохранён: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
