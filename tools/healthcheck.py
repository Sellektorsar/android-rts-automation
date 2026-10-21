"""
Проверка здоровья связки «ADB -> экран -> игра -> восприятие».

Примеры:
    python tools/healthcheck.py
    python tools/healthcheck.py --keep-awake
    python tools/healthcheck.py --json

Код возврата 0 — всё в порядке, 1 — есть проблема (удобно для скриптов).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.perception import HudParser, ScreenClassifier
from src.vision import Vision


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка состояния устройства и восприятия")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--keep-awake", action="store_true", help="Закрепить экран включённым")
    parser.add_argument("--json", action="store_true", help="Печать результата в JSON")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    dev = config["device"]
    game = config["game"]
    client = AdbClient(
        adb_path=dev["adb_path"],
        ip=dev["ip"],
        port=dev["port"],
        timeout=dev.get("timeout_sec", 10),
        frame_ttl=0.0,
    )

    report: dict[str, object] = {
        "address": client.device_addr,
        "connected": False,
        "screen_on": False,
        "game_foreground": False,
        "screen": None,
        "hud": None,
        "brightness": None,
        "problems": [],
    }
    problems: list[str] = report["problems"]  # type: ignore[assignment]

    if not client.connect(allow_discovery=True):
        problems.append("adb_connect_failed")
        report["ok"] = False
        _print(report, args.json)
        return 1
    report["connected"] = True
    # connect() мог переключиться на другой порт через mDNS/скан.
    report["address"] = client.device_addr

    report["screen_on"] = client.is_screen_on()
    if not report["screen_on"]:
        if not client.wake_screen():
            problems.append("screen_off_cannot_wake")
        report["screen_on"] = client.is_screen_on()

    if args.keep_awake:
        client.keep_screen_awake(True)

    report["game_foreground"] = client.is_game_foreground(game["package"])
    if not report["game_foreground"]:
        problems.append("game_not_foreground")

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        problems.append("screenshot_failed")
    else:
        report["brightness"] = round(float(frame.mean()), 1)
        if float(frame.mean()) < 2.0:
            problems.append("black_frame")

        vision = Vision(templates_dir="assets")
        perception_cfg = config.get("perception", {})
        classifier = ScreenClassifier(
            vision,
            ocr_scale=perception_cfg.get("ocr_scale", 0.5),
            allow_ocr=perception_cfg.get("allow_ocr", True),
            match_scale=perception_cfg.get("match_scale", 0.5),
            city_nav_min_brightness=perception_cfg.get("city_nav_min_brightness", 45.0),
        )
        classification = classifier.classify(frame)
        report["screen"] = classification.state
        report["screen_source"] = classification.source
        report["screen_confidence"] = round(classification.confidence, 3)
        report["screen_evidence"] = classification.evidence
        if not classification.is_known:
            problems.append("unknown_screen")

        hud = HudParser().parse(frame)
        report["hud"] = hud.as_dict()

    report["ok"] = not problems
    _print(report, args.json)
    return 0 if report["ok"] else 1


def _print(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    logger.info(f"Адрес: {report['address']}")
    logger.info(f"Подключение: {'OK' if report['connected'] else 'НЕТ'}")
    logger.info(f"Экран включён: {'да' if report['screen_on'] else 'нет'}")
    logger.info(f"Игра на переднем плане: {'да' if report['game_foreground'] else 'нет'}")
    logger.info(f"Яркость кадра: {report['brightness']}")
    logger.info(f"Экран: {report['screen']} ({report.get('screen_source')})")
    logger.info(f"HUD: {report['hud']}")
    if report["problems"]:
        logger.error(f"Проблемы: {', '.join(report['problems'])}")
    else:
        logger.success("Всё в порядке.")


if __name__ == "__main__":
    raise SystemExit(main())
