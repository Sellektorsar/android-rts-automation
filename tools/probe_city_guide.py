"""Безопасный зонд городских указателей.

Нажимает только найденный шаблон (`btn_quest_hammer` или
`btn_builder_idle`), классифицирует открывшийся экран и возвращается в
город. Координаты не задаются вручную: если шаблон не найден, скрипт
ничего не нажимает.

Использование:
    python tools/probe_city_guide.py --target quest_hammer
    python tools/probe_city_guide.py --target builder --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.ocr import read_text
from src.perception import ScreenClassifier
from src.vision import Vision

TARGETS = {
    "quest_hammer": "btn_quest_hammer",
    "builder": "btn_builder_idle",
}


def _connect(config: dict, args) -> AdbClient:
    device = config["device"]
    client = AdbClient(
        adb_path=device["adb_path"],
        ip=args.ip or device["ip"],
        port=args.port or device["port"],
        timeout=device.get("timeout_sec", 10),
    )
    if not client.connect(allow_discovery=True):
        raise RuntimeError("Не удалось подключиться к ADB")
    client.wake_screen()
    client.collapse_statusbar()
    client.keep_screen_awake()
    return client


def _return_to_city(
    client: AdbClient, classifier: ScreenClassifier, max_attempts: int = 4
) -> tuple[bool, List[str]]:
    states: List[str] = []
    for _ in range(max_attempts):
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break
        result = classifier.classify(frame)
        states.append(result.state)
        if result.state == "city_view":
            return True, states
        client.press_back()
        time.sleep(1.8)
    return False, states


def main() -> int:
    parser = argparse.ArgumentParser(description="Зонд городского указателя")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", choices=sorted(TARGETS), default="quest_hammer")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-dir", default="logs/inspect")
    parser.add_argument("--no-return", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not client.is_game_foreground(config["game"]["package"]):
        print("Игра не на переднем плане.")
        return 2

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        print("Нет кадра.")
        return 3
    start = classifier.classify(frame)
    if start.state != "city_view":
        print(f"Ожидался city_view, получен {start.state}.")
        return 4

    template = TARGETS[args.target]
    match = vision.find_template(frame, template, threshold=0.80)
    result: Dict[str, Any] = {
        "target": args.target,
        "template": template,
        "start_state": start.state,
        "match": None,
        "screen_after": None,
    }
    if match is None:
        print(f"Шаблон {template} не найден — не нажимаем вслепую.")
        return 5

    result["match"] = {
        "x": int(match[0]),
        "y": int(match[1]),
        "confidence": round(float(match[2]), 3),
    }
    client.tap(match[0], match[1])
    time.sleep(2.5)

    after = client.get_screenshot(fresh=True)
    if after is not None:
        classification = classifier.classify(after)
        path = save_dir / f"guide_{args.target}.png"
        cv2.imwrite(str(path), after)
        ocr = read_text(after, scale=0.5, min_confidence=0.35)
        result["screen_after"] = {
            "state": classification.state,
            "confidence": round(classification.confidence, 3),
            "source": classification.source,
            "evidence": classification.evidence,
            "saved_to": str(path),
            "ocr_text": ocr.text[:2000],
        }

    if not args.no_return:
        ok, states = _return_to_city(client, classifier)
        result["returned_to_city"] = ok
        result["return_states"] = states

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
