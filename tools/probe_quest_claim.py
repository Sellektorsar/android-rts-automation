"""Зонд одного нажатия «Забрать награды» в ежедневных заданиях.

Открывает `quests_menu`, нажимает **только** зелёную кнопку
`btn_claim_green`, сохраняет экран после нажатия и возвращается в город.
Никакие кнопки покупки/премиум-валюты не нажимаются.

Использование:
    python tools/probe_quest_claim.py --json
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


def _connect(config: dict) -> AdbClient:
    device = config["device"]
    client = AdbClient(
        adb_path=device["adb_path"],
        ip=device["ip"],
        port=device["port"],
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
        time.sleep(1.6)
    return False, states


def main() -> int:
    parser = argparse.ArgumentParser(description="Зонд награды ежедневного задания")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-dir", default="logs/inspect")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    client = _connect(config)
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
        return 3
    if classifier.classify(frame).state != "city_view":
        print("Ожидался city_view.")
        return 4

    nav = vision.find_template(
        frame,
        "btn_nav_quests",
        threshold=0.82,
        region=(0, int(frame.shape[0] * 0.84), frame.shape[1], int(frame.shape[0] * 0.16)),
    )
    if nav is None:
        print("Кнопка «Задание» не найдена.")
        return 5
    client.tap(nav[0], nav[1])
    time.sleep(2.2)
    menu = client.get_screenshot(fresh=True)
    if menu is None:
        return 6
    menu_result = classifier.classify(menu)
    if menu_result.state != "quests_menu":
        print(f"Ожидался quests_menu, получен {menu_result.state}.")
        _return_to_city(client, classifier)
        return 7

    height, width = menu.shape[:2]
    matches = vision.find_all(
        menu,
        "btn_claim_green",
        threshold=0.90,
        min_distance=70,
        region=(int(width * 0.72), int(height * 0.20), int(width * 0.24), int(height * 0.50)),
        scale=0.5,
    )
    result: Dict[str, Any] = {
        "menu_state": menu_result.state,
        "claim_buttons": len(matches),
        "tapped": None,
        "after": None,
    }
    if not matches:
        print("Зелёных кнопок нет.")
        _return_to_city(client, classifier)
        return 8

    matches.sort(key=lambda item: item[1])
    x, y, confidence = matches[0]
    result["tapped"] = {"x": int(x), "y": int(y), "confidence": round(float(confidence), 3)}
    client.tap(x, y)
    time.sleep(2.0)

    after = client.get_screenshot(fresh=True)
    if after is not None:
        after_result = classifier.classify(after)
        after_path = save_dir / "quest_claim_after.png"
        cv2.imwrite(str(after_path), after)
        ocr = read_text(after, scale=0.5, min_confidence=0.35)
        result["after"] = {
            "state": after_result.state,
            "confidence": round(after_result.confidence, 3),
            "source": after_result.source,
            "evidence": after_result.evidence,
            "saved_to": str(after_path),
            "ocr_text": ocr.text[:2000],
        }

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
