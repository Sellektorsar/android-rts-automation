"""
Безопасный зонд меню «Задание» (Quests).

Что делает:
1. Проверяет, что игра на переднем плане и мы в городе.
2. Находит нижнюю кнопку «Задание» по шаблону `btn_nav_quests`.
3. Открывает меню, классифицирует `quests_menu`, читает OCR.
4. Фиксирует найденные кнопки «Перейти/Вперёд», но НЕ нажимает их.
5. Возвращается в город по `Back`.

Скрипт не запускает бой, не покупает и не улучшает здания. Его задача —
понять, есть ли в заданиях безопасный переход к улучшаемому зданию.

Использование:
    python tools/probe_quests.py
    python tools/probe_quests.py --json
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


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    return client


def _return_to_city(
    client: AdbClient, classifier: ScreenClassifier, max_attempts: int = 4
) -> tuple[bool, List[str]]:
    states: List[str] = []
    for _ in range(max_attempts):
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break
        classification = classifier.classify(frame)
        states.append(classification.state)
        if classification.state == "city_view":
            return True, states
        client.press_back()
        time.sleep(1.8)
    return False, states


def main() -> int:
    parser = argparse.ArgumentParser(description="Безопасный зонд меню заданий")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-dir", default="logs/inspect")
    parser.add_argument("--no-return", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
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

    height, width = frame.shape[:2]
    nav = vision.find_template(
        frame,
        "btn_nav_quests",
        threshold=0.82,
        region=(0, int(height * 0.84), width, int(height * 0.16)),
    )
    result: Dict[str, Any] = {"start_state": start.state, "nav_match": None}
    if nav is None:
        print("Кнопка «Задание» не найдена — не нажимаем вслепую.")
        return 5

    nav_x, nav_y, nav_conf = nav
    result["nav_match"] = {
        "x": int(nav_x),
        "y": int(nav_y),
        "confidence": round(float(nav_conf), 3),
    }
    client.tap(nav_x, nav_y)
    time.sleep(2.5)

    menu_frame = client.get_screenshot(fresh=True)
    if menu_frame is None:
        print("Нет кадра после открытия заданий.")
        return 6
    menu_state = classifier.classify(menu_frame)
    result["menu_screen"] = {
        "state": menu_state.state,
        "confidence": round(menu_state.confidence, 3),
        "source": menu_state.source,
        "evidence": menu_state.evidence,
    }
    menu_path = save_dir / "quests_probe.png"
    cv2.imwrite(str(menu_path), menu_frame)
    result["saved_to"] = str(menu_path)

    ocr = read_text(menu_frame, scale=0.5, min_confidence=0.35)
    result["ocr_text"] = ocr.text[:2500]
    buttons: Dict[str, Any] = {}
    for template in ("btn_quest_go_forward", "btn_quest_forward", "btn_vpered_active"):
        match = vision.find_template(menu_frame, template, threshold=0.75)
        if match:
            buttons[template] = {
                "x": int(match[0]),
                "y": int(match[1]),
                "confidence": round(float(match[2]), 3),
            }
    result["buttons"] = buttons

    if not args.no_return:
        ok, states = _return_to_city(client, classifier)
        result["returned_to_city"] = ok
        result["return_states"] = states

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Экран: {result['menu_screen']}")
        print(f"Сохранено: {result['saved_to']}")
        print(f"Кнопки: {result['buttons']}")
        print(f"Возврат: {result.get('returned_to_city')} {result.get('return_states')}")
        print("OCR:")
        print(result["ocr_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
