"""
Один автономный такт BuildingUpgradeTask по конфигу.

Задача сама выбирает спот, ищет шаблон, открывает радиальное меню и
панель требований. Нажимается только активная синяя кнопка «Улучшить»;
зелёная кнопка гемов и серые неактивные кнопки игнорируются.

После такта, если панель осталась открытой, зонд закрывает её Back-ом
(диалог выхода отменяется повторным Back, «Да» не нажимается).

Использование:
    python tools/run_building_upgrade_once.py --json
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
from src.perception import ScreenClassifier
from src.tasks.building_upgrade import BuildingUpgradeTask
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
        raise SystemExit("Нет подключения к ADB")
    client.wake_screen()
    client.collapse_statusbar()
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description="Один такт BuildingUpgradeTask")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-dir", default="logs/inspect")
    args = parser.parse_args()

    config = _load_config(args.config)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        print("Нет кадра")
        return 2
    state = classifier.classify(frame).state
    result = {"start_state": state}
    if state != "city_view":
        result["error"] = "expected_city_view"
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
        return 3

    upgrade_cfg = (config.get("tasks", {}) or {}).get("building_upgrade", {}) or {}
    task = BuildingUpgradeTask(
        client,
        vision,
        interval_sec=upgrade_cfg.get("interval_sec", 60),
        spots=upgrade_cfg.get("spots"),
        sweep=upgrade_cfg.get("sweep"),
        world=None,
    )
    result["spots"] = [
        {"name": name, "template": template, "tap_dy": tap_dy}
        for name, _point, template, tap_dy in task.spots
    ]

    started = time.monotonic()
    outcome = task.execute(frame)
    result["executed"] = bool(outcome)
    result["seconds"] = round(time.monotonic() - started, 1)
    result["failure_reason"] = task.last_failure_reason
    after = client.get_screenshot(fresh=True)
    if after is not None:
        state_after = classifier.classify(after).state
        result["state_after"] = state_after
        path = save_dir / "run_upgrade_once_after.png"
        cv2.imwrite(str(path), after)
        result["after_frame"] = str(path)

        if state_after != "city_view":
            for attempt in range(2):
                client.press_back()
                time.sleep(1.6)
                fresh = client.get_screenshot(fresh=True)
                if fresh is None:
                    break
                state_after = classifier.classify(fresh).state
                if state_after == "city_view":
                    break
            result["state_after_recovery"] = state_after
            recovered = save_dir / "run_upgrade_once_recovered.png"
            if fresh is not None:
                cv2.imwrite(str(recovered), fresh)
                result["recovered_frame"] = str(recovered)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if outcome else 4


if __name__ == "__main__":
    raise SystemExit(main())
