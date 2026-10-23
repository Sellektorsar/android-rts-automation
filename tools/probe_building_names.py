"""
Точечный зонд названий зданий текущего города.

По списку координат тапа: снимает чистый кадр, тапает кандидата, читает
баннер с названием над зданием, сохраняет оба кадра и возвращается в город
аппаратным Back. Ничего не покупает и не подтверждает.

Использование:
    python tools/probe_building_names.py \
        --points "470,566:forge;260,360:castle;265,540:red_castle"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.ocr import fuzzy_contains, normalize_text, read_text
from src.perception import ScreenClassifier
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.vision import Vision

RADIAL_TEMPLATES = ("btn_building_upgrade", "btn_radial_upgrade")


def _match_name(text: str) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None
    for alias, name in BuildingUpgradeTask.BUILDING_ALIASES:
        if fuzzy_contains(normalized, [alias], threshold=0.78):
            return name
    return None


def _radial(vision: Vision, frame) -> Optional[dict]:
    for name in RADIAL_TEMPLATES:
        if not vision.has_template(name):
            continue
        match = vision.find_template_multiscale(
            frame, name, threshold=0.72, scales=(0.8, 0.9, 1.0, 1.1, 1.2)
        )
        if match:
            return {
                "template": name,
                "x": int(match[0]),
                "y": int(match[1]),
                "conf": round(float(match[2]), 3),
            }
    return None


def _banner_text(frame, tap: tuple[int, int]):
    tx, ty = tap
    height, width = frame.shape[:2]
    x1 = max(0, tx - 280)
    y1 = max(0, ty - 320)
    x2 = min(width, tx + 280)
    y2 = min(height, ty - 60)
    if y2 <= y1:
        return ""
    return read_text(
        frame, (x1, y1, x2 - x1, y2 - y1), scale=1.5, min_confidence=0.35
    ).text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR названий зданий по координатам")
    parser.add_argument(
        "--points",
        required=True,
        help="координаты через ';', формат x,y[:метка]",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default="logs/inspect")
    parser.add_argument("--settle", type=float, default=1.6)
    args = parser.parse_args()

    points = []
    for chunk in args.points.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        coords, _, label = chunk.partition(":")
        x_str, y_str = coords.split(",")
        points.append((int(x_str), int(y_str), label or ""))

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
        print("Нет подключения к ADB")
        return 2
    client.wake_screen()
    client.collapse_statusbar()

    package = config["game"]["package"]
    if not client.is_game_foreground(package):
        print("Игра не на переднем плане")
        return 3

    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = classifier.classify(client.get_screenshot(fresh=True)).state
    if state != "city_view":
        print(f"Ожидался город, получено: {state}")
        return 4

    report = []
    for index, (x, y, label) in enumerate(points, start=1):
        clean = client.get_screenshot(fresh=True)
        if clean is None:
            print(f"[{index}/{len(points)}] ({x},{y}) нет кадра")
            continue
        clean_path = out_dir / f"nameprobe_clean_{x}_{y}.png"
        cv2.imwrite(str(clean_path), clean)

        client.tap(x, y)
        time.sleep(args.settle)
        tapped = client.get_screenshot(fresh=True)
        if tapped is None:
            print(f"[{index}/{len(points)}] ({x},{y}) нет кадра после тапа")
            client.press_back()
            continue
        tapped_path = out_dir / f"nameprobe_tapped_{x}_{y}.png"
        cv2.imwrite(str(tapped_path), tapped)

        raw = _banner_text(tapped, (x, y))
        name = _match_name(raw)
        radial = _radial(vision, tapped)
        tapped_state = classifier.classify(tapped).state
        entry = {
            "tap": [x, y],
            "label": label,
            "building": name,
            "label_raw": raw[:120],
            "tapped_state": tapped_state,
            "radial": radial,
            "clean_frame": str(clean_path),
            "tapped_frame": str(tapped_path),
        }
        report.append(entry)
        print(
            f"[{index}/{len(points)}] ({x},{y}) {label or '?':12s} "
            f"state={tapped_state} radial={bool(radial)} "
            f"name={name} raw='{raw[:60]}'"
        )

        if tapped_state == "city_view" and radial is None:
            # Ничего не открылось: Back в городе вызвал бы диалог выхода.
            entry["back_state"] = "skipped_no_overlay"
            continue

        client.press_back()
        time.sleep(1.0)
        back = client.get_screenshot(fresh=True)
        back_state = classifier.classify(back).state if back is not None else "no_frame"
        if back_state != "city_view":
            # В городе Back мог открыть диалог выхода — второй Back его отменяет.
            client.press_back()
            time.sleep(1.2)
            back = client.get_screenshot(fresh=True)
            back_state = (
                classifier.classify(back).state if back is not None else "no_frame"
            )
        entry["back_state"] = back_state
        if back_state != "city_view":
            print(f"  ! возврат в город не удался: {back_state}; остановка")
            break

    report_path = out_dir / "building_names.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Отчёт: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
