"""
Сеточный зонд зданий города.

Автономно сканирует город сеткой тапов: после каждого тапа ищет
радиальное меню («Улучшить», шаблоны btn_building_upgrade /
btn_radial_upgrade), экран постройки (btn_build_free / btn_build) или
панель улучшения (title_upgrade_required). Найденные здания сохраняет
в logs/inspect/grid_hit_*.png и JSON-отчёт logs/inspect/grid_scan.json.

Это механизм обнаружения зданий без жёстких координат: после любой
смены камеры карту города можно пересканировать заново. Тап по зданию
только открывает меню; ничего не подтверждается и не покупается.
После каждого тапа меню закрывается аппаратным Back.

Использование:
    python tools/probe_city_grid.py --json
    python tools/probe_city_grid.py --xs 200,400 --ys 900,1150
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.ocr import fuzzy_contains, normalize_text, read_text
from src.perception import ScreenClassifier
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.vision import Vision

DEFAULT_XS = (200, 400, 600, 800, 960)
DEFAULT_YS = (650, 900, 1150, 1400, 1650, 1850)
# Точки, которые не тапаем: нижний правый угол (иконка события/черепа)
# и зона баннера квеста.
EXCLUDED = {(960, 1850), (200, 1850)}

RADIAL_TEMPLATES = ("btn_building_upgrade", "btn_radial_upgrade")
BUILD_TEMPLATES = ("btn_build_free", "btn_build")
PANEL_TEMPLATES = ("title_upgrade_required",)


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


def _find_any(
    vision: Vision, frame, names: Tuple[str, ...], threshold: float
) -> Optional[Tuple[str, int, int, float]]:
    # Мультимасштаб обязателен: радиальное меню — часть мирового слоя и
    # масштабируется вместе с зумом камеры (живой кадр 2026-10-22:
    # btn_building_upgrade 0.793 при scale 0.9; на scale 1.0 ниже 0.5).
    for name in names:
        if not vision.has_template(name):
            continue
        match = vision.find_template_multiscale(
            frame, name, threshold=threshold, scales=(0.8, 0.9, 1.0, 1.1, 1.2)
        )
        if match:
            return name, int(match[0]), int(match[1]), float(match[2])
    return None


def _building_label(
    frame, tap: Tuple[int, int]
) -> Tuple[Optional[str], str]:
    """OCR имени здания: баннер с названием висит над выбранным зданием."""
    tx, ty = tap
    h, w = frame.shape[:2]
    # Баннер висит над зданием вплотную к кольцу (живой кадр 2026-10-23:
    # тап (500,640) -> «Кузница» в (425,515)-(527,551)). Прежняя область
    # заканчивалась на ty-120 и срезала баннер, зато захватывала HUD
    # (отсюда мусор «7,314 0+ ...»).
    x1 = max(0, tx - 260)
    y1 = max(0, ty - 300)
    x2 = min(w, tx + 260)
    y2 = max(y1 + 60, min(h, ty - 80))
    region = (x1, y1, x2 - x1, y2 - y1)
    text = read_text(frame, region, scale=1.5, min_confidence=0.40)
    normalized = normalize_text(text.text)
    name = None
    if normalized:
        for alias, canonical in BuildingUpgradeTask.BUILDING_ALIASES:
            if fuzzy_contains(normalized, [alias], threshold=0.78):
                name = canonical
                break
    return name, text.text.strip()


def _classify_frame(
    vision: Vision, frame
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Определяет, что открылось после тапа."""
    radial = _find_any(vision, frame, RADIAL_TEMPLATES, 0.75)
    if radial:
        return "radial", {
            "template": radial[0],
            "x": radial[1],
            "y": radial[2],
            "conf": round(radial[3], 3),
        }
    build = _find_any(vision, frame, BUILD_TEMPLATES, 0.80)
    if build:
        return "build", {
            "template": build[0],
            "x": build[1],
            "y": build[2],
            "conf": round(build[3], 3),
        }
    panel = _find_any(vision, frame, PANEL_TEMPLATES, 0.78)
    if panel:
        return "panel", {
            "template": panel[0],
            "x": panel[1],
            "y": panel[2],
            "conf": round(panel[3], 3),
        }
    return "none", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Сеточный зонд зданий города")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-dir", default="logs/inspect")
    parser.add_argument(
        "--xs", default=",".join(str(v) for v in DEFAULT_XS),
        help="Колонки сетки по X через запятую",
    )
    parser.add_argument(
        "--ys", default=",".join(str(v) for v in DEFAULT_YS),
        help="Строки сетки по Y через запятую",
    )
    args = parser.parse_args()

    xs = [int(v) for v in args.xs.split(",")]
    ys = [int(v) for v in args.ys.split(",")]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.config)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )

    result: Dict[str, Any] = {"grid": {"xs": xs, "ys": ys}, "hits": []}

    if not client.is_game_foreground(config["game"]["package"]):
        result["error"] = "game_not_foreground"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        result["error"] = "no_frame"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 3
    state = classifier.classify(frame).state
    result["start_state"] = state
    if state != "city_view":
        result["error"] = f"expected_city_view_{state}"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 4

    points: List[Tuple[int, int]] = [
        (x, y) for y in ys for x in xs if (x, y) not in EXCLUDED
    ]
    scanned = 0
    for (x, y) in points:
        # ADB иногда таймаутится на input tap — это не должно ронять скан.
        try:
            client.tap(x, y)
        except Exception as exc:
            print(f"[grid] tap ({x},{y}) не прошёл: {exc}; пропуск точки.")
            time.sleep(2.0)
            continue
        time.sleep(1.5)
        try:
            frame = client.get_screenshot(fresh=True)
        except Exception as exc:
            print(f"[grid] кадр после ({x},{y}) не получен: {exc}; Back и дальше.")
            client.press_back()
            time.sleep(1.2)
            continue
        if frame is None:
            continue
        kind, detail = _classify_frame(vision, frame)
        scanned += 1
        entry: Dict[str, Any] = {"tap": [x, y], "kind": kind}
        if kind != "none":
            name, raw_text = _building_label(frame, (x, y))
            entry["building"] = name
            entry["label_raw"] = raw_text[:80]
        if detail:
            entry["match"] = detail
            stem = f"grid_hit_{x}_{y}_{kind}"
            if entry.get("building"):
                stem += f"_{entry['building']}"
            path = save_dir / f"{stem}.png"
            cv2.imwrite(str(path), frame)
            entry["frame"] = str(path)
            result["hits"].append(entry)
            print(f"[grid] HIT ({x},{y}) {kind} {detail['template']} "
                  f"conf={detail['conf']} building={entry.get('building')} "
                  f"-> {path.name}")

        if kind == "none":
            # Пустой тап ничего не открыл; Back в городе вызвал бы диалог
            # выхода, поэтому просто идём к следующей точке.
            continue

        # Закрываем меню и убеждаемся, что вернулись в город.
        try:
            client.press_back()
        except Exception:
            pass
        time.sleep(1.2)
        try:
            back_frame = client.get_screenshot(fresh=True)
        except Exception:
            time.sleep(2.0)
            back_frame = client.get_screenshot(fresh=True)
        if back_frame is not None:
            back_state = classifier.classify(back_frame).state
            if back_state != "city_view":
                client.press_back()
                time.sleep(2.2)
                back_frame = client.get_screenshot(fresh=True)
                if back_frame is not None:
                    back_state = classifier.classify(back_frame).state
                    if back_state != "city_view":
                        result["aborted_at"] = [x, y, back_state]
                        cv2.imwrite(
                            str(save_dir / "grid_abort_frame.png"), back_frame
                        )
                        break

    result["scanned"] = scanned
    report_path = save_dir / "grid_scan.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result["report"] = str(report_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
