"""
Панорамный зонд города: снимает сетку кадров свайпами камеры.

Выполняет только свайпы по пустой земле (как обход в
BuildingUpgradeTask) и сохраняет кадр на каждой позиции. По зданиям
не тапает, ничего не подтверждает. В конце отыгрывает свайпы назад,
чтобы вернуть камеру в исходное положение.

Использование:
    python tools/probe_city_panorama.py --cols 3 --rows 3 --step-px 600
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


def _connect(config: dict, args) -> AdbClient:
    device = config["device"]
    ip = args.ip or device["ip"]
    port = args.port or device["port"]
    client = AdbClient(
        adb_path=device["adb_path"],
        ip=ip,
        port=port,
        timeout=device.get("timeout_sec", 10),
    )
    if not client.connect(allow_discovery=True):
        raise SystemExit("Нет подключения к ADB")
    client.wake_screen()
    client.collapse_statusbar()
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description="Панорама города свайпами")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--step-px", type=int, default=600)
    parser.add_argument("--duration-ms", type=int, default=900)
    parser.add_argument("--settle", type=float, default=1.4)
    parser.add_argument("--out-dir", default="logs/inspect/panorama")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        print("Нет кадра")
        return 2
    state = classifier.classify(frame).state
    if state != "city_view":
        print(f"Ожидался город, получено: {state}")
        return 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = BuildingUpgradeTask(
        client,
        vision,
        interval_sec=1,
        sweep={
            "enabled": False,
            "cols": args.cols,
            "rows": args.rows,
            "step_px": args.step_px,
            "duration_ms": args.duration_ms,
            "settle_s": args.settle,
        },
    )

    cells = task._sweep_cells(args.cols, args.rows, args.step_px)
    report = {
        "cols": args.cols,
        "rows": args.rows,
        "step_px": args.step_px,
        "cells": cells,
        "frames": [],
    }
    moves: list[tuple[int, int]] = []
    prev_x = prev_y = 0
    ok = True

    for index, (cell_x, cell_y) in enumerate(cells):
        dx, dy = cell_x - prev_x, cell_y - prev_y
        if dx or dy:
            for axis_dx, axis_dy in ((dx, 0), (0, dy)):
                if not (axis_dx or axis_dy):
                    continue
                if not task._pan_camera(axis_dx, axis_dy):
                    print(f"Свайп ({axis_dx},{axis_dy}) не помещается — остановка")
                    ok = False
                    break
                moves.append((axis_dx, axis_dy))
            if not ok:
                break
            prev_x, prev_y = cell_x, cell_y
            time.sleep(args.settle)

        shot = client.get_screenshot(fresh=True)
        if shot is None:
            continue
        path = out_dir / f"pan_r{cell_y:+04d}_c{cell_x:+04d}.png"
        cv2.imwrite(str(path), shot)
        report["frames"].append(
            {
                "offset": [cell_x, cell_y],
                "path": str(path),
                "state": classifier.classify(shot).state,
            }
        )
        print(f"[{index + 1}/{len(cells)}] offset=({cell_x:+d},{cell_y:+d}) -> {path}")

    if moves:
        for axis_dx, axis_dy in reversed(moves):
            task._pan_camera(-axis_dx, -axis_dy)
        time.sleep(args.settle)

    final = client.get_screenshot(fresh=True)
    if final is not None:
        final_path = out_dir / "pan_restored.png"
        cv2.imwrite(str(final_path), final)
        report["restored_frame"] = str(final_path)

    report_path = out_dir / "panorama.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Отчёт: {report_path}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
