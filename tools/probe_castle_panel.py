"""Безопасный зонд перехода к замку через «Перейти».

Открывает панель упёршегося здания (из memory_bank/building_spots.json),
нажимает ТОЛЬКО кнопку «Перейти» в строке требования замка и снимает
последующие кадры. Синие/зелёные кнопки улучшения не нажимаются.

Использование:
    .venv/Scripts/python.exe tools/probe_castle_panel.py --frames 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adb_client import AdbClient
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.vision import Vision
from src.world_model import WorldModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Зонд перехода к панели замка")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--outdir", default="logs/inspect")
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
        logger.error("Не удалось подключиться к ADB")
        return 1
    client.wake_screen()
    client.collapse_statusbar()

    vision = Vision(templates_dir="assets")
    world = WorldModel()
    upgrade_cfg = config["tasks"]["building_upgrade"]
    task = BuildingUpgradeTask(
        client,
        vision,
        interval_sec=1,
        spots=upgrade_cfg.get("spots"),
        world=world,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        logger.error("Нет кадра")
        return 1
    cv2.imwrite(str(outdir / "castle_probe_00_start.png"), frame)

    if task._looks_like_upgrade_panel(frame):
        panel = frame
    else:
        panel = task._acquire_capped_panel(frame)
    if panel is None:
        logger.error("Не удалось открыть панель упёршегося здания")
        return 1
    cv2.imwrite(str(outdir / "castle_probe_01_capped_panel.png"), panel)

    match = task._find_go_button(panel)
    if not match:
        logger.error("На панели нет кнопки «Перейти» — зонд остановлен")
        task._close_panel(panel)
        return 1
    x, y, confidence = match
    logger.info(f"«Перейти»: ({x}, {y}), conf={confidence:.2f}. Нажимаем.")
    client.tap(x, y)

    for index in range(args.frames):
        time.sleep(args.interval)
        shot = client.get_screenshot(fresh=True)
        if shot is None:
            continue
        path = outdir / f"castle_probe_{index + 2:02d}_after_go.png"
        cv2.imwrite(str(path), shot)
        context = task._read_upgrade_context(shot)
        logger.info(
            f"кадр {index + 1}: name={context.get('name')} "
            f"level={context.get('current_level')}->{context.get('target_level')} "
            f"duration={context.get('duration')} panel={task._looks_like_upgrade_panel(shot)}"
        )
    logger.info("Зонд завершён. Кнопки улучшения не нажимались.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
