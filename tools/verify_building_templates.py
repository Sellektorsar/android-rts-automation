"""
Офлайн-проверка шаблонов зданий по эталонному датасету.

Для каждого шаблона и каждого полного кадра (1080x2436) из категорий
city/radial/plot/panel запускает мультимасштабный поиск — тот же
алгоритм, что BuildingUpgradeTask._resolve_spot использует вживую —
и печатает лучший матч с масштабом и позицией. Порог задачи — 0.70
(SPOT_TEMPLATE_THRESHOLD); здесь он не отсекает, а только размечает
PASS/MISS, чтобы видеть запас уверенности.

Использование:
    python tools/verify_building_templates.py [--templates a,b] [--categories a,b]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.vision import Vision

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "assets" / "dataset"
MANIFEST = DATASET / "manifest.json"

THRESHOLD = 0.70  # BuildingUpgradeTask.SPOT_TEMPLATE_THRESHOLD
SCALES = (0.80, 0.90, 1.00, 1.10, 1.20)  # BuildingUpgradeTask.SPOT_TEMPLATE_SCALES
DEFAULT_TEMPLATES = ("building_wall_tower", "building_forge", "plot_empty")
DEFAULT_CATEGORIES = ("city", "radial", "plot", "panel")


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка шаблонов зданий по датасету")
    parser.add_argument("--templates", default=",".join(DEFAULT_TEMPLATES))
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    args = parser.parse_args()

    templates = [t.strip() for t in args.templates.split(",") if t.strip()]
    categories = {c.strip() for c in args.categories.split(",") if c.strip()}

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    frames = [
        e for e in manifest["images"]
        if e.get("category") in categories and e.get("frame") == "full"
    ]
    if not frames:
        print("Нет подходящих кадров в манифесте.")
        return 1

    vision = Vision(str(ROOT / "assets"))
    missing = [t for t in templates if not vision.has_template(t)]
    if missing:
        print(f"Нет шаблонов: {', '.join(missing)}")
        return 1

    print(f"Кадров: {len(frames)} (категории: {', '.join(sorted(categories))})")
    print(f"Шаблоны: {', '.join(templates)}; порог задачи {THRESHOLD}; масштабы {SCALES}")
    print("-" * 100)

    stats: dict[str, list[float]] = {t: [] for t in templates}
    for entry in frames:
        img = cv2.imread(str(DATASET / entry["path"]))
        if img is None:
            print(f"[skip] {entry['path']} не прочитан")
            continue
        row = [entry["path"].split("/")[-1]]
        for name in templates:
            best = vision.find_template_multiscale(img, name, threshold=0.0, scales=SCALES)
            if best is None:
                row.append(f"{name}:—")
                continue
            x, y, conf, scale = best
            stats[name].append(conf)
            mark = "PASS" if conf >= THRESHOLD else "miss"
            row.append(f"{name}:{conf:.2f}@{scale:.2f} ({x},{y}) {mark}")
        print(f"{row[0]:42s} {' | '.join(row[1:])}")

    print("-" * 100)
    for name, confs in stats.items():
        if not confs:
            print(f"{name}: ни одного матча")
            continue
        above = sum(1 for c in confs if c >= THRESHOLD)
        print(
            f"{name}: PASS {above}/{len(confs)} кадров, "
            f"conf max={max(confs):.2f} средн={sum(confs) / len(confs):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
