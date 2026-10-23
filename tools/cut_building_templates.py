"""
Нарезка шаблонов зданий из размеченных кадров датасета.

Источник bbox — assets/dataset/labels.json (quality: verified_tap /
estimated). Скрипт вырезает область здания из указанного кадра и кладёт
её в assets/templates/building_<name>.png. tap_dy печатается для
конфигурации спота BuildingUpgradeTask: смещение тапа вниз от центра
шаблона до кликабельной подошвы (tap_point из разметки).

Использование:
    python tools/cut_building_templates.py [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "assets" / "dataset" / "labels.json"
TEMPLATES = ROOT / "assets" / "templates"

# Шаблон -> (исходный кадр, имя записи в labels.json).
# Добавляйте сюда здания с проверенными bbox по мере разметки.
TEMPLATE_SOURCES: dict[str, tuple[str, str]] = {
    # Кузница LV.5: после апгрейда LV.4->LV.5 облик сменился, старый шаблон
    # давал conf 0.60. Кадр новой камеры, тап (500,620) проверен живым
    # радиальным меню «Кузница».
    "building_forge": ("logs/inspect/live_now2.png", "forge"),
    # Камера сменилась (reset-зум): старый крупный план башни не совпадал
    # на масштабах 0.8-1.2 (conf 0.50). Новый эталон — башня текущей камеры.
    "building_wall_tower": ("logs/inspect/ds_city_now.png", "wall"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Нарезка шаблонов зданий из labels.json")
    parser.add_argument("--force", action="store_true", help="Перезаписывать существующие PNG")
    args = parser.parse_args()

    with open(LABELS, "r", encoding="utf-8") as f:
        labels = json.load(f)

    for template_name, (source, entry_name) in TEMPLATE_SOURCES.items():
        src_path = ROOT / source
        hand = labels.get(source)
        if not src_path.exists() or not hand:
            print(f"[cut] пропущено {template_name}: нет кадра или разметки {source}")
            continue
        entries = [e for e in hand.get("buildings", []) if e.get("name") == entry_name and e.get("bbox")]
        if not entries:
            print(f"[cut] пропущено {template_name}: в {source} нет bbox для '{entry_name}'")
            continue
        entry = entries[0]
        x1, y1, x2, y2 = entry["bbox"]
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"[cut] не прочитан кадр {source}")
            continue
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[cut] пустой bbox для {template_name}: {entry['bbox']}")
            continue

        dst = TEMPLATES / f"{template_name}.png"
        if dst.exists() and not args.force:
            print(f"[cut] {template_name} уже существует (--force для перезаписи)")
        else:
            cv2.imwrite(str(dst), crop)
            print(f"[cut] {template_name}: {crop.shape[1]}x{crop.shape[0]} из {source} bbox={entry['bbox']}")

        tap_point = entry.get("tap_point")
        if tap_point:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            tap_dy = int(tap_point[1] - center[1])
            quality = entry.get("quality", "?")
            print(
                f"     спот: name={entry_name} template={template_name} tap_dy={tap_dy} "
                f"(tap_point={tap_point}, центр bbox={center}, quality={quality})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
