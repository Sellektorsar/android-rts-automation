"""Диагностика ячеек: текущий кадр города и все совпадения шаблона пустой ячейки.

Сохраняет:
  logs/inspect/city_current.png       — полный кадр;
  logs/inspect/plots_current_montage.png — кадр с пронумерованными кандидатами;
  logs/inspect/plots_candidates.png   — крупные вырезки кандидатов.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adb_client import AdbClient  # noqa: E402
from src.vision import Vision  # noqa: E402


def main() -> None:
    import yaml

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))["device"]
    client = AdbClient(
        adb_path=cfg["adb_path"],
        ip=cfg["ip"],
        port=cfg["port"],
        timeout=cfg.get("timeout_sec", 10),
        frame_ttl=cfg.get("frame_ttl_sec", 1.5),
    )
    if not client.connect(allow_discovery=True):
        print("Не удалось подключиться")
        return

    screen = client.get_screenshot(fresh=True)
    if screen is None:
        print("Нет кадра")
        return

    out_dir = Path("logs/inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "city_current.png"), screen)

    vision = Vision(templates_dir="assets")
    height, width = screen.shape[:2]
    region = (int(width * 0.34), int(height * 0.34), int(width * 0.66), int(height * 0.62))
    matches = vision.find_all_templates_multiscale(
        screen,
        "plot_empty_live",
        threshold=0.45,
        min_distance=70,
        region=region,
        scales=(0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6),
        blur=3,
    )
    print(f"Кандидатов: {len(matches)}")
    for index, (x, y, conf, rel) in enumerate(matches):
        print(f"  {index}: ({x}, {y}) conf={conf:.3f} scale={rel}")

    annotated = screen.copy()
    crops = []
    for index, (x, y, conf, _rel) in enumerate(matches):
        cv2.circle(annotated, (x, y), 34, (0, 0, 255), 3)
        cv2.putText(
            annotated,
            str(index),
            (x - 12, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
        crop = screen[max(0, y - 75) : y + 75, max(0, x - 75) : x + 75]
        crop = cv2.resize(crop, (150, 150))
        cv2.putText(
            crop,
            f"{index}:{conf:.2f}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        crops.append(crop)

    cv2.imwrite(str(out_dir / "plots_current_montage.png"), annotated)
    if crops:
        columns = min(4, len(crops))
        rows = (len(crops) + columns - 1) // columns
        blank = np.zeros((150, 150, 3), dtype=np.uint8)
        while len(crops) < rows * columns:
            crops.append(blank)
        grid = np.vstack([np.hstack(crops[r * columns : (r + 1) * columns]) for r in range(rows)])
        cv2.imwrite(str(out_dir / "plots_candidates.png"), grid)

    print(f"Сохранено: {out_dir / 'city_current.png'}")


if __name__ == "__main__":
    main()
