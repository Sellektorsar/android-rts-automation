"""Живой зонд карусели типов зданий на экране постройки.

Тап по пустой ячейке открывает экран постройки. Сверху — красный
заголовок с названием выбранного типа, ниже горизонтальная карусель:
свайп по строке подписей переключает тип. Зонд фиксирует порядок типов,
их уровни/счётчики и стоимость, чтобы выбрать безопасную раскладку.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml
from loguru import logger

from src.adb_client import AdbClient
from src.ocr import read_text
from src.vision import Vision

OUT = Path("logs/inspect")
OUT.mkdir(parents=True, exist_ok=True)

TITLE_REL = (0.30, 0.03, 0.72, 0.10)
STRIP_Y = 0.265


def rel_to_px(region, width: int, height: int):
    x1, y1, x2, y2 = region
    return (
        int(x1 * width),
        int(y1 * height),
        int((x2 - x1) * width),
        int((y2 - y1) * height),
    )


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    dev = cfg["device"]
    client = AdbClient(adb_path=dev["adb_path"], ip=dev["ip"], port=dev["port"])
    client.connect()
    vision = Vision("assets/templates")

    x = int(sys.argv[1]) if len(sys.argv) > 1 else 728
    y = int(sys.argv[2]) if len(sys.argv) > 2 else 1856

    client.tap(x, y)
    time.sleep(2.5)
    frame = client.get_screenshot(fresh=True)
    if frame is None:
        print("no frame")
        return
    h, w = frame.shape[:2]
    build = vision.find_template(
        frame, "btn_build", threshold=0.80, region=(int(w * 0.55), int(h * 0.90), int(w * 0.45), int(h * 0.10))
    )
    print("build screen:", build)
    if not build:
        cv2.imwrite(str(OUT / "carousel_not_open.png"), frame)
        return

    items = []
    for index in range(8):
        title = read_text(frame, rel_to_px(TITLE_REL, w, h), scale=2.0, min_confidence=0.25)
        strip_y = int(h * STRIP_Y)
        strip = read_text(
            frame,
            rel_to_px((0.05, STRIP_Y - 0.02, 0.95, STRIP_Y + 0.03), w, h),
            scale=2.0,
            min_confidence=0.25,
        )
        items.append(
            {
                "index": index,
                "title": title.text,
                "title_conf": round(max([i.confidence for i in title.items], default=0.0), 3),
                "strip": strip.text,
            }
        )
        print(index, repr(title.text), "conf", round(max([i.confidence for i in title.items], default=0.0), 3), "| strip:", repr(strip.text))
        cv2.imwrite(str(OUT / f"carousel_{index}.png"), frame)

        # Свайп по строке подписей влево = следующий тип.
        client.swipe(int(w * 0.78), strip_y, int(w * 0.28), strip_y, 300)
        time.sleep(1.1)
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break

    (OUT / "carousel_report.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    client.press_back()
    time.sleep(1.2)
    print("done")


if __name__ == "__main__":
    main()
