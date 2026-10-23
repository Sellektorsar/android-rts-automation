"""Живой зонд постройки на одной ячейке с выбором типа.

Проверяет полный цикл: тап по ячейке -> экран постройки -> медленный
свайп карусели до нужного типа -> проверка требований и синей кнопки ->
постройка -> возврат в город. Ничего не нажимает, кроме синей кнопки.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml
from loguru import logger

from src.adb_client import AdbClient
from src.ocr import fuzzy_contains, read_text
from src.vision import Vision

OUT = Path("logs/inspect")
OUT.mkdir(parents=True, exist_ok=True)

TITLE_REL = (0.30, 0.03, 0.72, 0.10)
STRIP_Y = 0.265


def rel_px(region, w, h):
    x1, y1, x2, y2 = region
    return int(x1 * w), int(y1 * h), int((x2 - x1) * w), int((y2 - y1) * h)


def title_text(frame):
    h, w = frame.shape[:2]
    res = read_text(frame, rel_px(TITLE_REL, w, h), scale=2.0, min_confidence=0.2)
    conf = max([i.confidence for i in res.items], default=0.0)
    return res.text, conf


def find_blue(frame, vision):
    h, w = frame.shape[:2]
    return vision.find_template(
        frame, "btn_build", threshold=0.80,
        region=(int(w * 0.55), int(h * 0.90), int(w * 0.45), int(h * 0.10)),
    )


def blue_active(frame, match, vision):
    x, y, conf = match
    tpl = vision.templates_cache["btn_build"]
    th, tw = tpl.shape[:2]
    x1, y1 = max(0, x - tw // 2), max(0, y - th // 2)
    roi = frame[y1:y1 + th, x1:x1 + tw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = float(hsv[:, :, 1].mean())
    b, _, r = [float(roi[:, :, i].mean()) for i in range(3)]
    return s, b - r


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    dev = cfg["device"]
    client = AdbClient(adb_path=dev["adb_path"], ip=dev["ip"], port=dev["port"])
    client.connect()
    vision = Vision("assets/templates")

    x = int(sys.argv[1]) if len(sys.argv) > 1 else 728
    y = int(sys.argv[2]) if len(sys.argv) > 2 else 1856
    target = sys.argv[3] if len(sys.argv) > 3 else "лагерь"

    # 1. Открыть экран постройки.
    client.tap(x, y)
    time.sleep(2.6)
    frame = client.get_screenshot(fresh=True)
    cv2.imwrite(str(OUT / "flow1_open.png"), frame)
    blue = find_blue(frame, vision)
    print("build screen:", blue)
    if not blue:
        print("не открылось")
        return

    # 2. Выбрать тип: флик по строке подписей. Игра требует импульс,
    # медленный свайп не перелистывает. Флик может перескочить позицию,
    # поэтому направление выбираем по названию текущего типа.
    h, w = frame.shape[:2]
    strip_y = int(h * STRIP_Y)
    sequence = [
        ("farm", ("ферма", "ferma", "pepma", "epma")),
        ("sawmill", ("лесопил", "lesop", "necon")),
        ("camp", ("лагер", "lager", "ucheb")),
        ("tent", ("палат", "palat", "naatk", "med")),
        ("iron", ("рудник", "rudnik", "pyahnk")),
    ]
    aliases = dict(sequence)

    def current_index(text: str) -> int:
        for idx, (_, words) in enumerate(sequence):
            if fuzzy_contains(text, words, threshold=0.5):
                return idx
        return -1

    target_idx = current_index(target)
    for step in range(7):
        text, conf = title_text(frame)
        idx = current_index(text)
        print(f"step {step}: title={text!r} conf={conf:.2f} idx={idx}")
        if idx == target_idx or fuzzy_contains(text, [target], threshold=0.55):
            break
        direction = "left" if (idx < 0 or idx < target_idx) else "right"
        if direction == "left":
            client.swipe(int(w * 0.78), strip_y, int(w * 0.28), strip_y, 300)
        else:
            client.swipe(int(w * 0.28), strip_y, int(w * 0.78), strip_y, 300)
        time.sleep(1.2)
        frame = client.get_screenshot(fresh=True)
        cv2.imwrite(str(OUT / f"flow2_type_{step}.png"), frame)
    else:
        print("тип не выбран")
        client.press_back()
        return

    cv2.imwrite(str(OUT / "flow3_selected.png"), frame)

    # 3. Проверить синюю кнопку.
    blue = find_blue(frame, vision)
    print("blue:", blue)
    if not blue:
        print("синяя кнопка не найдена")
        client.press_back()
        return
    sat, b_r = blue_active(frame, blue, vision)
    print(f"blue active check: S={sat:.0f} B-R={b_r:.0f}")
    if sat < 60 or b_r < 10:
        print("кнопка неактивна — не нажимаем")
        client.press_back()
        return

    # 4. Построить.
    ok_before = frame.copy()
    client.tap(int(blue[0]), int(blue[1]))
    time.sleep(2.2)
    after = client.get_screenshot(fresh=True)
    cv2.imwrite(str(OUT / "flow4_after_build.png"), after)
    still = find_blue(after, vision) if after is not None else None
    print("after: build screen still:", still)
    diff = Vision.diff_score(ok_before, after)
    print("frame diff:", round(diff, 3))
    if still:
        print("экран постройки остался — нажатие не сработало")
        client.press_back()
        return

    print("ПОСТРОЕНО; ждём город")
    time.sleep(1.0)
    city = client.get_screenshot(fresh=True)
    cv2.imwrite(str(OUT / "flow5_city.png"), city)
    nav = vision.find_template(city, "btn_nav_quests", threshold=0.75, region=(0, int(h * 0.86), w, int(h * 0.14)))
    print("city nav:", nav)


if __name__ == "__main__":
    main()
