"""
Безопасный зонд выбора здания по шаблону (BuildingUpgrade).

Что делает:
1. Проверяет, что игра на переднем плане и мы в городе.
2. Находит здание мультимасштабно по шаблону (по умолчанию
   `building_wall_tower` — башня стены; камера города может менять зум).
3. ``--dry-run``: только координаты/уверенность, без единого тапа.
4. Живой режим: тап по найденному центру здания -> радиальное меню
   («Улучшить») -> панель требований. Панель читается OCR-ом, состояние
   синей кнопки проверяется, но НИЧЕГО не нажимается: ни синего
   подтверждения, ни зелёной кнопки гемов.
5. Возвращается в город по Back.

Задача зонда — проверить механику шаблонных точек и зафиксировать
реальное состояние здания на текущей камере города.

Использование:
    python tools/probe_building_template.py --dry-run
    python tools/probe_building_template.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.perception import ScreenClassifier
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.vision import Vision
from tools.probe_city_grid import _building_label


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


def _wait_city(
    client: AdbClient, classifier: ScreenClassifier, attempts: int = 5
) -> Tuple[bool, list]:
    states = []
    for _ in range(attempts):
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break
        state = classifier.classify(frame).state
        states.append(state)
        if state == "city_view":
            return True, states
        # Любое не-городское состояние закрываем Back-ом (диалог выхода
        # «Выйти из игры?» так же отменяется Back-ом). Пауза после —
        # чтобы классификатор не застал анимацию закрытия.
        client.press_back()
        time.sleep(2.2)
    frame = client.get_screenshot(fresh=True)
    if frame is not None:
        states.append(classifier.classify(frame).state)
        if states[-1] == "city_view":
            return True, states
    return False, states


def _output(result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Зонд выбора здания по шаблону")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--template", default="building_wall_tower")
    parser.add_argument("--building", default="wall")
    parser.add_argument(
        "--point",
        default=None,
        help="Проверочный тап по ручной точке X,Y вместо поиска шаблона "
        "(для верификации кадра и разметки нового шаблона)",
    )
    parser.add_argument(
        "--tap-dy",
        type=int,
        default=30,
        help="Тап ниже центра шаблона (кликабельная подошва в изометрии)",
    )
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

    result: Dict[str, Any] = {"building": args.building, "template": args.template}

    if not client.is_game_foreground(config["game"]["package"]):
        result["error"] = "game_not_foreground"
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["error"])
        return 2

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        result["error"] = "no_frame"
        _output(result, args.json)
        return 3
    state = classifier.classify(frame)
    result["start_state"] = state.state
    if state.state != "city_view":
        result["error"] = f"expected_city_view_{state.state}"
        _output(result, args.json)
        return 4

    spot = BuildingUpgradeTask(
        client,
        vision,
        interval_sec=1,
        spots=[
            {
                "name": args.building,
                "template": args.template,
                "tap_dy": args.tap_dy,
            }
        ],
    )
    if args.point:
        try:
            tap_x, tap_y = (int(v) for v in args.point.split(","))
        except ValueError:
            result["error"] = "bad_point_format"
            _output(result, args.json)
            return 5
        result["manual_point"] = [tap_x, tap_y]
    else:
        match = vision.find_template_multiscale(
            frame,
            args.template,
            threshold=BuildingUpgradeTask.SPOT_TEMPLATE_THRESHOLD,
            scales=BuildingUpgradeTask.SPOT_TEMPLATE_SCALES,
            blur=BuildingUpgradeTask.SPOT_TEMPLATE_BLUR,
        )
        if match is None:
            result["error"] = "building_template_not_found"
            _output(result, args.json)
            return 5
        bx, by, conf, scale = match
        tap_x = int(bx)
        tap_y = int(by) + round(args.tap_dy * scale)
        result["building_match"] = {
            "x": int(bx),
            "y": int(by),
            "confidence": round(float(conf), 3),
            "scale": round(float(scale), 2),
        }
        template = vision.templates_cache.get(args.template)
        if template is not None:
            marked = frame.copy()
            th, tw = template.shape[:2]
            hw, hh = int(tw * scale / 2), int(th * scale / 2)
            cv2.rectangle(marked, (bx - hw, by - hh), (bx + hw, by + hh), (0, 255, 0), 3)
            cv2.line(marked, (bx, by), (tap_x, tap_y), (255, 0, 0), 2)
            cv2.circle(marked, (tap_x, tap_y), 10, (0, 0, 255), -1)
            path = save_dir / f"probe_bld_match_{args.building}.png"
            cv2.imwrite(str(path), marked)
            result["match_frame"] = str(path)

    if args.dry_run:
        result["dry_run"] = True
        _output(result, args.json)
        return 0

    # --- Живой режим: тап по зданию -----------------------------------
    def _search_radial(candidate):
        return spot._find_radial(candidate, center=(tap_x, tap_y))

    def _wait_radial(timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            candidate = client.get_screenshot(fresh=True)
            if candidate is None:
                time.sleep(0.6)
                continue
            found = _search_radial(candidate)
            if found:
                return found, candidate
            time.sleep(0.5)
        return None, None

    # Меню могло остаться открытым с прошлого действия: тап по зданию в
    # этом состоянии его закрывает, а не открывает.
    radial, radial_frame = _search_radial(frame), frame
    result["menu_already_open"] = radial is not None
    if radial is None:
        client.tap(tap_x, tap_y)
        time.sleep(1.5)
        radial, radial_frame = _wait_radial(4.5)
    if radial is None:
        # Повторный тап только после проверки, что меню точно закрыто:
        # иначе он погасит меню, открывшееся с задержкой.
        confirm_frame = client.get_screenshot(fresh=True)
        if confirm_frame is not None and _search_radial(confirm_frame) is None:
            result["tap_retry"] = True
            client.tap(tap_x, tap_y)
            time.sleep(1.5)
            radial, radial_frame = _wait_radial(4.5)

    if radial is None:
        result["error"] = "radial_not_found"
        path = save_dir / f"probe_bld_no_radial_{args.building}.png"
        failed = client.get_screenshot(fresh=True)
        if failed is not None:
            cv2.imwrite(str(path), failed)
            result["saved_to"] = str(path)
        ok, states = _wait_city(client, classifier)
        result["returned_to_city"] = ok
        result["return_states"] = states
        _output(result, args.json)
        return 6

    rx, ry, rconf = radial[0], radial[1], radial[2]
    result["radial_match"] = {"x": int(rx), "y": int(ry), "confidence": round(float(rconf), 3)}
    label_name, label_raw = _building_label(radial_frame, (tap_x, tap_y))
    result["building_label"] = {"name": label_name, "raw": label_raw[:80]}
    path = save_dir / f"probe_bld_radial_{args.building}.png"
    cv2.imwrite(str(path), radial_frame)

    client.tap(rx, ry)
    time.sleep(1.8)

    # --- Панель требований -------------------------------------------
    panel_frame = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        candidate = client.get_screenshot(fresh=True)
        if candidate is None:
            time.sleep(0.6)
            continue
        required = vision.find_template_multiscale(
            candidate,
            BuildingUpgradeTask.REQUIRED_TEMPLATE,
            threshold=0.75,
            region=spot._required_region(candidate),
            scales=(0.8, 0.9, 1.0, 1.1, 1.2),
        )
        if required:
            panel_frame = candidate
            break
        time.sleep(0.5)

    if panel_frame is None:
        result["error"] = "upgrade_panel_not_found"
        path = save_dir / f"probe_bld_no_panel_{args.building}.png"
        failed = client.get_screenshot(fresh=True)
        if failed is not None:
            cv2.imwrite(str(path), failed)
            result["saved_to"] = str(path)
        ok, states = _wait_city(client, classifier)
        result["returned_to_city"] = ok
        result["return_states"] = states
        _output(result, args.json)
        return 7

    path = save_dir / f"probe_bld_panel_{args.building}.png"
    cv2.imwrite(str(path), panel_frame)
    result["panel_saved_to"] = str(path)

    # OCR панели и состояние синей кнопки (только чтение, без тапов).
    context = spot._read_upgrade_context(panel_frame)
    result["panel_context"] = {k: v for k, v in context.items() if k != "target_level"}
    if context.get("target_level") is not None:
        result["panel_context"]["target_level"] = context["target_level"]

    height, width = panel_frame.shape[:2]
    confirm_region = (int(width * 0.55), int(height * 0.90), int(width * 0.45), int(height * 0.10))
    confirm = vision.find_template(
        panel_frame,
        BuildingUpgradeTask.CONFIRM_TEMPLATE,
        threshold=0.84,
        region=confirm_region,
    )
    button_state: Dict[str, Any] = {"found": bool(confirm)}
    if confirm:
        cx, cy, cconf = confirm[:3]
        button_state.update({
            "x": int(cx),
            "y": int(cy),
            "confidence": round(float(cconf), 3),
        })
        template = vision.templates_cache.get(BuildingUpgradeTask.CONFIRM_TEMPLATE)
        if template is not None:
            th, tw = template.shape[:2]
            roi = panel_frame[max(0, cy - th // 2) : cy - th // 2 + th, max(0, cx - tw // 2) : cx - tw // 2 + tw]
            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                saturation = float(hsv[:, :, 1].mean())
                blue, _, red = (float(roi[:, :, i].mean()) for i in range(3))
                button_state["saturation"] = round(saturation, 1)
                button_state["blue_minus_red"] = round(blue - red, 1)
                button_state["active"] = bool(saturation >= 60.0 and blue >= red + 10.0)
    result["confirm_button"] = button_state

    # --- Возврат в город (без нажатия каких-либо кнопок панели) -------
    ok, states = _wait_city(client, classifier)
    result["returned_to_city"] = ok
    result["return_states"] = states

    _output(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

