"""
Безопасный зонд автономного поиска здания обходом камеры.

Что делает:
1. Проверяет, что игра на переднем плане и мы в городе.
2. Ищет шаблон здания на текущем кадре (мультимасштабно, с blur).
3. Если здание вне кадра — запускает `BuildingUpgradeTask._sweep_for_spots()`:
   сетка свайпов по пустому городу, на каждой позиции ищутся все
   шаблонные точки. При неудаче камера возвращается назад.
4. Сохраняет кадр с найденным зданием и печатает JSON-отчёт.

Только поиск: по зданиям и UI зонд не тапает. Найденное здание можно
затем проверить живым `tools/probe_building_template.py`.

Использование:
    python tools/probe_camera_search.py --template building_forge --building forge --tap-dy 93
    python tools/probe_camera_search.py --template building_forge --building forge --no-sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception import ScreenClassifier
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.vision import Vision
from tools.probe_building_template import _connect, _load_config


def _match(
    vision: Vision, frame, template: str
) -> Optional[Tuple[int, int, float, float]]:
    return vision.find_template_multiscale(
        frame,
        template,
        threshold=BuildingUpgradeTask.SPOT_TEMPLATE_THRESHOLD,
        scales=BuildingUpgradeTask.SPOT_TEMPLATE_SCALES,
        blur=BuildingUpgradeTask.SPOT_TEMPLATE_BLUR,
    )


def _match_json(match) -> Dict[str, Any]:
    x, y, confidence, scale = match
    return {
        "x": int(x),
        "y": int(y),
        "confidence": round(float(confidence), 3),
        "scale": round(float(scale), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Поиск здания обходом камеры")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--template", default="building_forge")
    parser.add_argument("--building", default="forge")
    parser.add_argument("--tap-dy", type=int, default=93)
    parser.add_argument("--save-dir", default="logs/inspect")
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Только прямой поиск на текущем кадре, без свайпов",
    )
    parser.add_argument("--cols", type=int, default=None, help="Колонок в сетке обхода")
    parser.add_argument("--rows", type=int, default=None, help="Строк в сетке обхода")
    parser.add_argument("--step-px", type=int, default=None, help="Шаг свайпа в пикселях")
    args = parser.parse_args()

    config = _load_config(args.config)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "building": args.building,
        "template": args.template,
    }

    if not client.is_game_foreground(config["game"]["package"]):
        result["error"] = "game_not_foreground"
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["error"])
        return 2

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        result["error"] = "no_frame"
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["error"])
        return 3
    state = classifier.classify(frame)
    result["start_state"] = state.state
    if state.state != "city_view":
        result["error"] = f"expected_city_view_{state.state}"
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["error"])
        return 4

    cv2.imwrite(str(save_dir / f"sweep_start_{args.building}.png"), frame)

    sweep_cfg = dict(
        (config.get("tasks", {}).get("building_upgrade", {}) or {}).get("sweep") or {}
    )
    sweep_cfg["enabled"] = not args.no_sweep
    if args.cols is not None:
        sweep_cfg["cols"] = args.cols
    if args.rows is not None:
        sweep_cfg["rows"] = args.rows
    if args.step_px is not None:
        sweep_cfg["step_px"] = args.step_px

    task = BuildingUpgradeTask(
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
        sweep=sweep_cfg,
    )
    result["sweep_settings"] = task.sweep

    direct = _match(vision, frame, args.template)
    result["direct_match"] = _match_json(direct) if direct else None

    found = direct
    if found is not None:
        result["found_by"] = "direct"
    elif task.sweep["enabled"]:
        started = time.monotonic()
        resolved = task._sweep_for_spots()
        result["sweep_seconds"] = round(time.monotonic() - started, 1)
        if resolved is None:
            result["found_by"] = None
            result["sweep"] = {"used": True, "found": False}
            # Камера возвращена; фиксируем кадр после возврата.
            restored = client.get_screenshot(fresh=True)
            if restored is not None:
                path = save_dir / f"sweep_not_found_{args.building}.png"
                cv2.imwrite(str(path), restored)
                result["restored_frame"] = str(path)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
            return 5

        result["sweep"] = {"used": True, "found": True, "tap": list(resolved[1])}
        # Даём камере догаснуть после последнего свайпа и пересчитываем
        # совпадение на свежем кадре для отчёта.
        after = None
        for _ in range(3):
            after = client.get_screenshot(fresh=True)
            found = _match(vision, after, args.template) if after is not None else None
            if found is not None:
                break
            time.sleep(0.8)
        if after is not None:
            path = save_dir / f"sweep_hit_{args.building}.png"
            cv2.imwrite(str(path), after)
            result["hit_frame"] = str(path)
    else:
        result["found_by"] = None
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
        return 5

    # При находке обходом пересчитываем совпадение на свежем кадре: нужна
    # уверенность для отчёта (метод обхода возвращает только координаты).
    if found is None and result.get("sweep", {}).get("found"):
        result["error"] = "building_lost_after_sweep"
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result)
        return 6

    result["building_match"] = _match_json(found)
    tap_x = int(found[0])
    tap_y = int(found[1]) + round(args.tap_dy * found[3])
    result["tap"] = [tap_x, tap_y]
    result["found_by"] = result.get("found_by") or "sweep"

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"Найдено {args.building} ({args.template}): "
            f"({found[0]}, {found[1]}), conf={found[2]:.3f}, scale={found[3]:.2f}; "
            f"tap=({tap_x}, {tap_y}) — источник: {result['found_by']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
