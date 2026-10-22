"""
Безопасный зонд карты кампании «Спаситель порядка» и этапа 7.

Маршрут:
1. Город → тап по основанию портала кампании (шаблон `portal_campaign`).
2. Классификация `campaign_map` по якорю `title_campaign`.
3. Тап по синему флагу этапа 7 (шаблон `campaign_stage_7`).
4. Снимок и классификация брифинга/карты; кнопка «Испытание» НЕ нажимается.
5. Возврат в город по `Back`.

Скрипт не запускает бой и не нажимает оплату. Это калибровочный
инструмент для сбора данных о кампании.

Использование:
    python tools/probe_campaign_map.py
    python tools/probe_campaign_map.py --json
    python tools/probe_campaign_map.py --stage7   # открыть этап 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.ocr import read_text
from src.perception import ScreenClassifier
from src.vision import Vision

PORTAL_TEMPLATE = "portal_campaign"
STAGE7_TEMPLATE = "campaign_stage_7"


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


def _return_to_city(
    client: AdbClient, classifier: ScreenClassifier, max_attempts: int = 4
) -> tuple[bool, List[str]]:
    states: List[str] = []
    for _ in range(max_attempts):
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break
        classification = classifier.classify(frame)
        states.append(classification.state)
        if classification.state == "city_view":
            return True, states
        client.press_back()
        time.sleep(2.0)
    return False, states


def _match_center(vision: Vision, frame, template: str, threshold: float, region=None):
    match = vision.find_template(frame, template, threshold=threshold, region=region)
    if not match:
        return None
    return {"x": int(match[0]), "y": int(match[1]), "confidence": round(float(match[2]), 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Безопасный зонд карты кампании")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stage7", action="store_true", help="Открыть этап 7 (без боя)")
    parser.add_argument("--save-dir", default="logs/inspect")
    parser.add_argument("--no-return", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision, ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5)
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not client.is_game_foreground(config["game"]["package"]):
        logger.error("Игра не на переднем плане.")
        return 2

    frame = client.get_screenshot(fresh=True)
    if frame is None or float(frame.mean()) < 2.0:
        logger.error("Нет пригодного кадра.")
        return 3

    result: Dict[str, Any] = {}
    start = classifier.classify(frame)
    result["start_state"] = start.state
    if start.state != "city_view":
        logger.error(f"Ожидался city_view, получен {start.state}.")
        return 4

    # Шаблон портала ищем в средней части города, ниже замка.
    height, width = frame.shape[:2]
    portal = _match_center(
        vision,
        frame,
        PORTAL_TEMPLATE,
        threshold=0.80,
        region=(int(width * 0.30), int(height * 0.36), int(width * 0.35), int(height * 0.22)),
    )
    if portal is None:
        logger.error("Портал кампании не найден по шаблону — не нажимаем вслепую.")
        result["error"] = "portal_not_found"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 5

    # vision.find_template возвращает центр совпадения. Проверенная
    # кликабельная точка — основание портала на ~98 px ниже центра арки.
    tap_x = portal["x"]
    tap_y = portal["y"] + 98
    result["portal_match"] = portal
    result["portal_tap"] = [tap_x, tap_y]
    logger.info(f"Портал найден {portal}; жмём основание ({tap_x}, {tap_y}).")
    client.tap(tap_x, tap_y)
    time.sleep(3.0)

    campaign_frame = client.get_screenshot(fresh=True)
    if campaign_frame is None:
        logger.error("Нет кадра после открытия карты.")
        return 1
    campaign_state = classifier.classify(campaign_frame)
    result["campaign_screen"] = {
        "state": campaign_state.state,
        "confidence": round(campaign_state.confidence, 3),
        "source": campaign_state.source,
        "evidence": campaign_state.evidence,
    }
    campaign_path = save_dir / "campaign_map_probe.png"
    cv2.imwrite(str(campaign_path), campaign_frame)
    result["campaign_saved_to"] = str(campaign_path)

    if campaign_state.state != "campaign_map":
        logger.error(f"Карта кампании не открылась: {campaign_state.state}.")
        if not args.no_return:
            ok, states = _return_to_city(client, classifier)
            result["returned_to_city"] = ok
            result["return_states"] = states
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 6

    if args.stage7:
        stage7 = _match_center(
            vision,
            campaign_frame,
            STAGE7_TEMPLATE,
            threshold=0.80,
            region=(int(width * 0.75), int(height * 0.42), int(width * 0.22), int(height * 0.16)),
        )
        if stage7 is None:
            logger.error("Флаг этапа 7 не найден — не нажимаем вслепую.")
            result["stage7_error"] = "template_not_found"
        else:
            # Центр синего флага; кнопка «Испытание» дальше не нажимается.
            stage_x = stage7["x"]
            stage_y = stage7["y"]
            result["stage7_match"] = stage7
            result["stage7_tap"] = [stage_x, stage_y]
            logger.info(f"Флаг этапа 7 найден {stage7}; жмём ({stage_x}, {stage_y}).")
            client.tap(stage_x, stage_y)
            time.sleep(3.0)
            stage_frame = client.get_screenshot(fresh=True)
            if stage_frame is not None:
                stage_state = classifier.classify(stage_frame)
                stage_ocr = read_text(stage_frame, scale=0.5, min_confidence=0.35)
                result["stage_screen"] = {
                    "state": stage_state.state,
                    "confidence": round(stage_state.confidence, 3),
                    "source": stage_state.source,
                    "ocr_text": stage_ocr.text[:1800],
                }
                stage_path = save_dir / "campaign_stage7_probe.png"
                cv2.imwrite(str(stage_path), stage_frame)
                result["stage_saved_to"] = str(stage_path)
                # Фиксируем кнопки боя, но не нажимаем.
                for template in ("btn_challenge", "btn_stage_challenge"):
                    match = vision.find_template(stage_frame, template, threshold=0.75)
                    if match:
                        result["stage_screen"][f"{template}_point"] = {
                            "x": int(match[0]),
                            "y": int(match[1]),
                            "confidence": round(float(match[2]), 3),
                        }

    if not args.no_return:
        ok, states = _return_to_city(client, classifier)
        result["returned_to_city"] = ok
        result["return_states"] = states

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.success(f"Карта: {result.get('campaign_screen')}")
        logger.info(f"Этап 7: {result.get('stage_screen')}")
        logger.info(f"Возврат: {result.get('returned_to_city')} {result.get('return_states')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
