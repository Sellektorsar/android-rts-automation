"""
Безопасный зонд экрана «Королевская Дорога».

Что делает:
1. Проверяет, что игра на переднем плане и мы в городе.
2. По OCR находит нижний баннер «Вызов Спасителя порядка, этап N».
3. Нажимает только проверенную иконку короны `(75, 1960)`.
4. Ждёт якорь `title_royal_road`, читает главу, текущий квест и кнопки.
5. Сохраняет диагностический кадр и возвращается в город по `Back`.

Скрипт НЕ нажимает «Вперёд», «Забрать награды», «Испытание» и любые
кнопки оплаты. Это калибровочный инструмент, а не боевой автопилот.

Использование:
    python tools/probe_royal_road.py
    python tools/probe_royal_road.py --json
    python tools/probe_royal_road.py --forward   # один тап «Вперёд», без боя
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
from src.ocr import fuzzy_contains, normalize_text, read_text
from src.perception import ScreenClassifier
from src.vision import Vision

CROWN_POINT = (75, 1960)  # проверено на живом устройстве 2026-10-22
BANNER_KEYWORDS = ("спасителя порядка", "вызов спасителя", "этап")
FORWARD_TEMPLATE = "btn_vpered_active"


def _cfg(args) -> dict:
    with open(args.config, "r", encoding="utf-8") as handle:
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


def _banner_item(screen):
    """Ищет в нижней части города текст баннера главы."""
    height, width = screen.shape[:2]
    region = (0, int(height * 0.72), int(width * 0.72), int(height * 0.12))
    result = read_text(screen, region, scale=2.0, min_confidence=0.25)
    for item in result.items:
        normalized = normalize_text(item.text)
        if not normalized:
            continue
        has_chapter = (
            fuzzy_contains(normalized, BANNER_KEYWORDS, threshold=0.68)
            or "cnac" in normalized
            or "nopg" in normalized
        )
        if has_chapter and any(char.isdigit() for char in normalized):
            return item, result
    return None, result


def _royal_road_details(screen, vision: Vision) -> Dict[str, Any]:
    """Читает заголовок главы, кнопку «Вперёд» и нижнюю награду."""
    height, width = screen.shape[:2]
    details: Dict[str, Any] = {"ocr_items": []}

    title_region = (int(width * 0.15), int(height * 0.32), int(width * 0.70), int(height * 0.06))
    title = read_text(screen, title_region, scale=2.0, min_confidence=0.30)
    details["chapter_title"] = title.text

    full = read_text(screen, scale=0.5, min_confidence=0.35)
    for item in full.items:
        # Оставляем только строки списка заданий и нижней награды.
        if int(height * 0.32) <= item.box[1] <= int(height * 0.86):
            details["ocr_items"].append(
                {
                    "text": item.text,
                    "confidence": round(item.confidence, 2),
                    "box": list(item.box),
                    "center": list(item.center),
                }
            )

    forward = vision.find_template(
        screen,
        FORWARD_TEMPLATE,
        threshold=0.75,
        region=(int(width * 0.68), int(height * 0.33), int(width * 0.24), int(height * 0.12)),
    )
    if forward:
        details["forward_button"] = {
            "x": int(forward[0]),
            "y": int(forward[1]),
            "confidence": round(float(forward[2]), 3),
            "source": "template",
        }
    else:
        # OCR-резерв: RapidOCR читает «BnepeA» даже когда шаблон не совпал.
        for item in full.items:
            normalized = normalize_text(item.text)
            if item.box[1] > int(height * 0.45):
                continue
            if "bnepe" in normalized or "vpered" in normalized:
                details["forward_button"] = {
                    "x": int(item.center[0]),
                    "y": int(item.center[1]),
                    "confidence": round(float(item.confidence), 3),
                    "source": "ocr",
                }
                break

    for item in full.items:
        normalized = normalize_text(item.text)
        looks_like_claim = (
            fuzzy_contains(item.text, ["забрать награды"], threshold=0.55)
            or "zabp" in normalized
        ) and item.box[1] >= int(height * 0.78)
        if looks_like_claim:
            details["claim_rewards_button"] = {
                "x": int(item.center[0]),
                "y": int(item.center[1]),
                "text": item.text,
                "confidence": round(float(item.confidence), 3),
            }
            break
    return details


def _return_to_city(client: AdbClient, classifier: ScreenClassifier, max_attempts: int = 3):
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Безопасный зонд Королевской Дороги")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Вывести только JSON")
    parser.add_argument("--save", default="logs/inspect/royal_road_probe.png")
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Проверить поведение «Вперёд» (закрывает окно); бой не начинается",
    )
    parser.add_argument("--no-return", action="store_true", help="Не возвращаться в город (отладка)")
    args = parser.parse_args()

    config = _cfg(args)
    client = _connect(config, args)
    vision = Vision(templates_dir="assets")
    classifier = ScreenClassifier(
        vision,
        ocr_scale=config.get("perception", {}).get("ocr_scale", 0.5),
    )

    if not client.is_game_foreground(config["game"]["package"]):
        logger.error("Игра не на переднем плане — зонд не запускается.")
        return 2

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        logger.error("Нет кадра")
        return 1
    if float(frame.mean()) < 2.0:
        logger.error("Кадр чёрный — экран выключен")
        return 3

    state = classifier.classify(frame)
    if state.state != "city_view":
        logger.error(f"Ожидался city_view, получен {state.state} ({state.source}).")
        return 4

    banner, banner_result = _banner_item(frame)
    if banner is None:
        logger.error(f"Баннер главы не найден. OCR: {banner_result.text!r}")
        return 5

    logger.info(f"Баннер: {banner.text!r}; жмём корону {CROWN_POINT}.")
    client.tap(*CROWN_POINT)
    time.sleep(3.0)

    after = client.get_screenshot(fresh=True)
    if after is None:
        logger.error("Нет кадра после открытия Королевской Дороги")
        return 1

    after_state = classifier.classify(after)
    if after_state.state != "royal_road":
        logger.error(
            f"Королевская Дорога не открылась: state={after_state.state}, "
            f"source={after_state.source}, conf={after_state.confidence:.2f}."
        )
        if not args.no_return:
            _return_to_city(client, classifier)
        return 6

    details = _royal_road_details(after, vision)
    details["screen_state"] = after_state.state
    details["confidence"] = round(after_state.confidence, 3)
    details["source"] = after_state.source
    details["banner_text"] = banner.text

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.save, after)
    details["saved_to"] = args.save

    if args.forward:
        forward = details.get("forward_button")
        if not forward:
            logger.error("Кнопка «Вперёд» не найдена — не нажимаем ничего.")
            details["forward_error"] = "button_not_found"
        else:
            logger.info(
                f"Жмём «Вперёд» ({forward['x']}, {forward['y']}); "
                "«Испытание» и оплату не нажимаем."
            )
            client.tap(forward["x"], forward["y"])
            time.sleep(3.0)
            stage_frame = client.get_screenshot(fresh=True)
            if stage_frame is None:
                details["stage_error"] = "no_frame"
            else:
                stage_state = classifier.classify(stage_frame)
                stage_ocr = read_text(stage_frame, scale=0.5, min_confidence=0.35)
                details["stage_screen"] = {
                    "state": stage_state.state,
                    "confidence": round(stage_state.confidence, 3),
                    "source": stage_state.source,
                    "ocr_text": stage_ocr.text[:1500],
                }
                stage_path = str(Path(args.save).with_name("royal_road_forward.png"))
                cv2.imwrite(stage_path, stage_frame)
                details["stage_saved_to"] = stage_path
                # Только фиксируем наличие кнопки боя, но не нажимаем её.
                for template in ("btn_challenge", "btn_stage_challenge"):
                    match = vision.find_template(stage_frame, template, threshold=0.75)
                    if match:
                        details["stage_screen"][f"{template}_point"] = {
                            "x": int(match[0]),
                            "y": int(match[1]),
                            "confidence": round(float(match[2]), 3),
                        }

    if not args.no_return:
        ok, states = _return_to_city(client, classifier)
        details["returned_to_city"] = ok
        details["return_states"] = states

    if args.json:
        print(json.dumps(details, ensure_ascii=False, indent=2))
    else:
        logger.success(f"Глава: {details.get('chapter_title')!r}")
        logger.info(f"Кнопка «Вперёд»: {details.get('forward_button')}")
        logger.info(f"Кнопка награды: {details.get('claim_rewards_button')}")
        logger.info(f"Сохранено: {details['saved_to']}")
        logger.info(f"Возврат в город: {details.get('returned_to_city')} {details.get('return_states')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
