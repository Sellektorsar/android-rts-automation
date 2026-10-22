"""Контролируемый вход в бой этапа 7 и безопасный выход.

Скрипт делает ровно один запуск боя и **не пытается его выиграть**:
1. Открывает карту кампании и брифинг этапа 7 проверенными шаблонами.
2. Запускает `FastCapture` (screenrecord + ffmpeg).
3. Нажимает «Испытание» и ждёт появления боевого экрана по якорю
   `battle_pause`.
4. Сразу открывает паузу, сохраняет её и пытается найти кнопку выхода
   только по OCR-тексту («Выйти», «Покинуть», «Сдаться»). Если кнопки
   выхода нет, возобновляет бой и ждёт его завершения.
5. Возвращается в город.

Защита:
* без флага `--confirm` бой не начинается;
* нажимаются только шаблон «Испытание», кнопка паузы, OCR-кнопки выхода
  и подтверждения, а также `Back`;
* кнопки «Купить», «Оплатить», гемы, «Повторить» и золотой молот
  игнорируются.

Использование:
    python tools/probe_stage7_combat.py --confirm
    python tools/probe_stage7_combat.py --confirm --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.fast_capture import FastCapture
from src.fsm import GameState
from src.ocr import OcrResult, fuzzy_score, read_text
from src.perception import ScreenClassifier, looks_like_payment
from src.vision import Vision

PORTAL_TEMPLATE = "portal_campaign"
STAGE7_TEMPLATE = "campaign_stage_7"
CHALLENGE_TEMPLATE = "btn_stage_challenge"
PAUSE_TEMPLATE = "battle_pause"

SAFE_RESULT_WORDS = ("продолжить", "ок", "закрыть", "выйти", "в город", "на карту")
FORBIDDEN_WORDS = (
    "купить",
    "оплатить",
    "улучшить",
    "гем",
    "алмаз",
    "повторить",
    "отмена",
    "нет",
)


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
    client.keep_screen_awake()
    return client


def _ocr_payload(result: OcrResult) -> List[Dict[str, Any]]:
    return [
        {
            "text": item.text,
            "confidence": round(float(item.confidence), 3),
            "box": [int(value) for value in item.box],
        }
        for item in result.items
    ]


def _find_text_item(
    ocr: OcrResult,
    keywords: tuple[str, ...],
    *,
    threshold: float = 0.72,
    forbidden: tuple[str, ...] = FORBIDDEN_WORDS,
):
    """Ищет OCR-строку по нечёткому совпадению, отсекая опасные слова."""
    best = None
    best_score = 0.0
    for item in ocr.items:
        if item.confidence < 0.40:
            continue
        if any(fuzzy_score(item.text, word) >= 0.78 for word in forbidden):
            continue
        score = max(fuzzy_score(item.text, word) for word in keywords)
        if score >= threshold and score > best_score:
            best = item
            best_score = score
    return best


def _match_center(
    vision: Vision,
    frame,
    template: str,
    threshold: float,
    region: Optional[tuple[int, int, int, int]] = None,
):
    match = vision.find_template(frame, template, threshold=threshold, region=region)
    if not match:
        return None
    return {"x": int(match[0]), "y": int(match[1]), "confidence": round(float(match[2]), 3)}


def _return_to_city(
    client: AdbClient,
    classifier: ScreenClassifier,
    *,
    max_attempts: int = 8,
) -> tuple[bool, List[str]]:
    states: List[str] = []
    for _ in range(max_attempts):
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break
        result = classifier.classify(frame)
        states.append(result.state)
        if result.state == GameState.CITY_VIEW:
            return True, states
        if result.state == GameState.LOADING:
            time.sleep(2.0)
            continue
        if result.state not in BACK_SAFE_STATES:
            logger.warning(
                f"Возврат остановлен на небезопасном экране "
                f"{result.state}; Back не нажимаем."
            )
            break
        client.press_back()
        time.sleep(1.5)
    return False, states


def _open_stage7(
    client: AdbClient,
    vision: Vision,
    classifier: ScreenClassifier,
    save_dir: Path,
    result: Dict[str, Any],
) -> bool:
    frame = client.get_screenshot(fresh=True)
    if frame is None:
        result["error"] = "no_frame"
        return False
    start = classifier.classify(frame)
    result["start_state"] = start.state
    if start.state != GameState.CITY_VIEW:
        result["error"] = f"expected_city_got_{start.state}"
        return False

    height, width = frame.shape[:2]
    # Портал кампании — синий анимированный объект в нижней части города.
    # Камера города может быть смещена/приближена, а свечение меняется,
    # поэтому ищем в широкой нижней области и допускаем более низкую
    # корреляцию. В этой области нет других похожих синих порталов.
    portal = _match_center(
        vision,
        frame,
        PORTAL_TEMPLATE,
        threshold=0.60,
        region=(int(width * 0.25), int(height * 0.45), int(width * 0.50), int(height * 0.40)),
    )
    if portal is None:
        result["error"] = "portal_not_found"
        return False
    result["portal"] = portal
    # Проверенная кликабельная точка — основание портала.
    tap_x, tap_y = portal["x"], portal["y"] + 98
    result["portal_tap"] = [tap_x, tap_y]
    client.tap(tap_x, tap_y)
    time.sleep(3.0)

    campaign = client.get_screenshot(fresh=True)
    if campaign is None:
        result["error"] = "no_campaign_frame"
        return False
    campaign_state = classifier.classify(campaign)
    result["campaign_state"] = campaign_state.state
    if campaign_state.state != GameState.CAMPAIGN_MAP:
        result["error"] = f"campaign_not_opened_{campaign_state.state}"
        return False

    # Флаг этапа 7 бывает ярко-синим (обычная карта) и тёмно-синим,
    # если рядом открыт босс главы и карта затемнена диалогом. Ищем оба
    # варианта в широкой правой области, а не в узком прямоугольнике.
    stage = None
    stage_region = (
        int(width * 0.70),
        int(height * 0.35),
        int(width * 0.30),
        int(height * 0.50),
    )
    for template in (STAGE7_TEMPLATE, "campaign_stage_7_dark"):
        match = _match_center(
            vision,
            campaign,
            template,
            threshold=0.90,
            region=stage_region,
        )
        if match is not None:
            stage = {**match, "template": template}
            break
    if stage is None:
        result["error"] = "stage7_not_found"
        cv2.imwrite(str(save_dir / "campaign_stage7_not_found.png"), campaign)
        return False
    result["stage7"] = stage
    client.tap(stage["x"], stage["y"])
    time.sleep(3.0)

    briefing = client.get_screenshot(fresh=True)
    if briefing is None:
        result["error"] = "no_briefing_frame"
        return False
    briefing_state = classifier.classify(briefing)
    result["briefing_state"] = briefing_state.state
    if briefing_state.state != GameState.STAGE_BRIEFING:
        result["error"] = f"briefing_not_opened_{briefing_state.state}"
        cv2.imwrite(str(save_dir / "stage7_unexpected.png"), briefing)
        return False

    challenge = _match_center(vision, briefing, CHALLENGE_TEMPLATE, threshold=0.85)
    if challenge is None:
        result["error"] = "challenge_button_not_found"
        cv2.imwrite(str(save_dir / "stage7_briefing_no_button.png"), briefing)
        return False
    result["challenge"] = challenge
    return True


def _confirm_hero_test(
    client: AdbClient,
    vision: Vision,
    save_dir: Path,
    result: Dict[str, Any],
) -> bool:
    """Закрывает предбоевой диалог «Укрепление героизма» зелёной кнопкой.

    После первого нажатия «Испытание» игра предлагает сначала усилить
    героя. Зелёная кнопка «Испытание» в этом диалоге продолжает бой;
    синюю «Укрепление героизма» и нижнюю неактивную кнопку не трогаем.
    """
    frame = client.get_screenshot(fresh=True)
    if frame is None:
        return False
    # OCR здесь только для диагностики: живой OCR сильно искажает
    # «увеличить силу»/«Укрепление героизма», поэтому решение принимает
    # шаблон зелёной кнопки, ограниченный областью диалога.
    ocr = read_text(frame, scale=0.5, min_confidence=0.30)
    result["hero_test_ocr"] = ocr.text[:1200]
    height, width = frame.shape[:2]
    match = vision.find_template(
        frame,
        "btn_hero_test_confirm",
        threshold=0.95,
        region=(
            int(width * 0.50),
            int(height * 0.50),
            int(width * 0.45),
            int(height * 0.12),
        ),
        scale=0.5,
    )
    if match is None:
        result["hero_test_confirm"] = None
        return False
    point = {
        "x": int(match[0]),
        "y": int(match[1]),
        "confidence": round(float(match[2]), 3),
    }
    result["hero_test_confirm"] = point
    logger.info(
        f"[Combat] Подтверждаем «Испытание» в диалоге героизма "
        f"({point['x']}, {point['y']})."
    )
    client.tap(point["x"], point["y"])
    time.sleep(1.2)
    return True


def _wait_for_battle(
    client: AdbClient,
    fast: Optional[FastCapture],
    vision: Vision,
    classifier: ScreenClassifier,
    *,
    timeout_sec: float,
    save_dir: Path,
    result: Dict[str, Any],
) -> Optional[Any]:
    started = time.time()
    frames = 0
    last_frame = None
    last_pull = 0.0
    while time.time() - started < timeout_sec:
        frame = None
        if fast is not None:
            frame = fast.read(timeout_sec=0.1)
        if frame is not None:
            frames += 1
            last_frame = frame
            height, width = frame.shape[:2]
            fast_match = vision.find_template(
                frame,
                "battle_pause_fast",
                threshold=0.90,
                region=(0, 0, int(width * 0.20), int(height * 0.12)),
                scale=1.0,
            )
            if fast_match:
                result["battle_detected"] = {
                    "elapsed_sec": round(time.time() - started, 2),
                    "fast_frames": frames,
                    "fast_fps": round(fast.stats.fps, 2) if fast else 0.0,
                    "confidence": round(float(fast_match[2]), 3),
                    "evidence": "battle_pause_fast",
                }
                cv2.imwrite(str(save_dir / "stage7_battle_fast.png"), frame)
                return last_frame
        else:
            time.sleep(0.2)

        # Резервный путь: раз в 2 секунды проверяем полный кадр. Это
        # страхует от того, что быстрый шаблон не совпал из-за сжатия.
        if time.time() - last_pull >= 2.0:
            last_pull = time.time()
            pull = client.get_screenshot(fresh=True)
            if pull is not None:
                classification = classifier.classify(pull)
                if classification.state == GameState.TUTORIAL_BATTLE:
                    result["battle_detected"] = {
                        "elapsed_sec": round(time.time() - started, 2),
                        "fast_frames": frames,
                        "fast_fps": round(fast.stats.fps, 2) if fast else 0.0,
                        "confidence": round(classification.confidence, 3),
                        "evidence": classification.evidence,
                        "source": "pull_fallback",
                    }
                    cv2.imwrite(str(save_dir / "stage7_battle_pull.png"), pull)
                    return pull
    result["battle_detected"] = None
    return None


def _observe_battle(
    client: AdbClient,
    fast: Optional[FastCapture],
    save_dir: Path,
    result: Dict[str, Any],
    seconds: float,
) -> None:
    """Смотрит бой без управления: сохраняет несколько кадров и выходит.

    Это диагностический режим для калибровки HUD. Никаких нажатий по
    джойстику/навыкам здесь нет; выход выполняет вызывающий код.
    """
    started = time.time()
    saved: list[tuple[str, float]] = []
    last_frame = None
    while time.time() - started < seconds:
        frame = fast.read(timeout_sec=0.8) if fast is not None else None
        if frame is None:
            time.sleep(0.2)
            continue
        last_frame = frame
        elapsed = time.time() - started
        if len(saved) < 6 and (not saved or elapsed - saved[-1][1] >= 0.8):
            path = save_dir / f"battle_observe_{len(saved) + 1}.png"
            cv2.imwrite(str(path), frame)
            saved.append((str(path), elapsed))
    if last_frame is not None:
        cv2.imwrite(str(save_dir / "battle_observe_last.png"), last_frame)
    full = client.get_screenshot(fresh=True)
    if full is not None:
        cv2.imwrite(str(save_dir / "battle_observe_full.png"), full)
    result["observe"] = {
        "requested_sec": seconds,
        "saved": [
            {"path": path, "at_sec": round(at_sec, 2)} for path, at_sec in saved
        ],
        "fast_frames": fast.stats.frames if fast is not None else 0,
    }
    logger.info(
        f"[Combat] Наблюдение {seconds:.1f}с без управления; "
        f"сохранено {len(saved)} быстрых кадров."
    )


def _move_test(client: AdbClient, save_dir: Path, result: Dict[str, Any]) -> None:
    """Один короткий свайп джойстика вверх, без навыков и атаки.

    Нужен только для калибровки: сравнить кадры «до/после» и понять,
    двигается ли герой. Координаты — центр нижней круглой площадки.
    """
    before = client.get_screenshot(fresh=True)
    if before is None:
        result["move_test"] = {"error": "no_before_frame"}
        return
    before_path = save_dir / "battle_move_before.png"
    cv2.imwrite(str(before_path), before)

    start = (540, 2010)
    end = (540, 1800)
    client.swipe(start[0], start[1], end[0], end[1], duration_ms=700)
    time.sleep(0.5)

    after = client.get_screenshot(fresh=True)
    if after is None:
        result["move_test"] = {"error": "no_after_frame"}
        return
    after_path = save_dir / "battle_move_after.png"
    cv2.imwrite(str(after_path), after)
    result["move_test"] = {
        "start": list(start),
        "end": list(end),
        "duration_ms": 700,
        "before_saved_to": str(before_path),
        "after_saved_to": str(after_path),
    }
    logger.info(
        f"[Combat] Move-test: свайп {start} -> {end}; кадры до/после сохранены."
    )


def _pause_and_try_exit(
    client: AdbClient,
    vision: Vision,
    classifier: ScreenClassifier,
    save_dir: Path,
    result: Dict[str, Any],
    *,
    attempt: int,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"attempt": attempt}
    # Heads-up уведомление может лежать поверх кнопки паузы: шаблон
    # «battle_pause» при этом всё ещё находится, но тап уходит в системную
    # плашку. Всегда убираем плашку перед паузой.
    client.dismiss_heads_up()
    time.sleep(0.5)
    full = client.get_screenshot(fresh=True)
    if full is None:
        info["error"] = "no_frame"
        return info
    # Оверлей повышения уровня может появиться прямо перед паузой.
    # Это не конец боя: подтверждаем навык и продолжаем искать паузу.
    current = classifier.classify(full)
    if current.state == GameState.SKILL_SELECTION:
        info["skill_selection_before_pause"] = True
        if not _handle_skill_selection(client, vision, classifier, result):
            info["error"] = "skill_selection_unhandled"
            info["action"] = "skill_selection_unhandled"
            return info
        full = client.get_screenshot(fresh=True)
        if full is None:
            info["error"] = "no_frame_after_skill"
            return info
    height, width = full.shape[:2]
    pause = _match_center(
        vision,
        full,
        PAUSE_TEMPLATE,
        threshold=0.85,
        region=(0, 0, int(width * 0.20), int(height * 0.12)),
    )
    if pause is None:
        info["error"] = "pause_not_found"
        return info
    info["pause_button"] = pause
    client.tap(pause["x"], pause["y"])
    time.sleep(1.5)

    pause_frame = client.get_screenshot(fresh=True)
    if pause_frame is None:
        info["error"] = "no_pause_frame"
        return info
    pause_path = save_dir / f"stage7_pause_menu_{attempt}.png"
    cv2.imwrite(str(pause_path), pause_frame)
    info["saved_to"] = str(pause_path)
    pause_ocr = read_text(pause_frame, scale=0.5, min_confidence=0.30)
    info["ocr"] = _ocr_payload(pause_ocr)
    logger.info(f"[Combat] Пауза OCR: {pause_ocr.text[:400]}")

    # Проверенный шаблон кнопки «Выйти» в нижней строке паузы. OCR здесь
    # опасен: «Выжить в течение …» легко распознаётся как «Выйти».
    height, width = pause_frame.shape[:2]
    exit_match = vision.find_template(
        pause_frame,
        "btn_battle_exit",
        threshold=0.90,
        region=(
            int(width * 0.05),
            int(height * 0.55),
            int(width * 0.30),
            int(height * 0.14),
        ),
        scale=0.5,
    )
    if exit_match is None:
        # Пауза не открылась (например, тап съела heads-up плашка).
        # Back здесь опасен: бой/оверлей навыка не закрываются им, а
        # система может выйти из игры в лаунчер. Поэтому только
        # сообщаем наверх, что меню паузы не открылось.
        info["exit_button"] = None
        info["action"] = "pause_menu_not_open"
        info["error"] = "exit_button_not_found"
        return info

    point = {
        "x": int(exit_match[0]),
        "y": int(exit_match[1]),
        "confidence": round(float(exit_match[2]), 3),
    }
    info["exit_button"] = point
    logger.info(
        f"[Combat] Нажимаем «Выйти» в паузе ({point['x']}, {point['y']})."
    )
    client.tap(point["x"], point["y"])
    time.sleep(1.5)

    after_frame = client.get_screenshot(fresh=True)
    if after_frame is None:
        info["error"] = "no_after_exit_frame"
        return info
    after_path = save_dir / f"stage7_after_exit_{attempt}.png"
    cv2.imwrite(str(after_path), after_frame)
    info["after_exit_saved_to"] = str(after_path)
    after_state = classifier.classify(after_frame)
    info["after_exit_state"] = after_state.state

    if after_state.state in (
        GameState.CITY_VIEW,
        GameState.CAMPAIGN_MAP,
        GameState.STAGE_BRIEFING,
        GameState.VICTORY_SCREEN,
    ):
        info["exit_confirmed"] = True
        info["action"] = "exit"
        return info

    if after_state.state == GameState.TUTORIAL_BATTLE:
        info["exit_confirmed"] = False
        info["action"] = "still_in_battle"
        return info

    # Неизвестный/подтверждающий экран: не нажимаем случайные кнопки.
    info["exit_confirmed"] = False
    info["action"] = "unexpected_after_exit"
    info["after_exit_evidence"] = after_state.evidence
    return info


BACK_SAFE_STATES = frozenset(
    {
        GameState.CAMPAIGN_MAP,
        GameState.STAGE_BRIEFING,
        GameState.ROYAL_ROAD,
        GameState.QUESTS_MENU,
        GameState.ALLIANCE_MENU,
        GameState.BUILDING_MENU,
        GameState.BUILDING_UPGRADE,
        GameState.POPUP_OFFER,
        GameState.POPUP_GENERIC,
    }
)


def _handle_skill_selection(
    client: AdbClient,
    vision: Vision,
    classifier: ScreenClassifier,
    result: Dict[str, Any],
    *,
    timeout_sec: float = 7.0,
) -> bool:
    """Подтверждает выбор навыка, появившийся поверх боя.

    Это не конец боя, а оверлей повышения уровня. Сначала ищем зелёную
    кнопку «Подтвердить». Если её нет, экран открылся без выбранной
    карточки: выбираем первую карточку (это бесплатный навык), после
    чего кнопка «Подтвердить» появляется. Back не нажимаем: он не
    закрывает оверлей и может вывести игру в лаунчер.
    """
    started = time.time()
    card_tapped = False
    while time.time() - started < timeout_sec:
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            time.sleep(0.4)
            continue
        state = classifier.classify(frame)
        if state.state != GameState.SKILL_SELECTION:
            result.setdefault("skill_selection_events", []).append(
                {"action": "resolved", "state": state.state}
            )
            return True
        height, width = frame.shape[:2]
        match = vision.find_template(
            frame,
            "btn_skill_confirm",
            threshold=0.90,
            region=(
                int(width * 0.25),
                int(height * 0.65),
                int(width * 0.55),
                int(height * 0.20),
            ),
            scale=0.5,
        )
        if match is not None:
            point = {
                "x": int(match[0]),
                "y": int(match[1]),
                "confidence": round(float(match[2]), 3),
            }
            result.setdefault("skill_selection_events", []).append(
                {"action": "confirm", "button": point}
            )
            logger.info(
                f"[Combat] Подтверждаем навык в оверлее "
                f"({point['x']}, {point['y']})."
            )
            client.tap(point["x"], point["y"])
            time.sleep(1.5)
            return True
        if not card_tapped:
            # Экран без выбранной карточки: кнопка подтверждения ещё не
            # появилась. Выбираем первую карточку (центр ~0.17W, 0.45H).
            # Это бесплатный выбор навыка; карточки покупки здесь нет.
            card = {
                "x": int(width * 0.17),
                "y": int(height * 0.45),
            }
            result.setdefault("skill_selection_events", []).append(
                {"action": "select_first_card", "point": card}
            )
            logger.info(
                f"[Combat] Выбираем первую карточку навыка "
                f"({card['x']}, {card['y']})."
            )
            client.tap(card["x"], card["y"])
            card_tapped = True
            time.sleep(0.8)
            continue
        time.sleep(0.4)
    result.setdefault("skill_selection_events", []).append(
        {"action": "unhandled_no_confirm_button"}
    )
    logger.warning(
        "Оверлей выбора навыка не удалось подтвердить; "
        "не жмём Back и не перебираем карточки."
    )
    return False


def _relaunch_game(
    client: AdbClient,
    config: Dict[str, Any],
    classifier: ScreenClassifier,
    *,
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """Аварийное восстановление: force-stop + запуск игры до города."""
    package = config["game"]["package"]
    activity = config["game"]["activity"]
    stopped = client.force_stop(package)
    time.sleep(1.0)
    launched = client.launch_game(package, activity)
    started = time.time()
    states: List[str] = []
    while time.time() - started < timeout_sec:
        time.sleep(2.0)
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            continue
        state = classifier.classify(frame).state
        states.append(state)
        if state == GameState.CITY_VIEW:
            break
    return {
        "stopped": stopped,
        "launched": launched,
        "states": states,
        "ok": bool(states and states[-1] == GameState.CITY_VIEW),
    }


def _wait_for_battle_end(
    client: AdbClient,
    fast: Optional[FastCapture],
    vision: Vision,
    classifier: ScreenClassifier,
    *,
    max_wait_sec: float,
    save_dir: Path,
    result: Dict[str, Any],
) -> str:
    started = time.time()
    last_log = 0.0
    while time.time() - started < max_wait_sec:
        frame = fast.read(timeout_sec=0.9) if fast is not None else None
        if frame is not None:
            height, width = frame.shape[:2]
            fast_match = vision.find_template(
                frame,
                "battle_pause_fast",
                threshold=0.90,
                region=(0, 0, int(width * 0.20), int(height * 0.12)),
                scale=1.0,
            )
            if fast_match:
                if time.time() - last_log >= 10.0:
                    logger.info(
                        f"[Combat] Бой ещё идёт, прошло {time.time() - started:.0f}с."
                    )
                    last_log = time.time()
                time.sleep(0.2)
                continue
            # Быстрый шаблон не нашёлся — проверяем полный кадр, чтобы
            # отличить конец боя от кратковременного перехода/оверлея.
            frame = client.get_screenshot(fresh=True)
        else:
            frame = client.get_screenshot(fresh=True)
        if frame is None:
            time.sleep(0.5)
            continue
        classification = classifier.classify(frame)
        if classification.state == GameState.SKILL_SELECTION:
            # Это оверлей повышения уровня поверх боя, а не его конец.
            result["skill_selection_count"] = (
                result.get("skill_selection_count", 0) + 1
            )
            if (
                result["skill_selection_count"] <= 4
                and _handle_skill_selection(
                    client, vision, classifier, result
                )
            ):
                last_log = 0.0
                continue
            cv2.imwrite(str(save_dir / "stage7_battle_end.png"), frame)
            result["battle_end"] = {
                "state": classification.state,
                "confidence": round(classification.confidence, 3),
                "evidence": classification.evidence,
                "elapsed_sec": round(time.time() - started, 2),
                "unhandled": True,
            }
            return "skill_selection_unhandled"
        if classification.state != GameState.TUTORIAL_BATTLE:
            cv2.imwrite(str(save_dir / "stage7_battle_end.png"), frame)
            result["battle_end"] = {
                "state": classification.state,
                "confidence": round(classification.confidence, 3),
                "evidence": classification.evidence,
                "elapsed_sec": round(time.time() - started, 2),
            }
            return classification.state
        if time.time() - last_log >= 10.0:
            logger.info(f"[Combat] Бой ещё идёт, прошло {time.time() - started:.0f}s.")
            last_log = time.time()
        time.sleep(0.2)
    result["battle_end"] = {"state": "timeout", "elapsed_sec": max_wait_sec}
    return "timeout"


def _dismiss_result(
    client: AdbClient,
    classifier: ScreenClassifier,
    save_dir: Path,
    result: Dict[str, Any],
    *,
    attempts: int = 4,
) -> None:
    """Закрывает экран победы/поражения только безопасными кнопками."""
    actions: List[Dict[str, Any]] = []
    for index in range(attempts):
        frame = client.get_screenshot(fresh=True)
        if frame is None:
            break
        classification = classifier.classify(frame)
        if classification.state == GameState.CITY_VIEW:
            break
        if classification.state in (
            GameState.TUTORIAL_BATTLE,
            GameState.SKILL_SELECTION,
        ):
            actions.append(
                {
                    "step": index,
                    "action": "stop_in_battle",
                    "state": classification.state,
                }
            )
            break
        ocr = read_text(frame, scale=0.5, min_confidence=0.30)
        if looks_like_payment(frame):
            actions.append({"step": index, "action": "payment_guard"})
            break
        item = _find_text_item(ocr, SAFE_RESULT_WORDS, threshold=0.78)
        if item is not None:
            actions.append(
                {
                    "step": index,
                    "action": "tap",
                    "text": item.text,
                    "center": [int(item.x), int(item.y)],
                }
            )
            client.tap(item.x, item.y)
            time.sleep(1.6)
            continue
        if classification.state in BACK_SAFE_STATES:
            actions.append({"step": index, "action": "back"})
            client.press_back()
            time.sleep(1.4)
            continue
        actions.append(
            {
                "step": index,
                "action": "stop_unsafe_back",
                "state": classification.state,
            }
        )
        break
    result["dismiss_actions"] = actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Контролируемый бой этапа 7")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-dir", default="logs/inspect")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Подтвердить реальный вход в бой (без флага бой не начинается)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только открыть брифинг этапа 7 и вернуться, без нажатия «Испытание»",
    )
    parser.add_argument("--detect-timeout", type=float, default=15.0)
    parser.add_argument("--max-wait", type=float, default=140.0)
    parser.add_argument(
        "--observe-sec",
        type=float,
        default=0.0,
        help="Смотреть бой N секунд без управления (диагностика HUD)",
    )
    parser.add_argument(
        "--move-test",
        action="store_true",
        help="Один короткий свайп джойстика вверх для калибровки, затем выход",
    )
    args = parser.parse_args()

    if args.move_test and args.dry_run:
        print("Отказ: --move-test несовместим с --dry-run.")
        return 2

    if not args.confirm and not args.dry_run:
        print("Отказ: для входа в бой нужен флаг --confirm.")
        return 2

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
        return 3

    result: Dict[str, Any] = {"confirm": bool(args.confirm) and not args.dry_run}
    fast: Optional[FastCapture] = None
    if not args.dry_run:
        # Запускаем поток заранее, пока открываем карту/брифинг: к моменту
        # боя screenrecord+ffmpeg уже прогреты и отдают кадры без задержки
        # на старт. Порт берём из client: config может быть устаревшим.
        device = config["device"]
        serial = f"{args.ip or client.ip}:{args.port or client.port}"
        fast = FastCapture(
            adb_path=device["adb_path"],
            serial=serial,
            width=540,
            height=1218,
            record_time_limit=int(max(90, args.max_wait + 45)),
        )
        result["fast_capture_started"] = fast.start()
        time.sleep(0.4)
    try:
        if not _open_stage7(client, vision, classifier, save_dir, result):
            logger.error(f"Не удалось открыть этап 7: {result.get('error')}")
            ok, states = _return_to_city(client, classifier)
            result["returned_to_city"] = ok
            result["return_states"] = states
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 4

        if args.dry_run:
            result["dry_run"] = True
            logger.info("Dry-run: брифинг открыт, «Испытание» не нажимаем.")
            ok, states = _return_to_city(client, classifier)
            result["returned_to_city"] = ok
            result["return_states"] = states
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        challenge = result.get("challenge") or {}
        # Убираем возможную heads-up плашку до входа в бой: она может
        # перекрыть кнопку паузы и украсть первый тап.
        client.dismiss_heads_up()
        time.sleep(0.5)
        # Сливаем устаревшие кадры брифинга, чтобы поток после нажатия
        # «Испытание» начинался с боевых кадров, а не со старых.
        if fast is not None:
            fast.drain(timeout_sec=0.2)
        logger.info(
            f"[Combat] Нажимаем «Испытание» ({challenge.get('x')}, {challenge.get('y')})."
        )
        client.tap(challenge["x"], challenge["y"])
        time.sleep(1.4)
        result["hero_test_confirmed"] = _confirm_hero_test(
            client, vision, save_dir, result
        )

        battle = _wait_for_battle(
            client,
            fast,
            vision,
            classifier,
            timeout_sec=args.detect_timeout,
            save_dir=save_dir,
            result=result,
        )
        if battle is None:
            logger.warning("Боевой экран не появился за отведённое время.")
            frame = client.get_screenshot(fresh=True)
            if frame is not None:
                cv2.imwrite(str(save_dir / "stage7_after_challenge.png"), frame)
                state = classifier.classify(frame)
                result["after_challenge"] = {
                    "state": state.state,
                    "confidence": round(state.confidence, 3),
                    "evidence": state.evidence,
                }
                ocr = read_text(frame, scale=0.5, min_confidence=0.30)
                result["after_challenge_ocr"] = ocr.text[:1500]
        else:
            full = client.get_screenshot(fresh=True)
            if full is not None:
                cv2.imwrite(str(save_dir / "stage7_battle_full.png"), full)
                full_state = classifier.classify(full)
                result["battle_full"] = {
                    "state": full_state.state,
                    "confidence": round(full_state.confidence, 3),
                    "evidence": full_state.evidence,
                }
                full_ocr = read_text(full, scale=0.5, min_confidence=0.30)
                result["battle_ocr"] = _ocr_payload(full_ocr)
                logger.info(f"[Combat] Боевой OCR: {full_ocr.text[:400]}")

            if args.observe_sec > 0:
                _observe_battle(client, fast, save_dir, result, args.observe_sec)
            if args.move_test:
                _move_test(client, save_dir, result)

            pause_1 = _pause_and_try_exit(
                client, vision, classifier, save_dir, result, attempt=1
            )
            result["pause_1"] = pause_1
            paused = bool(pause_1.get("exit_confirmed"))
            if not paused:
                # Одна быстрая повторная попытка: меню паузы иногда не
                # успевает открыться после первого тапа.
                time.sleep(1.0)
                pause_retry = _pause_and_try_exit(
                    client, vision, classifier, save_dir, result, attempt=2
                )
                result["pause_1_retry"] = pause_retry
                if pause_retry.get("exit_confirmed"):
                    pause_1 = pause_retry
                    paused = True

            if paused:
                end_state = (
                    pause_1.get("after_exit_state") or GameState.CAMPAIGN_MAP
                )
            else:
                # Если бой уже закончился сам (победа/поражение/карта),
                # обрабатываем этот экран. Если всё ещё бой или оверлей
                # навыка — не ждём 140 секунд и не жмём Back, а идём на
                # безопасный перезапуск.
                frame = client.get_screenshot(fresh=True)
                current_state = (
                    classifier.classify(frame).state
                    if frame is not None
                    else GameState.UNKNOWN
                )
                if current_state not in (
                    GameState.TUTORIAL_BATTLE,
                    GameState.SKILL_SELECTION,
                    GameState.UNKNOWN,
                ):
                    end_state = current_state
                else:
                    end_state = "pause_failed"
                    result["combat_unresolved"] = True
            if end_state not in (
                GameState.TUTORIAL_BATTLE,
                GameState.SKILL_SELECTION,
                "skill_selection_unhandled",
                "timeout",
            ):
                _dismiss_result(client, classifier, save_dir, result)
            else:
                result["combat_unresolved"] = True
                logger.warning(
                    "Бой/выбор навыка всё ещё не завершён; не нажимаем "
                    "случайные кнопки и не жмём Back. Нужен ручной контроль "
                    "или перезапуск зонда."
                )
    finally:
        if fast is not None:
            fast.stop()
            result["fast_stats"] = {
                "frames": fast.stats.frames,
                "fps": round(fast.stats.fps, 2),
            }

    ok, states = _return_to_city(client, classifier)
    if not ok:
        last_state = states[-1] if states else GameState.UNKNOWN
        package = config["game"]["package"]
        if last_state in (
            GameState.TUTORIAL_BATTLE,
            GameState.SKILL_SELECTION,
        ) or not client.is_game_foreground(package):
            # Неизвестный экран при игре в foreground — это, скорее всего,
            # ежедневная награда/событие. force-stop там не нужен и может
            # снова привести к череде попапов; зонд просто остановится.
            logger.warning(
                "Штатный возврат не удался; выполняем force-stop + запуск игры."
            )
            result["recovery"] = _relaunch_game(client, config, classifier)
            ok = bool(result["recovery"].get("ok"))
            states = states + list(result["recovery"].get("states", []))
    result["returned_to_city"] = ok
    result["return_states"] = states
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
