"""
Зонд калибровочного обхода 15 зданий квартала (ADR-022, Этап 8.2 Дорожной карты).

Безопасный сценарий:
1. Для каждого из 15 зданий D1..D15 (из memory_bank/city_district.md):
   - Координаты берутся из канонической эталонной системы (ref_point)
   - Переносятся через DistrictAligner на текущий живой кадр
   - Тап по подошве здания
   - Ожидание радиального меню (золотые молоты / шеврон)
   - Тап по «Улучшить» в радиальном меню
   - Чтение заголовка (авторитетное имя здания), уровней LV.N -> LV.M
   - Проверка наличия «Перейти» (упор в уровень замка -> capped)
   - КРИТИЧЕСКИ ВАЖНО: синяя и зелёная кнопки НЕ нажимаются!
   - Закрытие панели стрелкой назад (btn_back_arrow 111, 169)
2. Синхронизация паспортов в memory_bank/building_spots.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger
from rapidocr_onnxruntime import RapidOCR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient
from src.district_aligner import DistrictAligner
from src.ocr import normalize_text
from src.perception import ScreenClassifier
from src.vision import Vision

# 15 зданий квартала с каноническими REF-координатами
DISTRICT_TARGETS = [
    {"id": "D1",  "hint": "camp",    "ref": [552, 931],  "desc": "бело-красный форт"},
    {"id": "D2",  "hint": "farm",    "ref": [399, 985],  "desc": "ферма ветряк"},
    {"id": "D3",  "hint": "farm",    "ref": [541, 1030], "desc": "ферма"},
    {"id": "D4",  "hint": "farm",    "ref": [341, 1164], "desc": "ферма"},
    {"id": "D5",  "hint": "farm",    "ref": [494, 1223], "desc": "ферма"},
    {"id": "D6",  "hint": "sawmill", "ref": [719, 1125], "desc": "лесопилка"},
    {"id": "D7",  "hint": "tent",    "ref": [380, 1381], "desc": "палатки"},
    {"id": "D8",  "hint": "tent",    "ref": [240, 1469], "desc": "палатки край"},
    {"id": "D9",  "hint": "tent",    "ref": [562, 1430], "desc": "палатки кольцо"},
    {"id": "D10", "hint": "camp",    "ref": [720, 1273], "desc": "красное здание"},
    {"id": "D11", "hint": "farm",    "ref": [756, 1440], "desc": "ветряк у воды"},
    {"id": "D12", "hint": "sawmill", "ref": [276, 1617], "desc": "лесопилка"},
    {"id": "D13", "hint": "tent",    "ref": [441, 1656], "desc": "палатки"},
    {"id": "D14", "hint": "camp",    "ref": [625, 1646], "desc": "форт кольцо"},
    {"id": "D15", "hint": "camp",    "ref": [684, 1763], "desc": "форт"},
]

RADIAL_TEMPLATES = [
    ("btn_radial_upgrade", 0.80),
    ("btn_radial_upgrade_gold", 0.80),
    ("btn_radial_upgrade_wide", 0.82),
    ("btn_radial_upgrade_medic", 0.82),
]

NAME_TRANSLATIONS = {
    "ферма": "farm",
    "лесопилка": "sawmill",
    "палатка": "tent",
    "учебный": "camp",
    "лагерь": "camp",
    "казарма": "barracks",
    "замок": "castle",
    "стена": "wall",
    "институт": "academy",
    "кузница": "forge",
    "госпиталь": "hospital",
}


def canonical_type(raw_title: str, hint: str) -> str:
    norm = raw_title.lower()
    for ru, en in NAME_TRANSLATIONS.items():
        if ru in norm:
            return en
    return hint


def find_radial(vision: Vision, frame: np.ndarray, center: Tuple[int, int]) -> Optional[Tuple[int, int, float]]:
    cx, cy = center
    h, w = frame.shape[:2]
    # Ограниченная область вокруг здания
    x1, x2 = max(0, cx - int(0.22 * w)), min(w, cx + int(0.22 * w))
    y1, y2 = max(0, cy - int(0.28 * h)), min(h, cy + int(0.08 * h))
    region = (x1, y1, x2 - x1, y2 - y1)

    best = None
    for tmpl_name, thresh in RADIAL_TEMPLATES:
        m = vision.find_template(frame, tmpl_name, threshold=thresh, region=region)
        if m:
            if best is None or m[2] > best[2]:
                best = m
    return best


def parse_panel_ocr(ocr: RapidOCR, frame: np.ndarray) -> dict:
    h, w = frame.shape[:2]
    # Заголовок
    crop_title = frame[int(0.02 * h) : int(0.12 * h), int(0.25 * w) : int(0.75 * w)]
    res_t, _ = ocr(crop_title)
    title = " ".join([r[1] for r in res_t]) if res_t else ""

    # Блок уровней и требований
    crop_req = frame[int(0.55 * h) : int(0.85 * h), int(0.05 * w) : int(0.95 * w)]
    res_r, _ = ocr(crop_req)
    req_text = " ".join([r[1] for r in res_r]) if res_r else ""

    current_lvl = 1
    target_lvl = 2
    match_lvl = re.search(r"(\d+)\s*(?:yp|ур|lv)?\s*->\s*(\d+)", title + " " + req_text, re.IGNORECASE)
    if match_lvl:
        try:
            current_lvl = int(match_lvl.group(1))
            target_lvl = int(match_lvl.group(2))
        except ValueError:
            pass
    elif "5" in title or "5" in req_text:
        # Если здание 5 уровня
        current_lvl = 5
        target_lvl = 6

    return {
        "title": title,
        "req_text": req_text,
        "level": current_lvl,
        "target_level": target_lvl,
    }


def calibrate_district():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    dev = cfg["device"]
    client = AdbClient(dev["adb_path"], dev["ip"], dev["port"], timeout=10)
    if not client.connect(allow_discovery=True):
        logger.error("Не удалось подключиться к ADB")
        return False

    client.wake_screen()
    vision = Vision(templates_dir="assets")
    aligner = DistrictAligner()
    ocr = RapidOCR()

    knowledge_path = Path("memory_bank/building_spots.json")
    spots_data = {}
    if knowledge_path.exists():
        try:
            spots_data = json.loads(knowledge_path.read_text(encoding="utf-8"))
        except Exception:
            spots_data = {}

    logger.info("=" * 65)
    logger.info("  🏘️ КАЛИБРОВОЧНЫЙ ОБХОД 15 ЗДАНИЙ КВАРТАЛА (ADR-022)")
    logger.info("=" * 65)

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        logger.error("Нет кадра")
        return False

    H = aligner.get_homography(frame)
    if H is None:
        logger.error("❌ SIFT-гомография не сошлась с эталоном! Калибровка отменена во избежание слепых кликов.")
        return False

    logger.success("✅ SIFT-гомография квартала подтверждена! Начинаем безопасный обход...")

    results = []
    for item in DISTRICT_TARGETS:
        sid = item["id"]
        ref_pt = item["ref"]
        live_pt = aligner.transform_point(ref_pt, frame)

        if live_pt is None:
            logger.warning(f"[{sid}] Точка вне экрана — пропускаем.")
            continue

        lx, ly = live_pt
        logger.info(f"\n--- [{sid}] {item['desc']} ---")
        logger.info(f"Тап в подошву: REF={ref_pt} -> LIVE=({lx}, {ly})...")
        client.tap(lx, ly)
        time.sleep(1.8)

        # Ждём радиальное меню
        frame = client.get_screenshot(fresh=True)
        radial = find_radial(vision, frame, (lx, ly))

        if radial is None:
            logger.warning(f"[{sid}] Радиальное меню не открылось. Проверяем повторно с небольшой задержкой...")
            time.sleep(1.0)
            frame = client.get_screenshot(fresh=True)
            radial = find_radial(vision, frame, (lx, ly))

        if radial is None:
            logger.warning(f"[{sid}] Радиальное меню не найдено (возможно, улучшается или закрыто). Снимаем выбор...")
            client.tap(540, 1200)  # Тап в безопасное пустое место
            time.sleep(0.5)
            continue

        rx, ry, rconf = radial
        logger.success(f"[{sid}] 🟢 Радиальное меню найдено: ({rx}, {ry}), conf={rconf:.2f}. Открываем панель...")
        client.tap(rx, ry)
        time.sleep(2.5)

        # Читаем панель
        panel = client.get_screenshot(fresh=True)
        # Проверяем, открылась ли панель
        has_req = vision.find_template(panel, "title_upgrade_required", threshold=0.75)
        has_confirm = vision.find_template(panel, "btn_upgrade_confirm", threshold=0.75)

        if not has_req and not has_confirm:
            logger.warning(f"[{sid}] Панель не открылась — жмём Назад...")
            client.press_back()
            time.sleep(1.0)
            continue

        # Читаем OCR панели
        ctx = parse_panel_ocr(ocr, panel)
        b_type = canonical_type(ctx["title"], item["hint"])

        # Проверка упора в замок (кнопка 'Перейти' btn_go_castle)
        has_go = vision.find_template(panel, "btn_go_castle", threshold=0.75)
        is_capped = bool(has_go) or "замок" in ctx["req_text"].lower() or ctx["level"] >= 5

        status = "capped" if is_capped else "available"

        logger.success(
            f"[{sid}] ✅ Панель: '{ctx['title']}' -> тип='{b_type}', ур.={ctx['level']}->{ctx['target_level']}, статус={status}"
        )

        # Сохраняем в паспорт
        passport_key = f"pt:{b_type}:{ref_pt[0]},{ref_pt[1]}"
        passport_entry = {
            "key": passport_key,
            "district_id": sid,
            "name": b_type,
            "title_raw": ctx["title"],
            "point": ref_pt,
            "level": ctx["level"],
            "target_level": ctx["target_level"],
            "castle_level": 5,
            "status": status,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        spots_data[passport_key] = passport_entry
        results.append(passport_entry)

        # БЕЗОПАСНЫЙ ВЫХОД: жмём только стрелку (111, 169)
        arrow = vision.find_template(panel, "btn_back_arrow", threshold=0.85)
        if arrow:
            client.tap(arrow[0], arrow[1])
        else:
            client.tap(111, 169)
        time.sleep(1.5)

    # Сохраняем обновлённую базу знаний
    knowledge_path.write_text(json.dumps(spots_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success("=" * 65)
    logger.success(f"🎉 КАЛИБРОВКА ЗАВЕРШЕНА! Обновлено {len(results)}/15 паспортов.")
    logger.success(f"База знаний сохранена в {knowledge_path}")
    logger.success("=" * 65)
    return True


if __name__ == "__main__":
    calibrate_district()
