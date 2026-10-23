"""
Задача постройки второстепенных зданий в городе (CityBuildTask).

Проверенный на живом устройстве сценарий (2026-10-23):
1. Пустая ячейка — квадрат из камней на земле справа-снизу от замка.
2. Тап по ячейке открывает экран постройки: красный заголовок с типом,
   3D-макет, две строки требований и две кнопки внизу.
3. Тип выбирается горизонтальной каруселью: свайп-флик по строке подписей
   (≈420 px за 220 мс) перелистывает на один пункт. Медленный свайп игра
   игнорирует, а слишком быстрый может перескочить пункт, поэтому после
   каждого флика заголовок перечитывается OCR и направление при
   необходимости меняется.
4. Кнопка «Построить» — СИНЯЯ справа. Слева — зелёная кнопка с гемом
   «Получить сейчас / Бесплатно»: её задача не нажимает НИКОГДА.
5. Перед нажатием проверяются: обе строки требований с зелёными галочками
   и активность синей кнопки по цвету (насыщенность и перевес синего).
6. Постройка мгновенная (таймер 00:00), экран возвращается в город.

За один вызов `execute` строится ровно одно здание: движок между тактами
даёт игре доиграть анимации, а другие задачи успевают проверить попапы.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.ocr import fuzzy_contains, normalize_text, parse_compact_number, read_text
from src.tasks.base_task import BaseTask
from src.vision import Vision

# (x, y) в пикселях кадра 1080x2436.
Point = Tuple[int, int]


class CityBuildTask(BaseTask):
    BUILD_TEMPLATE = "btn_build"            # синяя «Построить» — безопасная
    BUILD_FREE_TEMPLATE = "btn_build_free"  # зелёная с гемом — НЕ НАЖИМАТЬ
    PLOT_TEMPLATE = "plot_empty_live"
    NAV_TEMPLATE = "btn_nav_quests"

    # Порядок карусели на живом экране 2026-10-23.
    TYPE_ORDER: Tuple[str, ...] = ("farm", "sawmill", "camp", "tent", "quarry", "iron")
    TYPE_TITLES = {
        "farm": "Ферма",
        "sawmill": "Лесопилка",
        "camp": "Учебный лагерь",
        "tent": "Медицинская палатка",
        "quarry": "Каменная шахта",
        "iron": "Железный рудник",
    }
    # OCR читает кириллицу латинскими двойниками («Учебный лагерь» ->
    # «yyebHbin Aarepb»), поэтому в списке есть и такие варианты.
    TYPE_ALIASES = {
        "farm": ("ферма", "ferma", "pepma", "oepma", "epma"),
        "sawmill": ("лесопил", "lesopil", "lesop", "necon"),
        "camp": ("учебн", "лагер", "ucheb", "lager", "aarep", "arepb", "yyeb"),
        "tent": ("медиц", "палат", "medic", "palat", "naatk"),
        "quarry": ("камен", "шахт", "kamen", "shaht", "waxta"),
        "iron": ("желез", "рудник", "zhelez", "rudnik", "pyah", "xere3"),
    }
    # Цель миссии: все четыре второстепенных типа, по кругу.
    BUILD_CYCLE: Tuple[str, ...] = ("farm", "sawmill", "camp", "tent")

    TITLE_REL = (0.30, 0.03, 0.72, 0.10)
    # Живой замер 2026-10-23: флик по этой строке (даже если он попадает
    # на 3D-макет) перелистывает карусель ровно на один пункт.
    STRIP_Y_REL = 0.265
    REQUIREMENTS_REL = (0.05, 0.64, 0.88, 0.065)
    CHECK_ZONE_REL = (0.80, 0.63, 0.92, 0.71)
    CHECK_MIN_AREA = 150

    FLICK_DISTANCE = 420
    # 300 мс — проверенная на живом устройстве скорость; на 220 мс часть
    # свайпов игра игнорирует. Пауза после флика должна покрывать инерцию
    # карусели, иначе следующий свайп приходит в ещё движущийся список.
    FLICK_MS = 300
    TYPE_SETTLE_S = 2.2
    BUILD_WAIT_S = 7.0
    CITY_WAIT_S = 8.0
    MAX_TYPE_FLICKS = 10
    MAX_PLOT_RETRIES = 3

    # Область поиска ячеек: почти весь правый нижний квартал. Камера после
    # постройки слегка смещается, поэтому фиксированные точки не годятся —
    # ячейки ищутся шаблоном на каждом свежем кадре.
    PLOT_REGION = (300, 850, 780, 1450)  # x, y, w, h
    PLOT_THRESHOLD = 0.40
    PLOT_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
    PLOT_MIN_DISTANCE = 70
    # Иконка демона и нижняя навигация перекрывают ячейки и не должны
    # приниматься за них: тап по ним открывает чужие окна.
    EXCLUDE_ZONES = (
        (0.90, 0.665, 1.00, 0.75),  # демон-событие справа (только сама иконка)
        (0.00, 0.93, 1.00, 1.00),   # нижняя навигация
        (0.00, 0.00, 1.00, 0.36),   # верхний HUD и замок
        (0.00, 0.00, 0.12, 1.00),   # узкие кнопки у левого края
        (0.00, 0.84, 0.66, 0.90),   # баннер «Всадник дракона»
    )
    KNOWLEDGE_PATH = Path("memory_bank/built_plots.json")
    def __init__(
        self,
        client: AdbClient,
        vision: Vision,
        interval_sec: int = 15,
        plots: Optional[Iterable] = None,
        reflection=None,
    ):
        super().__init__("CityBuild", client, vision, interval_sec=interval_sec, reflection=reflection)
        self.config_plots: List[Point] = self._parse_points(plots)
        self.built: List[dict] = self._load_knowledge()
        self.attempted: set[Point] = set()
        self.failed_plots: set[Point] = set()
        self.retry_counts: dict[Point, int] = {}
        self._cycle_index = len(self.built) % len(self.BUILD_CYCLE)
        self._exhausted = False
        self.last_build: Optional[dict] = None

    # ------------------------------------------------------------------
    # Конфигурация и база знаний
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_points(raw: Optional[Iterable]) -> List[Point]:
        points: List[Point] = []
        for item in raw or []:
            if isinstance(item, dict):
                point = item.get("point") or item.get("xy") or item.get("coords")
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and all(
                isinstance(value, (int, float)) for value in item[:2]
            ):
                point = item
            else:
                point = None
            if point and len(point) >= 2:
                points.append((int(point[0]), int(point[1])))
        return points

    @classmethod
    def _load_knowledge(cls) -> List[dict]:
        try:
            if cls.KNOWLEDGE_PATH.exists():
                data = json.loads(cls.KNOWLEDGE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
        except Exception as exc:  # noqa: BLE001 - база знаний не должна ронять задачу
            logger.warning(f"[CityBuild] Не удалось прочитать {cls.KNOWLEDGE_PATH}: {exc}")
        return []

    @classmethod
    def _save_knowledge(cls, entries: List[dict]) -> None:
        try:
            cls.KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cls.KNOWLEDGE_PATH.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[CityBuild] Не удалось сохранить {cls.KNOWLEDGE_PATH}: {exc}")

    def _built_points(self) -> set[Point]:
        return {(int(item["plot"][0]), int(item["plot"][1])) for item in self.built if item.get("plot")}

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------
    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates(
            [self.BUILD_TEMPLATE, self.BUILD_FREE_TEMPLATE, self.PLOT_TEMPLATE, self.NAV_TEMPLATE]
        ):
            return False

        screen = self.ensure_screen(screen, max_age=2.0)
        if screen is None:
            return False

        # Восстановление: экран постройки уже открыт (например, после сбоя).
        if self._is_build_screen(screen):
            logger.info("[CityBuild] Экран постройки уже открыт — завершаем начатое.")
            return self._complete_build(screen, None)

        candidates = self._find_plots(screen)
        if not candidates:
            if not self._exhausted:
                logger.info(
                    "[CityBuild] Свободных ячеек не видно. Известно построек: "
                    f"{len(self.built)}, неудачных проверок: {len(self.failed_plots)}."
                )
                self._exhausted = True
            return False

        plot = candidates[0]
        logger.info(
            f"[CityBuild] Проверяем ячейку {plot} (всего кандидатов {len(candidates)}, "
            f"построено {len(self.built)})."
        )
        self.client.tap(plot[0], plot[1])

        frame = self._wait_for_build_screen(self.BUILD_WAIT_S)
        if frame is None:
            logger.info(f"[CityBuild] Ячейка {plot} не открыла экран постройки — пропускаем.")
            self.failed_plots.add(plot)
            self._leave_foreign_screen()
            return False

        self.last_build = {"plot": list(plot)}
        return self._complete_build(frame, plot)

    # ------------------------------------------------------------------
    # Поиск свободных ячеек
    # ------------------------------------------------------------------
    def _find_plots(self, screen: np.ndarray) -> List[Point]:
        height, width = screen.shape[:2]
        built = self._built_points()
        candidates: List[Tuple[Point, float]] = []

        for point in self.config_plots:
            if self._usable(point, width, height, built):
                candidates.append((point, 1.0))

        matches = self.vision.find_all_templates_multiscale(
            screen,
            self.PLOT_TEMPLATE,
            threshold=self.PLOT_THRESHOLD,
            min_distance=self.PLOT_MIN_DISTANCE,
            region=self.PLOT_REGION,
            scales=self.PLOT_SCALES,
            blur=3,
        )
        for x, y, conf, _rel in matches:
            point = (int(x), int(y))
            if not self._usable(point, width, height, built):
                continue
            if any((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2 < 60 ** 2 for p, _ in candidates):
                continue
            candidates.append((point, conf))

        # Сначала заданные и проверенные точки, затем шаблонные совпадения.
        candidates.sort(key=lambda item: -item[1])
        return [point for point, _ in candidates]

    def _usable(self, point: Point, width: int, height: int, built: set[Point]) -> bool:
        if point in self.attempted or point in self.failed_plots or point in built:
            return False
        x, y = point
        if not (0 < x < width and 0 < y < height):
            return False
        for x1, y1, x2, y2 in self.EXCLUDE_ZONES:
            if x1 * width <= x <= x2 * width and y1 * height <= y <= y2 * height:
                return False
        # Точка рядом с уже построенной ячейкой — скорее всего та же ячейка.
        for bx, by in built:
            if (x - bx) ** 2 + (y - by) ** 2 < 70 ** 2:
                return False
        return True

    # ------------------------------------------------------------------
    # Экран постройки
    # ------------------------------------------------------------------
    def _is_build_screen(self, frame: np.ndarray) -> bool:
        return self._find_buttons(frame)[0] is not None

    def _find_buttons(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, float]], Optional[Tuple[int, int, float]]]:
        height, width = frame.shape[:2]
        blue = self.vision.find_template(
            frame,
            self.BUILD_TEMPLATE,
            threshold=0.80,
            region=(int(width * 0.55), int(height * 0.90), int(width * 0.45), int(height * 0.10)),
        )
        green = self.vision.find_template(
            frame,
            self.BUILD_FREE_TEMPLATE,
            threshold=0.80,
            region=(int(width * 0.00), int(height * 0.90), int(width * 0.45), int(height * 0.10)),
        )
        return blue, green

    def _wait_for_build_screen(self, timeout: float) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout
        while True:
            frame = self.client.get_screenshot(fresh=True)
            if frame is not None and self._is_build_screen(frame):
                return frame
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.7)

    # ------------------------------------------------------------------
    # Выбор типа и постройка
    # ------------------------------------------------------------------
    def _complete_build(self, frame: np.ndarray, plot: Optional[Point]) -> bool:
        blue, _green = self._find_buttons(frame)
        if blue is None:
            return False

        target = self._next_type()
        logger.info(f"[CityBuild] Целевой тип: {target} ({self.TYPE_TITLES.get(target, target)}).")

        frame, current = self._select_type(frame, target)
        if current != target:
            logger.warning(
                f"[CityBuild] Не удалось выбрать тип {target}: OCR дал {current!r} — отмена."
            )
            self._register_soft_failure(plot)
            self._leave_build_screen()
            return False

        blue, _green = self._find_buttons(frame)
        if blue is None:
            self._mark_failed(plot)
            self._leave_build_screen()
            return False

        ok, details = self._requirements_ok(frame)
        if not ok:
            logger.warning(f"[CityBuild] Требования не выполнены ({details}) — отмена.")
            self._register_soft_failure(plot)
            self._leave_build_screen()
            return False

        saturation, blue_red = self._blue_button_color(frame, blue)
        if saturation < 60.0 or blue_red < 10.0:
            logger.info(
                f"[CityBuild] Синяя кнопка неактивна (S={saturation:.0f}, B-R={blue_red:.0f}) — не нажимаем."
            )
            self._mark_failed(plot)
            self._leave_build_screen()
            return False

        logger.success(
            f"[CityBuild] Строим {self.TYPE_TITLES.get(target, target)} в {self.last_build['plot'] if self.last_build else '?'} "
            f"({blue[0]}, {blue[1]}), conf={blue[2]:.2f}, S={saturation:.0f}, B-R={blue_red:.0f}."
        )
        self.client.tap(int(blue[0]), int(blue[1]))

        if not self._wait_for_city(self.CITY_WAIT_S):
            self.report_failure(
                "build_not_confirmed",
                "Экран постройки не закрылся после нажатия синей кнопки",
            )
            self._mark_failed(plot)
            self._leave_build_screen()
            return False

        self._cycle_index += 1
        entry = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "plot": list(self.last_build["plot"]) if self.last_build else None,
            "type": target,
            "title": self.TYPE_TITLES.get(target, target),
            "requirements": details,
            "blue_saturation": round(saturation, 1),
            "blue_minus_red": round(blue_red, 1),
        }
        self.built.append(entry)
        self._save_knowledge(self.built)
        if entry["plot"]:
            self.attempted.add((int(entry["plot"][0]), int(entry["plot"][1])))
        self.mark_executed()
        logger.success(f"[CityBuild] Построено: {entry['title']} в ячейке {entry['plot']}.")
        return True

    def _register_soft_failure(self, plot: Optional[Point]) -> None:
        """Сбой выбора типа/требований: ячейка остаётся годной, но с лимитом попыток."""
        if plot is None:
            return
        self.retry_counts[plot] = self.retry_counts.get(plot, 0) + 1
        logger.info(
            f"[CityBuild] Ячейка {plot}: мягкий сбой {self.retry_counts[plot]}/{self.MAX_PLOT_RETRIES}."
        )
        if self.retry_counts[plot] >= self.MAX_PLOT_RETRIES:
            self._mark_failed(plot)

    def _mark_failed(self, plot: Optional[Point]) -> None:
        if plot is not None:
            self.failed_plots.add(plot)

    def _next_type(self) -> str:
        return self.BUILD_CYCLE[self._cycle_index % len(self.BUILD_CYCLE)]

    def _select_type(
        self, frame: np.ndarray, target: str
    ) -> Tuple[np.ndarray, str]:
        """Листает карусель фликами, пока OCR заголовка не совпадёт с целью."""
        height, width = frame.shape[:2]
        strip_y = int(height * self.STRIP_Y_REL)
        target_index = self.TYPE_ORDER.index(target)
        current = ""
        for attempt in range(self.MAX_TYPE_FLICKS):
            text = self._read_title(frame)
            index = self._type_index(text)
            current = self.TYPE_ORDER[index] if index >= 0 else ""
            logger.debug(f"[CityBuild] Карусель, шаг {attempt}: OCR={text!r} -> {current or '?'}.")
            if index == target_index:
                return frame, target

            if index < 0 or index < target_index:
                x1, x2 = int(width * 0.78), int(width * 0.78) - self.FLICK_DISTANCE
            else:
                x1, x2 = int(width * 0.28), int(width * 0.28) + self.FLICK_DISTANCE
            self.client.swipe(x1, strip_y, x2, strip_y, self.FLICK_MS)
            time.sleep(self.TYPE_SETTLE_S)
            fresh = self.client.get_screenshot(fresh=True)
            if fresh is None:
                break
            frame = fresh

        return frame, current

    def _read_title(self, frame: np.ndarray) -> str:
        height, width = frame.shape[:2]
        region = (
            int(self.TITLE_REL[0] * width),
            int(self.TITLE_REL[1] * height),
            int((self.TITLE_REL[2] - self.TITLE_REL[0]) * width),
            int((self.TITLE_REL[3] - self.TITLE_REL[1]) * height),
        )
        result = read_text(frame, region, scale=2.0, min_confidence=0.20)
        return result.text

    def _type_index(self, text: str) -> int:
        normalized = normalize_text(text)
        if not normalized:
            return -1
        for index, name in enumerate(self.TYPE_ORDER):
            if fuzzy_contains(normalized, self.TYPE_ALIASES[name], threshold=0.62):
                return index
        return -1

    # ------------------------------------------------------------------
    # Требования и цвет кнопки
    # ------------------------------------------------------------------
    def _requirements_ok(self, frame: np.ndarray) -> Tuple[bool, str]:
        ok, details = self._check_requirements(frame)
        if ok or self.client is None:
            return ok, details
        # OCR иногда путает цифры в строке ресурса; один повтор на свежем
        # кадре дешевле, чем потеря свободной ячейки.
        fresh = self.client.get_screenshot(fresh=True)
        if fresh is None:
            return ok, details
        ok2, details2 = self._check_requirements(fresh)
        if ok2:
            return ok2, details2
        return False, f"{details} | retry: {details2}"

    def _check_requirements(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Проверяет две зелёные галочки и пары «есть/нужно» в требованиях.

        Живой кадр 2026-10-23: OCR блока даёт
        «5 yp. / 1 yp. 3aMOK 166,000 / 200» — числа идут парами
        (уровень замка, затем ресурс). Галочки — единственные ярко-зелёные
        пятна в правой части блока.
        """
        height, width = frame.shape[:2]
        rx, ry, rw, rh = self.REQUIREMENTS_REL
        region = (int(rx * width), int(ry * height), int(rw * width), int(rh * height))
        text = read_text(frame, region, scale=2.0, min_confidence=0.20).text
        numbers = self._row_numbers(text)

        cx, cy, cw, ch = self.CHECK_ZONE_REL
        zone = frame[int(cy * height) : int(ch * height), int(cx * width) : int(cw * width)]
        ticks = self._green_ticks(zone)

        pairs_ok = (
            len(numbers) >= 4
            and numbers[0] >= numbers[1]
            and numbers[2] >= numbers[3]
        )
        ok = ticks >= 2 and pairs_ok
        return ok, f"text='{text.strip()}' numbers={numbers} ticks={ticks}"

    @staticmethod
    def _green_ticks(roi: np.ndarray) -> int:
        if roi.size == 0:
            return 0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 90, 80), (95, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(
            1
            for contour in contours
            if cv2.contourArea(contour) >= CityBuildTask.CHECK_MIN_AREA
        )

    @staticmethod
    def _row_numbers(text: str) -> List[int]:
        """Достаёт числа из блока требований по порядку.

        «5 yp. / 1 yp. ... 166,000 / 200» -> [5, 1, 166000, 200].
        OCR читает «Замок» как «3aMOK», поэтому число, за которым сразу
        идёт буква, отбрасывается — иначе в список попадала «3». Слэш
        перед стоимостью OCR иногда читает как «1» и приклеивает к числу
        ресурса («136,321 1/200»); такой «1» вырезается до разбора.
        """
        text = re.sub(r"\s*[1lI|]\s*/", " / ", text)
        numbers: List[int] = []
        for match in re.finditer(r"\d[\d\s.,]*", text):
            raw = match.group(0)
            core = raw.rstrip(" .,")
            end = match.start() + len(core)
            if end < len(text) and text[end].isalpha():
                continue
            parsed = parse_compact_number(core)
            if parsed is not None:
                numbers.append(int(parsed))
        return numbers

    @staticmethod
    def _blue_button_color(
        frame: np.ndarray, match: Tuple[int, int, float]
    ) -> Tuple[float, float]:
        """Цвет найденной синей кнопки: (насыщенность, B-R).

        Живой замер 2026-10-23: активная кнопка S≈166, B-R≈74; неактивная
        серая — S≈22, B-R≈-4. Порог с запасом отсекает серую.
        """
        x, y, _conf = match
        height, width = frame.shape[:2]
        roi = frame[max(0, y - 30) : min(height, y + 30), max(0, x - 80) : min(width, x + 80)]
        if roi.size == 0:
            return 0.0, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean())
        blue, _, red = [float(roi[:, :, i].mean()) for i in range(3)]
        return saturation, blue - red

    # ------------------------------------------------------------------
    # Возврат в город
    # ------------------------------------------------------------------
    def _wait_for_city(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            frame = self.client.get_screenshot(fresh=True)
            if frame is not None and self._is_city(frame):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.7)

    def _is_city(self, frame: np.ndarray) -> bool:
        height, width = frame.shape[:2]
        nav = self.vision.find_template(
            frame,
            self.NAV_TEMPLATE,
            threshold=0.75,
            region=(0, int(height * 0.86), width, int(height * 0.14)),
        )
        return nav is not None and not self._is_build_screen(frame)

    def _leave_build_screen(self) -> None:
        """Выходит с экрана постройки, не подтверждая покупку."""
        self.client.press_back()
        time.sleep(1.0)

    def _leave_foreign_screen(self) -> None:
        """Закрывает случайно открытое чужое окно, не трогая город."""
        frame = self.client.get_screenshot(fresh=True)
        if frame is None:
            return
        if self._is_city(frame):
            return
        logger.info("[CityBuild] Открылось чужое окно — закрываем Back.")
        self.client.press_back()
        time.sleep(0.9)
