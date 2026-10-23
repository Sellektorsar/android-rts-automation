"""
Задача улучшения зданий (BuildingUpgradeTask).

Проверенный на живом устройстве сценарий:
1. В городе тапнуть по зданию (по умолчанию — Стена Замка).
2. В радиальном меню найти кнопку «Улучшить» (три молота).
3. На панели требований найти блок «Требуется».
4. Прочитать название здания, уровни ``LV.N -> LV.M`` и таймер над синей
   кнопкой (например, ``11:52``).
5. Нажать ТОЛЬКО синюю кнопку «Улучшить» в правом нижнем углу.
6. После исчезновения панели записать в WorldModel подтверждённый таймер.

Зелёная кнопка с гемами («Получить сейчас») не нажимается никогда: поиск
подтверждения ограничен правой половиной нижней полосы. Если синей кнопки
нет (не хватает ресурсов/уровня замка), задача просто выходит, а навигация
возвращает бота в город по стрелке назад.

Проверки взяты из esp32-android-agent: после каждого шага не фиксированная
пауза, а ожидание конкретного шаблона. Если панель требований не появилась
или не исчезла после подтверждения — это фиксируется как сбой, а не успех.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from src.adb_client import AdbClient
from src.district_aligner import DistrictAligner
from src.ocr import fuzzy_contains, normalize_text, read_text
from src.tasks.base_task import BaseTask
from src.time_parser import parse_duration
from src.vision import Vision
from src.world_model import WorldModel

# Точка здания: (имя, фиксированная точка | None, имя шаблона | None, tap_dy).
# Шаблонные точки не зависят от камеры: здание ищется на свежем кадре.
# tap_dy — смещение тапа вниз от центра шаблона: в изометрии кликабельна
# подошва здания (проверено 2026-10-22: тап по телу башни ничего не
# открывает, тап по подошве открывает радиальное меню).
Spot = Tuple[Optional[str], Optional[Tuple[int, int]], Optional[str], int]


class BuildingUpgradeTask(BaseTask):
    # Живой кадр 2026-10-23: радиальная кнопка «Улучшить» — тёмный круг
    # с золотым шевроном, 130x130 при камере по умолчанию. Старый шаблон
    # btn_building_upgrade — иконка синей кнопки панели: на радиале она
    # даёт лишь 0.676 и порог 0.75 не проходит.
    RADIAL_TEMPLATE = "btn_radial_upgrade"
    RADIAL_THRESHOLD = 0.82
    # Производственные здания (лесопилка) открывают широкое меню снизу:
    # «Детали / Улучшить / Собрать». «Улучшить» — синие молоты в круге,
    # а не тёмный круг с шевроном. Живой кадр 2026-10-23: свой шаблон 1.0,
    # на узком радиале — 0.54.
    RADIAL_WIDE_TEMPLATE = "btn_radial_upgrade_wide"
    RADIAL_WIDE_THRESHOLD = 0.84
    # Медицинская палатка открывает то же двухкнопочное меню снизу
    # («Детали / Улучшить»), но молоты золотые: шаблон wide даёт на них
    # 0.776 при пороге 0.84. Свой шаблон — 1.0, на синих молотах 0.782.
    RADIAL_MEDIC_TEMPLATE = "btn_radial_upgrade_medic"
    RADIAL_MEDIC_THRESHOLD = 0.84
    # Только глиф трёх золотых молотов, без кольца и подписи: кольцо
    # меняет яркость от анимации и фона, а глиф стабилен. Живые замеры
    # 2026-10-23: золотые меню 0.89–1.0, у нижнего края экрана — 0.825
    # (кнопка мельче и затенена), синие молоты 0.718, город 0.44.
    RADIAL_GOLD_TEMPLATE = "btn_radial_upgrade_gold"
    RADIAL_GOLD_THRESHOLD = 0.82
    LEGACY_RADIAL_TEMPLATE = "btn_building_upgrade"
    LEGACY_RADIAL_THRESHOLD = 0.70
    CONFIRM_TEMPLATE = "btn_upgrade_confirm"
    REQUIRED_TEMPLATE = "title_upgrade_required"
    # Кнопка «Перейти» в строке требований появляется, когда следующий
    # уровень здания упирается в уровень замка. Вместе с ней синяя кнопка
    # «Улучшить» становится серой; такой спот помечается как упёршийся.
    GO_TEMPLATE = "btn_go_castle"
    # Живой замер 2026-10-23: с анимацией-подсказкой поверх кнопки
    # «Перейти» шаблон даёт 0.771 вместо 1.0; на панелях без капа
    # максимум 0.738 (другая кнопка внизу). Порог 0.75 разделяет их,
    # а поиск ограничен строкой требований, где ложных совпадений нет.
    GO_THRESHOLD = 0.75
    # Верхняя левая стрелка панели — безопасный выход. Android Back на
    # радиальном меню открывает диалог «Выйти из игры?», поэтому Back
    # здесь не используется.
    BACK_TEMPLATE = "btn_back_arrow"
    BACK_THRESHOLD = 0.90
    # Синяя «Построить» на экране участка. Это не панель улучшения: если
    # такой экран открыт, задача обязана закрыть его стрелкой, иначе цикл
    # вечно тапает по городу поверх занятой рабочей зоны (живой случай
    # 2026-10-23: точка попала на вход составного участка «Учебный лагерь»).
    PLOT_BUILD_TEMPLATE = "btn_build"
    PLOT_BUILD_THRESHOLD = 0.85

    # Паспорта точек: тип, уровни и состояние по каждому споту. Файл
    # пополняется при каждом открытии панели и переживает перезапуски.
    KNOWLEDGE_PATH = Path("memory_bank/building_spots.json")

    # Точки зданий в городе при камере по умолчанию. Тап по точке только
    # выделяет здание; улучшение начинается лишь после проверки шаблонов.
    DEFAULT_SPOTS: Tuple[Spot, ...] = (("wall", (639, 1500), None, 0),)

    # Здание улучшается: в меню вместо «Улучшить» видна «Ускорить» (гемы)
    # и таймер. Это не сбой точки — возвращаемся, когда постройка кончится.
    # Живой случай 2026-10-23: camp 455,1800 после старта улучшения
    # открывал меню без радиала, и задача повторяла точку бесконечно.
    BUSY_HINTS: Tuple[str, ...] = (
        "ускорить",
        "ускори",
        "yckop",
        "осталось",
        "ocтaлocь",
        "завершится",
    )
    BUSY_COOLDOWN_S = 300.0
    # Меню могло открыться, но не распознаться: повторяем точку, но не
    # бесконечно — после лимита пропускаем её до следующего круга.
    OVERLAY_RETRY_LIMIT = 4

    # Порог и перечень размеров для поиска здания по шаблону. Камера города
    # меняет зум между сессиями (перезапуск игры сбрасывает камеру),
    # поэтому шаблон ищется в нескольких относительных размерах; порог
    # 0.70: пересэмплирование и лёгкий зум снижают conf (0.749 при scale
    # 0.95 на живом кадре), при этом конкуренты ≤0.53 (см. Vision).
    # Live 2026-10-23: после панорамы кузница перерисовывается на 2%
    # меньше (scale 0.98), корреляция без сглаживания падает до 0.661;
    # blur=5 поднимает до 0.765. Поэтому сетка масштабов частая, а кадр
    # и шаблон слегка сглаживаются.
    SPOT_TEMPLATE_THRESHOLD = 0.70
    SPOT_TEMPLATE_SCALES: Tuple[float, ...] = (
        0.80, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20
    )
    SPOT_TEMPLATE_BLUR = 5
    # Похожие красные церкви дают ложные совпадения госпиталя ~0.65–0.75,
    # поэтому для него порог выше. Живые замеры 2026-10-23: истинные
    # совпадения forge 0.96–1.0, hospital 0.92–1.0, wall_gate 0.81–1.0.
    SPOT_TEMPLATE_THRESHOLDS = {
        "building_forge": 0.80,
        "building_hospital": 0.85,
        "building_wall_gate": 0.75,
    }

    # Систематический обход камеры, когда шаблонное здание вне кадра.
    # Замер 2026-10-23: свайп 300 px двигает мир на ~240 px (0.8).
    # Шаг 600 px по пальцу даёт ~480 px мира; сетка 3x3 покрывает
    # примерно ±1000 px мира вокруг стартовой камеры. Свайпы начинаются
    # на пустой земле (y=0.75 высоты) и не задевают HUD.
    SWEEP_STEP_PX = 600
    SWEEP_DURATION_MS = 900
    SWEEP_SETTLE_S = 1.5
    SWEEP_COLS = 3
    SWEEP_ROWS = 3

    # Нормализованные названия зданий (кириллица уже приведена к латинице).
    BUILDING_ALIASES: Tuple[Tuple[str, str], ...] = (
        # OCR часто читает кириллицу латиницей: «Стена» -> «CTeHa».
        ("stena", "wall"),
        ("cteha", "wall"),
        ("ctena", "wall"),
        ("zamok", "castle"),
        ("kazarm", "barracks"),
        ("kazap", "barracks"),
        ("ferma", "farm"),
        ("oepma", "farm"),
        ("lesopil", "sawmill"),
        ("kamenolom", "quarry"),
        ("rudnik", "iron_mine"),
        ("institut", "academy"),
        ("akadem", "academy"),
        ("gospital", "hospital"),
        ("arena", "arena"),
        ("tavern", "tavern"),
        ("kuzn", "forge"),
        # «Кузница» латиницей: живой OCR дал «Ky3HWUa», канон — «kyzhwua».
        ("kyzh", "forge"),
        ("ky3h", "forge"),
        ("kuzh", "forge"),
        ("rynok", "market"),
        ("posol", "embassy"),
        ("sobor", "cathedral"),
    )

    def __init__(
        self,
        client: AdbClient,
        vision: Vision,
        interval_sec: int = 60,
        spots: Optional[Iterable] = None,
        reflection=None,
        world: Optional[WorldModel] = None,
        sweep: Optional[dict] = None,
        castle: Optional[dict] = None,
        aligner: Optional[DistrictAligner | bool] = None,
        align_district: bool = False,
    ):
        super().__init__(
            "BuildingUpgrade", client, vision, interval_sec=interval_sec, reflection=reflection
        )
        self.spots: Tuple[Spot, ...] = self._parse_spots(spots)
        self.sweep = self._parse_sweep(sweep)
        self.castle = self._parse_castle(castle)
        self.world = world
        if aligner is True or (aligner is None and align_district):
            ref_path = DistrictAligner.DEFAULT_REF_PATH
            self.aligner: Optional[DistrictAligner] = (
                DistrictAligner(ref_path=ref_path) if ref_path.exists() else None
            )
        elif isinstance(aligner, DistrictAligner):
            self.aligner = aligner
        else:
            self.aligner = None
        self._spot_index = 0
        # Ключ точки -> уровень замка, на котором здание упёрлось в кап.
        # Пока замок не поднялся, такие точки пропускаются.
        self._capped: dict[str, int] = {}
        self._spot_failures: dict[str, int] = {}
        # Ключ точки -> monotonic-время, до которого здание улучшается.
        self._busy_until: dict[str, float] = {}
        self._last_castle_level: Optional[int] = None
        self._current_spot_key: Optional[str] = None
        self._current_spot: Optional[Spot] = None
        # Последнее выбранное в городе здание. Радиальное меню и панель
        # могут жить несколько тактов: после первого апгрейда панель
        # закрывается, но меню остаётся, и следующие уровни того же
        # здания приходят без повторного тапа по точке.
        self._selected_spot_key: Optional[str] = None
        self._selected_spot: Optional[Spot] = None
        self._reported_all_capped = False
        self._castle_blocked_until = 0.0
        self._capped.update(self._load_capped_from_knowledge())
        self._last_castle_level = self._castle_level()

    @classmethod
    def _parse_castle(cls, raw: Optional[dict]) -> dict:
        """Настройки апгрейда замка через «Перейти» с панели здания."""
        raw = raw or {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "retry_sec": max(60, int(raw.get("retry_sec", 600))),
            "panel_timeout_sec": max(6.0, float(raw.get("panel_timeout_sec", 15.0))),
        }

    def _load_capped_from_knowledge(self) -> dict[str, int]:
        """Возвращает точки, упёршиеся в текущий замок, из паспортов.

        После перезапуска бота не нужно заново открывать все панели:
        статус ``capped`` пишется вместе с уровнем замка, на котором он
        получен. Паспорт от прошлого уровня замка игнорируется.
        """
        path = self.KNOWLEDGE_PATH
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - база знаний не должна ронять задачу
            logger.warning(f"[BuildingUpgrade] Не удалось прочитать паспорта точек: {exc}")
            return {}
        if not isinstance(data, dict):
            return {}
        castle_level = self._castle_level()
        capped: dict[str, int] = {}
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "capped":
                continue
            try:
                entry_level = int(entry.get("castle_level") or 0)
            except (TypeError, ValueError):
                continue
            if entry_level == castle_level:
                capped[key] = castle_level
        if capped:
            logger.info(
                f"[BuildingUpgrade] Из паспортов восстановлено {len(capped)} "
                f"упёршихся точек (замок LV.{castle_level})."
            )
        return capped

    @classmethod
    def _parse_sweep(cls, raw: Optional[dict]) -> dict:
        """Настройки обхода камерой (по умолчанию выключен)."""
        raw = raw or {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "cols": min(9, max(1, int(raw.get("cols", cls.SWEEP_COLS)))),
            "rows": min(9, max(1, int(raw.get("rows", cls.SWEEP_ROWS)))),
            "step_px": max(50, int(raw.get("step_px", cls.SWEEP_STEP_PX))),
            "duration_ms": max(100, int(raw.get("duration_ms", cls.SWEEP_DURATION_MS))),
            "settle_s": max(0.0, float(raw.get("settle_s", cls.SWEEP_SETTLE_S))),
        }

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------
    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates(
            [self.RADIAL_TEMPLATE, self.CONFIRM_TEMPLATE, self.REQUIRED_TEMPLATE]
        ):
            return False

        screen = self.ensure_screen(screen, max_age=2.0)
        if screen is None:
            return False

        if self.world is not None:
            self.world.clean_expired_timers()
            if not self.world.is_builder_free():
                logger.info("[BuildingUpgrade] Все строители заняты — пропускаем.")
                return False

        self._sync_castle_level()

        # Открыт экран строительства участка — сначала закрываем его, иначе
        # точки города недоступны, а тапы уходят в кнопки чужого экрана.
        if self._looks_like_plot_build_panel(screen):
            logger.info(
                "[BuildingUpgrade] Открыт экран строительства участка — закрываем стрелкой."
            )
            self._close_panel(screen)
            return False

        # Все здания упёрлись в замок — следующий шаг цикла: поднять замок.
        if self._castle_ready():
            return self._attempt_castle_upgrade(screen)

        # Уже на панели требований (например, после ручного открытия).
        if self._looks_like_upgrade_panel(screen):
            return self._finish_upgrade(screen)

        # Уже открыто радиальное меню здания.
        if self._tap_radial(screen):
            return self._confirm_panel(screen)

        # Город: разрешаем следующую точку (по шаблону или координате),
        # затем выделяем здание и проверяем радиальное меню. Точки,
        # упёршиеся в уровень замка, пропускаются до его повышения.
        castle_level = self._castle_level()
        resolved = None
        spot_key: Optional[str] = None
        spot_def: Optional[Spot] = None
        busy_skipped = 0
        for _ in range(len(self.spots)):
            index = self._spot_index % len(self.spots)
            candidate = self.spots[index]
            self._spot_index += 1
            key = self._spot_key(candidate)
            if self._capped.get(key) == castle_level:
                continue
            if self._spot_is_busy(key):
                busy_skipped += 1
                continue
            resolved = self._resolve_spot(screen, candidate)
            if resolved is not None:
                spot_key = key
                spot_def = candidate
                break
        if resolved is None and self.sweep["enabled"]:
            # Ни одно здание не видно на текущей камере — обходим город
            # свайпами и ищем те же шаблоны на каждой позиции.
            resolved = self._sweep_for_spots()
            if resolved is not None:
                spot_key = f"sweep:{resolved[0]}:{resolved[1]}"
        if resolved is None:
            if self._all_spots_capped(castle_level):
                self._reported_all_capped = True
                logger.info(
                    "[BuildingUpgrade] Все точки упираются в уровень замка "
                    f"LV.{castle_level} — нужен апгрейд замка."
                )
                return False
            if busy_skipped:
                logger.info(
                    "[BuildingUpgrade] Свободные точки сейчас улучшаются — "
                    "ждём завершения постройки."
                )
                return False
            self.report_failure(
                "no_building_spot_found",
                "Ни одно здание из списка точек не найдено ни на кадре, ни обходом камеры",
            )
            logger.info("[BuildingUpgrade] Точки зданий не найдены даже обходом — пропуск.")
            return False
        self._reported_all_capped = False

        building_name, spot, center = resolved
        self._current_spot_key = spot_key
        self._current_spot = spot_def
        self._selected_spot_key = spot_key
        self._selected_spot = spot_def
        logger.info(f"[BuildingUpgrade] Проверяем здание {building_name or ''} в точке {spot}.")
        self.client.tap(spot[0], spot[1])

        if self.guard is not None:
            # Guard.wait_for_match ищет одномасштабно; радиальное меню —
            # мировой слой и сжимается с зумом камеры (живой кадр: 0.794
            # при scale 0.9). Поэтому ждём своим мультимасштабным поиском.
            radial = None
            after_frame = screen
            # Живой замер 2026-10-23: меню появляется через 1.5–10 с после
            # тапа (Wi-Fi ADB + анимация), а один скриншот идёт ~2 с. Окно
            # 8 с оказывалось на грани, поэтому 20 с.
            deadline = time.monotonic() + 20.0
            while True:
                fresh = self.client.get_screenshot(fresh=True)
                if fresh is not None:
                    after_frame = fresh
                    radial = self._find_radial(after_frame, center=center)
                if radial is not None or time.monotonic() >= deadline:
                    break
                time.sleep(0.6)
            if radial is None:
                attempts = self._spot_failures.get(spot_key or "", 0) + 1
                if spot_key:
                    self._spot_failures[spot_key] = attempts
                self._current_spot_key = None
                self._current_spot = None
                self.report_failure(
                    "radial_not_found",
                    "Кнопка «Улучшить» не появилась после выбора здания",
                )
                # Меню могло открыться, но не распознаться: любой следующий
                # тап по другой точке попадёт в его кнопки. Поэтому сначала
                # проверяем, есть ли рядом текст радиала, и повторяем ту же
                # точку, а не переходим к следующей. Но повтор ограничен:
                # у улучшающегося здания радиала нет вовсе.
                overlay_text = self._overlay_text_near(after_frame, center)
                busy = self._text_looks_busy(overlay_text)
                retry = False
                if busy:
                    if spot_key:
                        self._busy_until[spot_key] = time.monotonic() + self.BUSY_COOLDOWN_S
                        self._spot_failures.pop(spot_key, None)
                    logger.info(
                        "[BuildingUpgrade] Здание улучшается — точка занята, "
                        f"пропускаем на {int(self.BUSY_COOLDOWN_S)}с."
                    )
                elif len(overlay_text) >= 6:
                    if attempts < self.OVERLAY_RETRY_LIMIT:
                        retry = True
                        logger.info(
                            "[BuildingUpgrade] Рядом виден текст меню — повторяем ту же "
                            f"точку (попытка {attempts}/{self.OVERLAY_RETRY_LIMIT})."
                        )
                    else:
                        if spot_key:
                            self._spot_failures.pop(spot_key, None)
                        logger.info(
                            "[BuildingUpgrade] Меню не распознаётся — пропускаем точку "
                            f"(попыток: {attempts})."
                        )
                elif attempts >= 3 and spot_key:
                    self._spot_failures.pop(spot_key, None)
                    logger.info(
                        "[BuildingUpgrade] Точка не отвечает и меню не видно — "
                        f"пропускаем её (попыток: {attempts})."
                    )
                else:
                    retry = True
                    logger.info(
                        "[BuildingUpgrade] Кнопка «Улучшить» не появилась — повторяем "
                        f"точку (попытка {attempts}/3)."
                    )
                if retry and spot_key:
                    self._spot_index = (self._spot_index - 1) % len(self.spots)
                return False
            if not self._activate_radial(radial, after_frame):
                return False
            if self.world is not None and building_name:
                self.world.record_coord_hit(building_name, spot[0], spot[1])
            return self._confirm_panel(after_frame, building_name=building_name)

        time.sleep(1.2)
        selected = self.client.get_screenshot(fresh=True)
        if selected is None:
            self.report_failure("no_frame_after_building_tap", "Нет кадра после выбора здания")
            return False
        if not self._tap_radial(selected, center=center):
            self._current_spot_key = None
            logger.info("[BuildingUpgrade] Кнопка «Улучшить» не появилась — здание занято или недоступно.")
            return False
        return self._confirm_panel(selected, building_name=building_name)

    # ------------------------------------------------------------------
    # Постройка на пустых ячейках
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Разбор конфигурации точек
    # ------------------------------------------------------------------
    @classmethod
    def _parse_spots(cls, raw: Optional[Iterable]) -> Tuple[Spot, ...]:
        """Поддерживает форматы конфигурации.

        * ``- [639, 1500]`` — точка без имени;
        * ``- name: wall`` + ``point: [639, 1500]`` — точка с именем;
        * ``- [wall, [639, 1500]]`` — компактная пара;
        * ``- name: wall`` + ``template: building_wall_tower`` — здание
          ищется по шаблону на свежем кадре (камерно-независимо);
          ``point`` при этом не используется: если шаблон не найден,
          точка считается недоступной и пропускается;
        * ``tap_dy`` (словарный формат) — смещение тапа вниз от центра
          найденного шаблона (кликабельная подошва в изометрии), по
          умолчанию 0.
        """
        parsed: list[Spot] = []
        for item in raw or cls.DEFAULT_SPOTS:
            name: Optional[str] = None
            point = None
            template: Optional[str] = None
            tap_dy = 0

            if isinstance(item, dict):
                name = item.get("name")
                point = item.get("point") or item.get("xy") or item.get("coords")
                template = item.get("template") or None
                tap_dy = int(item.get("tap_dy") or 0)
            elif isinstance(item, (list, tuple)):
                if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (list, tuple)):
                    name = item[0]
                    point = item[1]
                elif len(item) >= 2 and all(isinstance(value, (int, float)) for value in item[:2]):
                    point = item[:2]

            if template:
                parsed.append((str(name) if name else None, None, str(template), tap_dy))
                continue

            if not point or len(point) < 2:
                continue
            parsed.append(
                (str(name) if name else None, (int(point[0]), int(point[1])), None, tap_dy)
            )

        return tuple(parsed) or cls.DEFAULT_SPOTS

    # ------------------------------------------------------------------
    # Разрешение точек зданий
    # ------------------------------------------------------------------
    @staticmethod
    def _spot_key(spot: Spot) -> str:
        """Стабильный ключ точки для учёта упора в уровень замка."""
        name, point, template, _tap_dy = spot
        if template:
            return f"tpl:{name or template}"
        if point is not None:
            return f"pt:{name or ''}:{point[0]},{point[1]}"
        return f"pt:{name or ''}"

    def _castle_level(self) -> int:
        if self.world is None:
            return 0
        account = getattr(self.world, "state", {}).get("account", {}) or {}
        try:
            return int(account.get("castle_level") or 0)
        except (TypeError, ValueError):
            return 0

    def _sync_castle_level(self) -> None:
        """Сбрасывает пометки капа, когда замок повысил уровень."""
        level = self._castle_level()
        if self._last_castle_level is not None and level != self._last_castle_level:
            if self._capped:
                logger.info(
                    f"[BuildingUpgrade] Уровень замка {self._last_castle_level} -> {level}: "
                    f"снимаем кап с {len(self._capped)} точек."
                )
            self._capped.clear()
            self._reported_all_capped = False
        self._last_castle_level = level

    def _all_spots_capped(self, castle_level: int) -> bool:
        if not self.spots:
            return False
        return all(
            self._capped.get(self._spot_key(spot)) == castle_level for spot in self.spots
        )

    @property
    def all_capped(self) -> bool:
        """Все точки упёрлись в текущий уровень замка — пора улучшать замок."""
        return self._reported_all_capped

    # ------------------------------------------------------------------
    # Замок: «Перейти» с панели упёршегося здания
    # ------------------------------------------------------------------
    def _castle_ready(self) -> bool:
        """Можно ли попробовать улучшить замок прямо сейчас."""
        if not self.castle["enabled"] or self.world is None:
            return False
        if time.time() < self._castle_blocked_until:
            return False
        if not self.world.is_builder_free():
            return False
        return self._all_spots_capped(self._castle_level())

    def _attempt_castle_upgrade(self, screen: np.ndarray) -> bool:
        """Открывает панель упёршегося здания и переходит к замку.

        «Перейти» в строке «Замок N ур. / M ур.» — единственный проверенный
        вход в панель замка. После перехода заголовок должен быть «Замок»;
        иначе синяя кнопка не нажимается.
        """
        panel = self._acquire_capped_panel(screen)
        if panel is None:
            self._castle_blocked_until = time.time() + self.castle["retry_sec"]
            return False

        match = self._find_go_button(panel)
        if not match:
            logger.info("[BuildingUpgrade] На панели нет «Перейти» — к замку не идём.")
            self._close_panel(panel)
            self._castle_blocked_until = time.time() + self.castle["retry_sec"]
            return False

        x, y, confidence = match
        logger.info(
            f"[BuildingUpgrade] Жмём «Перейти» к замку ({x}, {y}), conf={confidence:.2f}."
        )
        self.client.tap(x, y)
        self._castle_blocked_until = time.time() + self.castle["retry_sec"]

        castle_panel = self._wait_for_castle_panel()
        if castle_panel is None:
            logger.info("[BuildingUpgrade] Панель замка не открылась — отмена.")
            return False
        return self._finish_upgrade(castle_panel, building_name="castle")

    def _acquire_capped_panel(self, screen: np.ndarray) -> Optional[np.ndarray]:
        """Возвращает панель упёршегося здания, при необходимости открывая её."""
        if self._looks_like_upgrade_panel(screen):
            if self._is_capped_panel(screen):
                return screen
            self._close_panel(screen)

        radial = self._find_radial(screen)
        if radial is not None:
            if not self._activate_radial(radial, screen):
                return None
            panel = self._wait_for_panel(timeout=10.0)
            if panel is not None and self._is_capped_panel(panel):
                return panel
            if panel is not None:
                self._close_panel(panel)

        # Перебираем все упёршиеся точки: часть из них может быть занята
        # анимацией улучшения или скрыта значком ресурсов, и тап по ним
        # не открывает радиал. Раньше бралась только первая точка, и при
        # её недоступности переход к замку отменялся полностью.
        castle_level = self._castle_level()
        for candidate in self.spots:
            if self._capped.get(self._spot_key(candidate)) != castle_level:
                continue
            resolved = self._resolve_spot(screen, candidate, log_miss=False)
            if resolved is None:
                continue
            _, spot, center = resolved
            self.client.tap(spot[0], spot[1])
            radial = self._wait_for_radial(center, timeout=8.0)
            if radial is None:
                continue
            if not self._activate_radial(radial, screen):
                continue
            panel = self._wait_for_panel(timeout=8.0)
            if panel is not None:
                return panel
        return None

    def _wait_for_radial(
        self, center: Tuple[int, int], timeout: float
    ) -> Optional[Tuple[int, int, float, float]]:
        deadline = time.monotonic() + timeout
        while True:
            frame = self.client.get_screenshot(fresh=True)
            if frame is not None:
                radial = self._find_radial(frame, center=center)
                if radial is not None:
                    return radial
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.6)

    def _wait_for_panel(self, timeout: float) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout
        while True:
            frame = self.client.get_screenshot(fresh=True)
            if frame is not None and self._looks_like_upgrade_panel(frame):
                return frame
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.6)

    def _wait_for_castle_panel(self) -> Optional[np.ndarray]:
        """Ждёт панель с заголовком «Замок» после нажатия «Перейти»."""
        deadline = time.monotonic() + self.castle["panel_timeout_sec"]
        last_panel: Optional[np.ndarray] = None
        while True:
            frame = self.client.get_screenshot(fresh=True)
            if frame is not None and self._looks_like_upgrade_panel(frame):
                last_panel = frame
                context = self._read_upgrade_context(frame)
                if context.get("name") == "castle":
                    return frame
            if time.monotonic() >= deadline:
                if last_panel is not None:
                    self._close_panel(last_panel)
                return None
            time.sleep(0.8)

    def _resolve_spot(
        self, screen: np.ndarray, spot: Spot, *, log_miss: bool = True
    ) -> Optional[Tuple[Optional[str], Tuple[int, int], Tuple[int, int]]]:
        """Преобразует точку из конфигурации в тапабельные координаты.

        * Шаблонная точка: здание ищется мультимасштабно на текущем кадре;
          при отсутствии (вне экрана из-за камеры) точка пропускается.
        * Точечная: фиксированная координата в эталонной системе (ADR-022).
          Если активен aligner, переносится через SIFT-гомографию в систему кадра.
          Если гомография не сошлась, точка пропускается (слепой клик запрещён).
        """
        name, point, template, tap_dy = spot
        if not template:
            assert point is not None
            if self.aligner is not None:
                aligned = self.aligner.transform_point(point, screen)
                if aligned is None:
                    if log_miss:
                        logger.warning(
                            f"[BuildingUpgrade] SIFT-выравнивание не сошлось для точки '{name or ''}' {point}. "
                            "Пропускаем точку для предотвращения слепого клика."
                        )
                    return None
                logger.info(
                    f"[BuildingUpgrade] Здание '{name or ''}' (SIFT-aligned): ref={point} -> live={aligned}."
                )
                return name, aligned, aligned
            return name, point, point

        found = self.vision.find_template_multiscale(
            screen,
            template,
            threshold=self.SPOT_TEMPLATE_THRESHOLDS.get(
                template, self.SPOT_TEMPLATE_THRESHOLD
            ),
            scales=self.SPOT_TEMPLATE_SCALES,
            blur=self.SPOT_TEMPLATE_BLUR,
        )
        if found is None:
            if log_miss:
                logger.info(
                    f"[BuildingUpgrade] Здание-шаблон '{template}' не найдено на кадре — пропускаем."
                )
            return None
        x, y, confidence, scale = found
        tap_x, tap_y = int(x), int(y) + round(tap_dy * scale)
        logger.info(
            f"[BuildingUpgrade] Здание '{name or template}' по шаблону: ({x}, {y}), "
            f"conf={confidence:.2f}, scale={scale:.2f}, tap=({tap_x}, {tap_y})."
        )
        return name, (tap_x, tap_y), (tap_x, tap_y)

    # ------------------------------------------------------------------
    # Систематический обход камеры
    # ------------------------------------------------------------------
    @classmethod
    def _sweep_cells(
        cls, cols: int, rows: int, step_px: int
    ) -> list[tuple[int, int]]:
        """Змейкой обходит сетку вокруг стартовой камеры.

        Возвращает смещения камеры (не свайпы) от стартовой позиции:
        сначала верхняя строка слева направо, следующая — справа налево
        и т.д. Сетка отцентрована: при 3x3 стартовая камера — центральная
        ячейка, поэтому она пропускается (её уже проверил execute).
        """
        left = (cols - 1) // 2
        up = (rows - 1) // 2
        cells: list[tuple[int, int]] = []
        for row in range(rows):
            columns = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
            for col in columns:
                cells.append(((col - left) * step_px, (row - up) * step_px))
        return cells

    def _pan_camera(self, dx: int, dy: int) -> bool:
        """Сдвигает камеру свайпом по пустой земле на (dx, dy) пикселей пальца.

        Свайп не может начинаться за пределами экрана, поэтому для
        горизонтали старт берётся у противоположного края (20% или 80%
        ширины), для вертикали — на 35%/60% высоты. Направление движения
        мира совпадает с пальцем; для покрытия сетки это не критично,
        главное — одинаковый шаг.
        """
        size = getattr(self.client, "screen_size", None)
        width, height = size if size else (1080, 2436)
        if dx and dy:
            return False
        if dx:
            y = int(height * 0.75)
            x1 = int(width * 0.20) if dx > 0 else int(width * 0.80)
            x2 = x1 + dx
            if x2 < 0 or x2 >= width:
                return False
            self.client.swipe(
                x1, y, x2, y, duration_ms=self.sweep["duration_ms"]
            )
            return True
        if dy:
            x = width // 2
            y1 = int(height * 0.35) if dy > 0 else int(height * 0.60)
            y2 = y1 + dy
            if y2 < 0 or y2 >= height:
                return False
            self.client.swipe(
                x, y1, x, y2, duration_ms=self.sweep["duration_ms"]
            )
            return True
        return False

    def _sweep_for_spots(
        self,
    ) -> Optional[Tuple[Optional[str], Tuple[int, int], Tuple[int, int]]]:
        """Ищет шаблонные здания обходом камеры.

        На каждой позиции сетки проверяются все шаблонные точки подряд
        (приоритет — порядок в конфиге). Если здание найдено, камера
        остаётся на нём; если нет — все свайпы отыгрываются назад, чтобы
        не оставлять камеру в случайном углу.
        """
        template_spots = [spot for spot in self.spots if spot[2]]
        if not template_spots:
            return None

        cols, rows = self.sweep["cols"], self.sweep["rows"]
        step = self.sweep["step_px"]
        cells = self._sweep_cells(cols, rows, step)
        moves: list[tuple[int, int]] = []
        searched = 0

        logger.info(
            f"[BuildingUpgrade] Здания вне кадра — обход камерой {cols}x{rows}, "
            f"шаг {step}px, пауза после свайпа {self.sweep['settle_s']:.1f}с."
        )

        prev_x = prev_y = 0
        aborted = False
        for cell_x, cell_y in cells:
            dx, dy = cell_x - prev_x, cell_y - prev_y
            if dx or dy:
                # Диагональ возможна только на первом переходе к углу
                # сетки: выполняем двумя свайпами по осям.
                for axis_dx, axis_dy in ((dx, 0), (0, dy)):
                    if not (axis_dx or axis_dy):
                        continue
                    if not self._pan_camera(axis_dx, axis_dy):
                        logger.warning(
                            f"[BuildingUpgrade] Свайп ({axis_dx}, {axis_dy}) "
                            f"не помещается на экран — обход прерван."
                        )
                        aborted = True
                        break
                    moves.append((axis_dx, axis_dy))
                if aborted:
                    break
                prev_x, prev_y = cell_x, cell_y
                time.sleep(self.sweep["settle_s"])

            if (cell_x, cell_y) == (0, 0):
                # Стартовую камеру уже проверил execute.
                continue

            frame = self.client.get_screenshot(fresh=True)
            if frame is None:
                continue
            searched += 1
            logger.info(
                f"[BuildingUpgrade] Обход: позиция {searched} "
                f"(смещение {cell_x:+d},{cell_y:+d}) — ищем."
            )
            for spot in template_spots:
                resolved = self._resolve_spot(frame, spot, log_miss=False)
                if resolved is not None:
                    logger.success(
                        f"[BuildingUpgrade] Обход нашёл "
                        f"'{resolved[0] or spot[2]}' на позиции {searched}."
                    )
                    return resolved

        logger.info(
            f"[BuildingUpgrade] Обход завершён: проверено {searched} позиций, "
            f"здание не найдено. Возвращаю камеру."
        )
        for axis_dx, axis_dy in reversed(moves):
            self._pan_camera(-axis_dx, -axis_dy)
        if moves:
            time.sleep(self.sweep["settle_s"])
        return None

    # ------------------------------------------------------------------
    # Области поиска
    # ------------------------------------------------------------------
    @staticmethod
    def _radial_region(
        frame: np.ndarray, center: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int, int]:
        """Область радиального меню.

        Без центра — историческая фрейм-относительная область. С центром
        здания область должна быть щедрой: слоты кольца вращаются вокруг
        здания в зависимости от позиции на экране (живые замеры
        2026-10-22: тап (639,1500)->кнопка (590,1358); (710,1610)->(584,1382);
        (476,1379)->(591,1377) — смещение от -228 до -2 по вертикали и от
        -126 до +115 по горизонтали).
        """
        height, width = frame.shape[:2]
        if center is None:
            return (
                int(width * 0.42),
                int(height * 0.50),
                int(width * 0.26),
                int(height * 0.11),
            )
        cx, cy = center
        region_width = int(width * 0.40)
        region_height = int(height * 0.33)
        x = max(0, min(int(cx - width * 0.20), width - region_width))
        y = max(0, min(int(cy - height * 0.28), height - region_height))
        return (x, y, region_width, region_height)

    @staticmethod
    def _required_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.30),
            int(height * 0.56),
            int(width * 0.40),
            int(height * 0.08),
        )

    @staticmethod
    def _title_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.30),
            int(height * 0.04),
            int(width * 0.45),
            int(height * 0.05),
        )

    @staticmethod
    def _levels_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        return (
            int(width * 0.38),
            int(height * 0.42),
            int(width * 0.28),
            int(height * 0.05),
        )

    @staticmethod
    def _timer_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Область таймера над синей кнопкой, правее зелёной кнопки гемов."""
        height, width = frame.shape[:2]
        return (
            int(width * 0.60),
            int(height * 0.90),
            int(width * 0.30),
            int(height * 0.06),
        )

    @staticmethod
    def _build_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Нижняя правая полоса, где стоит синяя «Построить»."""
        height, width = frame.shape[:2]
        return (
            int(width * 0.55),
            int(height * 0.85),
            int(width * 0.45),
            int(height * 0.15),
        )

    # ------------------------------------------------------------------
    # Шаги
    # ------------------------------------------------------------------
    def _looks_like_upgrade_panel(self, screen: np.ndarray) -> bool:
        return bool(
            self.vision.find_template(
                screen,
                self.REQUIRED_TEMPLATE,
                threshold=0.78,
                region=self._required_region(screen),
            )
        )

    def _looks_like_plot_build_panel(self, screen: np.ndarray) -> bool:
        """Синяя «Построить» внизу справа — открыт экран участка."""
        return bool(
            self.vision.find_template(
                screen,
                self.PLOT_BUILD_TEMPLATE,
                threshold=self.PLOT_BUILD_THRESHOLD,
                region=self._build_region(screen),
            )
        )

    def _find_radial(
        self, screen: np.ndarray, center: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int, float, float]]:
        """Ищет «Улучшить» мультимасштабно: область у здания, затем весь кадр.

        Живой кадр 2026-10-23: кнопка радиала — тёмный круг 130x130 с золотым
        шевроном. Старый шаблон панельной кнопки даёт на ней 0.676, поэтому
        основной шаблон — btn_radial_upgrade. Запасной поиск по всему кадру
        нужен потому, что кольцо может выйти за область вокруг исходного тапа.
        """
        for template, threshold in (
            (self.RADIAL_TEMPLATE, self.RADIAL_THRESHOLD),
            (self.RADIAL_WIDE_TEMPLATE, self.RADIAL_WIDE_THRESHOLD),
            (self.RADIAL_GOLD_TEMPLATE, self.RADIAL_GOLD_THRESHOLD),
            (self.RADIAL_MEDIC_TEMPLATE, self.RADIAL_MEDIC_THRESHOLD),
            (self.LEGACY_RADIAL_TEMPLATE, self.LEGACY_RADIAL_THRESHOLD),
        ):
            match = self.vision.find_template_multiscale(
                screen,
                template,
                threshold=threshold,
                region=self._radial_region(screen, center=center),
                scales=(0.8, 0.9, 1.0, 1.1, 1.2),
            )
            if match:
                return match
            match = self.vision.find_template_multiscale(
                screen,
                template,
                threshold=threshold,
                scales=(0.8, 0.9, 1.0, 1.1, 1.2),
            )
            if match:
                return match
        return None

    def _overlay_text_near(
        self, frame: np.ndarray, center: Tuple[int, int]
    ) -> str:
        """Нормализованный текст вокруг точки после неудачного ожидания радиала.

        Нужен, чтобы отличить открытое меню улучшающегося здания от пустой
        точки: следующий тап по другой точке иначе попадёт в кнопки меню.
        """
        height, width = frame.shape[:2]
        cx, cy = center
        x1 = max(0, cx - 340)
        x2 = min(width, cx + 340)
        y1 = max(0, cy - 420)
        y2 = min(int(height * 0.87), cy + 330)
        if x2 - x1 < 40 or y2 - y1 < 40:
            return ""
        text = read_text(
            frame,
            (x1, y1, x2 - x1, y2 - y1),
            scale=1.5,
            min_confidence=0.30,
        ).text
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _text_looks_busy(self, normalized: str) -> bool:
        return any(hint in normalized for hint in self.BUSY_HINTS)

    def _spot_is_busy(self, key: Optional[str]) -> bool:
        if not key:
            return False
        until = self._busy_until.get(key)
        if not until:
            return False
        if until <= time.monotonic():
            self._busy_until.pop(key, None)
            return False
        return True

    def _tap_radial(
        self, screen: np.ndarray, center: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Нажимает «Улучшить» в радиальном меню выбранного здания."""
        match = self._find_radial(screen, center=center)
        if not match:
            return False
        return self._activate_radial(match, screen)

    def _activate_radial(
        self,
        match: Optional[Tuple[int, int, float]],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Проверяет границы и нажимает найденную кнопку радиального меню."""
        if not match:
            return False
        if frame is not None:
            height, width = frame.shape[:2]
        else:
            size = getattr(self.client, "screen_size", None)
            if size:
                width, height = size
            else:
                width, height = 1080, 2436
        # match из find_template — (x, y, conf), из guard (find_best) — (x, y, conf, name).
        x, y, confidence = match[:3]
        if not (0.10 * width < x < 0.90 * width and 0.25 * height < y < 0.85 * height):
            logger.warning(f"[BuildingUpgrade] Кнопка вне игрового поля: ({x}, {y}).")
            return False
        logger.info(
            f"[BuildingUpgrade] Радиальное меню: «Улучшить» ({x}, {y}), conf={confidence:.2f}."
        )
        self.client.tap(x, y)
        return True

    def _confirm_panel(
        self,
        reference: np.ndarray,
        building_name: Optional[str] = None,
    ) -> bool:
        """Ждёт панель требований и нажимает синюю кнопку подтверждения."""
        if self.guard is not None:
            found = self.guard.wait_for_match(
                self.REQUIRED_TEMPLATE,
                region=self._required_region(reference),
                threshold=0.78,
                timeout=6.0,
            )
            if not found.ok:
                self._current_spot_key = None
                self.report_failure(
                    "upgrade_panel_not_found",
                    "Панель «Требуется» не появилась после нажатия «Улучшить»",
                )
                logger.info("[BuildingUpgrade] Панель требований не открылась — отмена.")
                return False
            panel = found.after
        else:
            time.sleep(1.8)
            panel = self.client.get_screenshot(fresh=True)

        if panel is None:
            return False
        return self._finish_upgrade(panel, building_name=building_name)

    def _finish_upgrade(
        self,
        panel: np.ndarray,
        building_name: Optional[str] = None,
    ) -> bool:
        """Читает контекст панели, нажимает синюю кнопку и обновляет таймер."""
        context = self._read_upgrade_context(panel)
        resolved_name = context.get("name") or building_name
        is_castle = resolved_name == "castle" or building_name == "castle"
        if is_castle:
            spot_key, spot_def = None, None
            self._current_spot_key = None
            self._current_spot = None
        else:
            # Панель могла открыться из уже висящего радиального меню:
            # spot_key текущего такта пуст, но выбранное здание известно.
            spot_key = self._current_spot_key or self._selected_spot_key
            spot_def = self._current_spot or self._selected_spot
            self._current_spot_key = None
            self._current_spot = None

        state = self._tap_confirm(panel)
        if state == "inactive":
            # Guard отдаёт кадр в момент появления заголовка «Требуется»;
            # строка замка с «Перейти» в этот момент может быть ещё не
            # отрисована. Перепроверяем на свежем кадре, иначе кап-панель
            # уходит в «не хватает условий» и точка не запоминается.
            check_frame = self.client.get_screenshot(fresh=True)
            if check_frame is None:
                check_frame = panel
            if self._is_capped_panel(check_frame):
                self._mark_capped(spot_key)
                self._remember_spot(spot_key, spot_def, context, "capped")
                logger.info(
                    "[BuildingUpgrade] Здание упёрлось в уровень замка — точка "
                    f"пропускается до повышения замка (LV.{self._castle_level()})."
                )
            else:
                self._remember_spot(spot_key, spot_def, context, "unavailable")
                logger.info(
                    "[BuildingUpgrade] Синяя кнопка «Улучшить» неактивна "
                    "(не хватает условий) — отмена."
                )
            self._close_panel(check_frame)
            return False
        if state != "pressed":
            self._remember_spot(spot_key, spot_def, context, "confirm_missing")
            logger.info("[BuildingUpgrade] Синяя кнопка «Улучшить» недоступна — отмена.")
            self._close_panel(panel)
            return False

        self._remember_spot(spot_key, spot_def, context, "upgraded")
        self._record_upgrade(
            name=resolved_name,
            duration=context.get("duration"),
            current_level=context.get("current_level"),
        )
        self.mark_executed()
        return True

    def _mark_capped(self, spot_key: Optional[str]) -> None:
        if not spot_key:
            return
        self._capped[spot_key] = self._castle_level()
        logger.info(
            f"[BuildingUpgrade] Точка {spot_key} помечена как упёршаяся "
            f"в замок LV.{self._castle_level()}."
        )

    def _remember_spot(
        self,
        spot_key: Optional[str],
        spot: Optional[Spot],
        context: dict,
        status: str,
    ) -> None:
        """Пишет паспорт точки (тип, уровни, статус) в базу знаний."""
        if not spot_key:
            return
        try:
            path = self.KNOWLEDGE_PATH
            data: dict = {}
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            name = context.get("name") or (spot[0] if spot else None)
            point = spot[1] if spot else None
            template = spot[2] if spot else None
            entry = {
                "key": spot_key,
                "name": name,
                "point": list(point) if point else None,
                "template": template,
                "level": context.get("current_level"),
                "target_level": context.get("target_level"),
                "castle_level": self._castle_level(),
                "status": status,
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
            data[spot_key] = entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - знания не должны ронять задачу
            logger.warning(f"[BuildingUpgrade] Не удалось сохранить паспорт точки: {exc}")

    @staticmethod
    def _go_region(frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Строка требования замка с кнопкой «Перейти»."""
        height, width = frame.shape[:2]
        return (
            int(width * 0.72),
            int(height * 0.58),
            int(width * 0.28),
            int(height * 0.09),
        )

    def _find_go_button(
        self, panel: np.ndarray
    ) -> Optional[Tuple[int, int, float]]:
        return self.vision.find_template(
            panel,
            self.GO_TEMPLATE,
            threshold=self.GO_THRESHOLD,
            region=self._go_region(panel),
        )

    def _is_capped_panel(self, panel: np.ndarray) -> bool:
        """На панели есть «Перейти» — не хватает именно уровня замка."""
        return bool(self._find_go_button(panel))

    def _close_panel(self, reference: Optional[np.ndarray] = None) -> bool:
        """Закрывает панель верхней левой стрелкой, не трогая навигацию игры."""
        frame = reference
        if frame is None:
            frame = self.client.get_screenshot(fresh=True)
        if frame is None:
            return False
        match = self.vision.find_template(
            frame, self.BACK_TEMPLATE, threshold=self.BACK_THRESHOLD
        )
        if not match:
            logger.warning(
                "[BuildingUpgrade] Стрелка возврата на панели не найдена — оставляем как есть."
            )
            return False
        x, y, confidence = match
        logger.info(f"[BuildingUpgrade] Закрываем панель стрелкой ({x}, {y}), conf={confidence:.2f}.")
        self.client.tap(x, y)
        if self.guard is not None:
            gone = self.guard.wait_until_gone(
                self.REQUIRED_TEMPLATE,
                region=self._required_region(frame),
                threshold=0.78,
                timeout=4.0,
            )
            if not gone:
                logger.warning("[BuildingUpgrade] Панель не закрылась стрелкой.")
                return False
        return True

    # ------------------------------------------------------------------
    # OCR панели и WorldModel
    # ------------------------------------------------------------------
    def _read_upgrade_context(self, panel: np.ndarray) -> dict:
        """Читает название, уровни и длительность с панели улучшения.

        Все области маленькие, поэтому OCR быстрый (~0.4–0.7 с на область).
        Если что-то не распознано, возвращается частичный словарь: действие
        всё равно можно выполнить, но таймер мира не будет создан.
        """
        context: dict = {}

        title = read_text(
            panel,
            self._title_region(panel),
            scale=2.0,
            min_confidence=0.30,
        )
        name = self._building_name_from_text(title.text)
        if name:
            context["name"] = name

        levels = read_text(
            panel,
            self._levels_region(panel),
            scale=2.0,
            min_confidence=0.30,
        )
        parsed_levels = []
        for item in levels.items:
            match = re.search(r"L\s*V\.?\s*(\d+)", item.text, re.IGNORECASE)
            if match and item.confidence >= 0.55:
                parsed_levels.append(int(match.group(1)))
        if parsed_levels:
            context["current_level"] = parsed_levels[0]
            if len(parsed_levels) > 1:
                context["target_level"] = parsed_levels[1]

        timer = read_text(
            panel,
            self._timer_region(panel),
            scale=2.0,
            min_confidence=0.25,
        )
        duration = parse_duration(timer.text)
        if duration is None and len(timer.items) > 1:
            duration = parse_duration(" ".join(item.text for item in timer.items))
        if duration is not None:
            context["duration"] = duration

        logger.debug(
            "[BuildingUpgrade] Панель: name={}, level={}, target={}, duration={}, timer='{}'.".format(
                context.get("name"),
                context.get("current_level"),
                context.get("target_level"),
                context.get("duration"),
                timer.text,
            )
        )
        return context

    def _building_name_from_text(self, text: str) -> Optional[str]:
        normalized = normalize_text(text)
        if not normalized:
            return None
        for alias, name in self.BUILDING_ALIASES:
            # Нечёткое сравнение нужно из-за OCR: «Стена» может прийти как
            # «CTeHa» или «CTena», а «Казарма» — как «Ka3apma».
            if fuzzy_contains(normalized, [alias], threshold=0.78):
                return name
        return None

    def _record_upgrade(
        self,
        *,
        name: Optional[str],
        duration: Optional[int],
        current_level: Optional[int],
    ) -> None:
        if self.world is None:
            return
        building = name or "building"
        if not duration or duration <= 0:
            logger.warning(
                "[BuildingUpgrade] Улучшение подтверждено, но длительность не распознана — "
                "таймер WorldModel не создаём."
            )
            return
        if self.world.start_upgrade(building, duration, current_level=current_level):
            logger.info(
                f"[BuildingUpgrade] WorldModel: {building} улучшается, "
                f"осталось {duration}с."
            )
        else:
            logger.warning(
                "[BuildingUpgrade] WorldModel не принял таймер: все строители заняты."
            )

    # ------------------------------------------------------------------
    # Подтверждение
    # ------------------------------------------------------------------
    def _tap_confirm(self, screen: np.ndarray) -> str:
        """Нажимает только синюю кнопку в правом нижнем углу панели улучшения.

        Возвращает состояние: ``pressed`` (нажата, панель закрылась),
        ``inactive`` (серая), ``missing`` (кнопки нет) или ``tap_failed``
        (нажали, но панель осталась).
        """
        height, width = screen.shape[:2]
        required_region = self._required_region(screen)

        required = self.vision.find_template(
            screen, self.REQUIRED_TEMPLATE, threshold=0.78, region=required_region
        )
        if not required:
            return "missing"

        # Правая половина нижней полосы: зелёная кнопка гемов сюда не попадает.
        confirm_region = (
            int(width * 0.55),
            int(height * 0.90),
            int(width * 0.45),
            int(height * 0.10),
        )
        match = self.vision.find_template(
            screen, self.CONFIRM_TEMPLATE, threshold=0.84, region=confirm_region
        )
        if not match:
            return "missing"

        x, y, confidence = match
        if x < 0.60 * width:
            logger.warning(
                f"[BuildingUpgrade] Подтверждение слишком слева ({x}); возможно, это кнопка гемов."
            )
            return "missing"

        # Отсекаем неактивную (серую) кнопку: у активной синей насыщенность
        # высокая и синий канал заметно преобладает над красным.
        template = self.vision.templates_cache.get(self.CONFIRM_TEMPLATE)
        if template is not None:
            th, tw = template.shape[:2]
            x1, y1 = max(0, x - tw // 2), max(0, y - th // 2)
            roi = screen[y1 : y1 + th, x1 : x1 + tw]
            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                saturation = float(hsv[:, :, 1].mean())
                blue, _, red = [float(roi[:, :, i].mean()) for i in range(3)]
                if saturation < 60.0 or blue < red + 10.0:
                    logger.info(
                        f"[BuildingUpgrade] Кнопка неактивна (S={saturation:.0f}, B-R={blue - red:.0f}) — не нажимаем."
                    )
                    return "inactive"

        logger.success(
            f"[BuildingUpgrade] Нажимаем бесплатное «Улучшить» ({x}, {y}), conf={confidence:.2f}."
        )
        self.client.tap(x, y)

        if self.guard is not None:
            gone = self.guard.wait_until_gone(
                self.REQUIRED_TEMPLATE,
                region=required_region,
                threshold=0.78,
                timeout=5.0,
            )
            if not gone:
                self.report_failure(
                    "upgrade_not_started",
                    "Панель «Требуется» осталась после подтверждения — улучшение не началось",
                )
                logger.warning("[BuildingUpgrade] Панель не закрылась — улучшение не началось.")
                return "tap_failed"
            return "pressed"

        time.sleep(1.5)
        after = self.client.get_screenshot(fresh=True)
        if after is not None:
            still = self.vision.find_template(
                after, self.REQUIRED_TEMPLATE, threshold=0.78, region=required_region
            )
            if still:
                logger.warning("[BuildingUpgrade] Панель не закрылась — улучшение не началось.")
                return "tap_failed"
        return "pressed"
