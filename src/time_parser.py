"""
Разбор игровых длительностей из OCR.

В War & Order таймер улучшения отображается над кнопкой подтверждения
в формате ``11:52`` (минуты:секунды), реже — ``1:02:03`` (часы:минуты:секунды)
или с русскими единицами: ``11 мин 52 сек``, ``1 ч 2 мин 3 с``.

Модуль намеренно не зависит от OCR: на вход подаётся строка, на выходе —
целое число секунд или ``None``. Это позволяет тестировать разбор отдельно
от распознавания и не создавать ложный таймер при сомнительном тексте.
"""

from __future__ import annotations

import re
from typing import Optional

# Единицы измерения. Порядок в регулярном выражении важен: сначала длинные
# варианты ("мин", "час"), иначе "м" перехватит начало "мин".
_UNIT_RE = re.compile(
    r"(\d+)\s*(дн|д|час|ч|мин|м|сек|с)(?![а-яa-z])",
    re.IGNORECASE,
)

_COLON_RE = re.compile(
    r"(?<!\d)(\d{1,3})\s*:\s*(\d{1,2})(?:\s*:\s*(\d{1,2}))?(?!\d)"
)

# OCR иногда читает двоеточие как точку: "11.52".
_DOT_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[.]\s*(\d{2})(?!\d)")


def _unit_seconds(unit: str) -> Optional[int]:
    unit = unit.lower()
    if unit in {"д", "дн"}:
        return 24 * 60 * 60
    if unit in {"ч", "час"}:
        return 60 * 60
    if unit in {"м", "мин"}:
        return 60
    if unit in {"с", "сек"}:
        return 1
    return None


def parse_duration(text: str) -> Optional[int]:
    """Преобразует строку с длительностью в секунды.

    Поддерживаются форматы:
    * ``11:52`` -> 712
    * ``1:02:03`` -> 3723
    * ``11.52`` -> 712 (OCR перепутал двоеточие с точкой)
    * ``11 мин 52 сек`` -> 712
    * ``1 ч 2 мин 3 с`` -> 3723
    * ``1152`` -> 712 (компактная запись MMSS)

    Возвращает ``None``, если строка не похожа на длительность или
    содержит невозможные значения (например, 75 секунд).
    """
    if not text:
        return None

    cleaned = (
        text.lower()
        .replace("ё", "е")
        .replace("\u00a0", " ")
        .replace("：", ":")  # fullwidth colon
        .strip()
    )

    # 1. Часы:минуты:секунды или минуты:секунды.
    match = _COLON_RE.search(cleaned)
    if match:
        first = int(match.group(1))
        second = int(match.group(2))
        third = match.group(3)
        if third is None:
            if second >= 60:
                return None
            return first * 60 + second
        minutes = second
        seconds = int(third)
        if minutes >= 60 or seconds >= 60:
            return None
        return first * 3600 + minutes * 60 + seconds

    # 2. OCR прочитал двоеточие как точку.
    match = _DOT_RE.search(cleaned)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        if seconds >= 60:
            return None
        return minutes * 60 + seconds

    # 3. Русские единицы: "11 мин 52 сек", "1 ч 2 мин 3 с".
    units = _UNIT_RE.findall(cleaned)
    if units:
        total = 0
        for raw_value, unit in units:
            multiplier = _unit_seconds(unit)
            if multiplier is None:
                continue
            total += int(raw_value) * multiplier
        return total or None

    # 4. Компактная запись без разделителей: "1152" -> 11:52.
    compact = re.fullmatch(r"\s*(\d{2,4})\s*", cleaned)
    if compact:
        digits = compact.group(1)
        if len(digits) == 2:
            return int(digits)
        minutes = int(digits[:-2])
        seconds = int(digits[-2:])
        if seconds >= 60:
            return None
        return minutes * 60 + seconds

    return None
