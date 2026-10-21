"""
Единый OCR-слой проекта.

Решает три задачи:
1. Один экземпляр RapidOCR на весь процесс (модель весит сотни мегабайт,
   раньше каждая задача создавала свой).
2. Чтение текста по областям интереса (ROI) с масштабированием — полный
   кадр 1080x2436 распознаётся в разы дольше, чем нужный фрагмент.
3. Нормализация и нечёткое сравнение русского текста: OCR путает
   кириллицу с латиницей ("вперед" -> "bnepe"), поэтому сравнение идёт
   по канонизированным строкам, а не по точному вхождению.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2) в координатах исходного кадра
Region = Tuple[int, int, int, int]  # (x, y, w, h)


@lru_cache(maxsize=1)
def get_ocr():
    """Ленивая инициализация единственного экземпляра RapidOCR."""
    from rapidocr_onnxruntime import RapidOCR

    logger.info("[OCR] Инициализация RapidOCR...")
    started = time.perf_counter()
    engine = RapidOCR()
    logger.info(f"[OCR] RapidOCR готов за {time.perf_counter() - started:.2f}с")
    return engine


@dataclass(frozen=True)
class TextItem:
    """Распознанная строка с координатами в пикселях исходного кадра."""

    text: str
    confidence: float
    box: Box
    center: Tuple[int, int]

    @property
    def x(self) -> int:
        return self.center[0]

    @property
    def y(self) -> int:
        return self.center[1]


@dataclass
class OcrResult:
    items: List[TextItem] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(item.text for item in self.items)

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)

    def find(self, *keywords: str, threshold: float = 0.8) -> Optional[TextItem]:
        """Первая строка, похожая на любой из ключевых слов."""
        for item in self.items:
            if fuzzy_contains(item.text, keywords, threshold=threshold):
                return item
        return None


# ---------------------------------------------------------------------------
# Нормализация текста
# ---------------------------------------------------------------------------

# Кириллица -> латиница. Сводит визуально похожие символы к одному алфавиту.
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
)

# Частые ошибки OCR: латиница, цифры и служебные символы.
_OCR_CONFUSION = str.maketrans(
    {
        "0": "o", "1": "l", "3": "z", "4": "a", "5": "s", "6": "b", "8": "b",
        "|": "l", "$": "s", "@": "a",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Приводит строку к канону для нечёткого сравнения."""
    if not text:
        return ""
    lowered = text.lower().replace("ё", "е")
    translated = lowered.translate(_CYRILLIC_TO_LATIN).translate(_OCR_CONFUSION)
    # Оставляем только буквы и цифры: пунктуация в UI шумит.
    cleaned = re.sub(r"[^a-z0-9]+", " ", translated)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _compact(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def fuzzy_contains(text: str, keywords: Iterable[str], threshold: float = 0.8) -> bool:
    """Проверяет, встречается ли хотя бы одно ключевое слово в тексте.

    Сначала ищет точное вхождение в канонизированной строке, затем —
    нечёткое совпадение скользящим окном (для ошибок OCR).
    """
    haystack = _compact(text)
    if not haystack:
        return False
    for keyword in keywords:
        needle = _compact(keyword)
        if not needle:
            continue
        if needle in haystack:
            return True
        if _partial_ratio(needle, haystack) >= threshold:
            return True
    return False


def fuzzy_score(text: str, keyword: str) -> float:
    """Оценка похожести 0..1 (точное вхождение = 1.0)."""
    haystack = _compact(text)
    needle = _compact(keyword)
    if not haystack or not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    return _partial_ratio(needle, haystack)


def _partial_ratio(needle: str, haystack: str) -> float:
    if len(needle) > len(haystack):
        needle, haystack = haystack, needle
    window = len(needle)
    best = 0.0
    # Шаг 1 для коротких строк, иначе можно пропустить совпадение.
    for start in range(0, len(haystack) - window + 1):
        candidate = haystack[start : start + window]
        score = SequenceMatcher(None, needle, candidate).ratio()
        if score > best:
            best = score
            if best >= 0.99:
                break
    return best


# ---------------------------------------------------------------------------
# Числа из HUD
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)*)\s*([kmbкмв]|тыс|млн)?")

_MULTIPLIERS = {
    "k": 1_000,
    "к": 1_000,
    "тыс": 1_000,
    "m": 1_000_000,
    "м": 1_000_000,
    "млн": 1_000_000,
    "b": 1_000_000_000,
    "в": 1_000_000_000,
}


def parse_compact_number(text: str) -> Optional[int]:
    """Парсит "128.8K", "3,091", "1 200 000", "1.2M" в int."""
    if not text:
        return None
    cleaned = text.lower().replace("\u00a0", " ").strip()
    # "1 200 000" -> "1200000", но "1 2" не склеиваем без нужды.
    cleaned = re.sub(r"(?<=\d)[ ](?=\d{3}\b)", "", cleaned)
    cleaned = re.sub(r"(?<=\d)[ ](?=\d)", "", cleaned)
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None

    raw_number, suffix = match.group(1), (match.group(2) or "").strip()
    suffix = suffix.replace(".", "")

    # "3,091" и "1.234.567" — разделители тысяч, а не дробная часть.
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw_number):
        value = float(raw_number.replace(",", "").replace(".", ""))
    else:
        value = float(raw_number.replace(",", "."))

    multiplier = _MULTIPLIERS.get(suffix, 1)
    return int(round(value * multiplier))


# ---------------------------------------------------------------------------
# Чтение кадра / ROI
# ---------------------------------------------------------------------------


def _crop(image: np.ndarray, region: Optional[Region]) -> Tuple[np.ndarray, int, int]:
    if region is None:
        return image, 0, 0
    x, y, w, h = region
    height, width = image.shape[:2]
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return image[y : y + h, x : x + w], x, y


def read_text(
    image: np.ndarray,
    region: Optional[Region] = None,
    *,
    scale: float = 1.0,
    min_confidence: float = 0.45,
    engine=None,
) -> OcrResult:
    """Распознаёт текст в ROI и возвращает координаты в системе исходного кадра."""
    if image is None or image.size == 0:
        return OcrResult()

    crop, offset_x, offset_y = _crop(image, region)
    if scale and scale != 1.0:
        crop = _resize(crop, scale)

    started = time.perf_counter()
    raw, _ = (engine or get_ocr())(crop)
    elapsed = time.perf_counter() - started

    items: List[TextItem] = []
    for row in raw or []:
        try:
            points, text, score = row
        except (TypeError, ValueError):
            continue
        if score is not None and float(score) < min_confidence:
            continue
        text = (text or "").strip()
        if not text:
            continue

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1 = int(round(min(xs) / scale)) + offset_x
        x2 = int(round(max(xs) / scale)) + offset_x
        y1 = int(round(min(ys) / scale)) + offset_y
        y2 = int(round(max(ys) / scale)) + offset_y
        box = (x1, y1, x2, y2)
        items.append(
            TextItem(
                text=text,
                confidence=float(score) if score is not None else 0.0,
                box=box,
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            )
        )

    # Сверху вниз, слева направо — предсказуемый порядок для логов.
    items.sort(key=lambda item: (item.box[1] // 20, item.box[0]))
    return OcrResult(items=items, elapsed=elapsed)


def read_region_text(
    image: np.ndarray,
    region: Optional[Region],
    *,
    scale: float = 1.0,
    min_confidence: float = 0.45,
    engine=None,
) -> str:
    return read_text(
        image,
        region,
        scale=scale,
        min_confidence=min_confidence,
        engine=engine,
    ).text


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
