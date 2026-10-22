"""
Слой восприятия (Perception).

Отвечает за два вопроса, на которых держится автономность:
1. "Какой экран сейчас открыт?" — классификация по якорным шаблонам,
   а при их отсутствии — по тексту OCR.
2. "Что написано в HUD?" — ресурсы, сила, уровень лорда, VIP.

Все координаты задаются в относительных долях кадра (0..1), поэтому
логика не привязана жёстко к разрешению 1080x2436.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from src.fsm import GameState
from src.ocr import (
    OcrResult,
    fuzzy_score,
    get_ocr,
    parse_compact_number,
    read_text,
)
from src.vision import Vision

# Регион задаётся как (x1, y1, x2, y2) в долях кадра.
RelRegion = Tuple[float, float, float, float]


def to_pixels(region: Optional[RelRegion], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """Переводит относительный регион в пиксельный (x, y, w, h)."""
    if region is None:
        return None
    x1, y1, x2, y2 = region
    px1 = max(0, min(width - 1, int(x1 * width)))
    py1 = max(0, min(height - 1, int(y1 * height)))
    px2 = max(px1 + 1, min(width, int(x2 * width)))
    py2 = max(py1 + 1, min(height, int(y2 * height)))
    return px1, py1, px2 - px1, py2 - py1


@dataclass(frozen=True)
class Anchor:
    """Якорный шаблон: имя в кэше Vision, порог и необязательный регион."""

    template: str
    threshold: float = 0.8
    region: Optional[RelRegion] = None


@dataclass(frozen=True)
class ScreenProfile:
    """Правило распознавания экрана.

    groups — список групп "хотя бы один якорь из группы". Все группы
    должны выполниться. keywords — резервный путь через OCR.
    """

    state: str
    priority: int = 100
    groups: Tuple[Tuple[Anchor, ...], ...] = ()
    keywords: Tuple[str, ...] = ()
    keyword_threshold: float = 0.82


@dataclass
class Classification:
    state: str
    confidence: float = 0.0
    source: str = "none"  # anchor | ocr | none
    evidence: str = ""
    ocr_text: str = ""
    ocr_elapsed: float = 0.0
    near_misses: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def is_known(self) -> bool:
        return self.state != GameState.UNKNOWN

    @property
    def is_popup(self) -> bool:
        return self.state in {
            GameState.POPUP_OFFER,
            GameState.POPUP_GENERIC,
            GameState.POPUP_REWARD,
        }


# ---------------------------------------------------------------------------
# Профили экранов
# ---------------------------------------------------------------------------

# Верхняя панель города: старая версия и текущая. Достаточно одной.
_TOP_HUD_ANCHOR = (
    Anchor("hud_top_strip", 0.90, (0.0, 0.0, 1.0, 0.09)),
    Anchor("hud_top_row2", 0.88, (0.0, 0.02, 1.0, 0.14)),
)

# Нижняя навигация города.
_BOTTOM_NAV_ANCHOR = (
    Anchor("btn_nav_quests", 0.82, (0.0, 0.86, 1.0, 1.0)),
    Anchor("btn_nav_lord", 0.82, (0.0, 0.86, 1.0, 1.0)),
    Anchor("btn_world_map", 0.85, (0.0, 0.84, 0.30, 1.0)),
    Anchor("btn_nav_bag", 0.82, (0.0, 0.86, 1.0, 1.0)),
)

DEFAULT_PROFILES: Tuple[ScreenProfile, ...] = (
    # Самые специфичные состояния идут первыми.
    ScreenProfile(
        state=GameState.LOADING,
        priority=5,
        keywords=("загрузка", "loading", "подключение к игре"),
    ),
    ScreenProfile(
        state=GameState.SKILL_SELECTION,
        priority=10,
        groups=((Anchor("title_skill_choice", 0.78, (0.0, 0.05, 1.0, 0.45)),),),
        keywords=("выбор навыка", "подтвердить"),
    ),
    ScreenProfile(
        state=GameState.TUTORIAL_BATTLE,
        priority=12,
        # Кнопка паузы в левом верхнем углу боевого экрана. Якорь
        # проверен на реальном кадре этапа 6 и не срабатывает на городе,
        # карте кампании, брифинге, Королевской Дороге и заданиях.
        groups=((Anchor("battle_pause", 0.90, (0.0, 0.0, 0.20, 0.12)),),),
        keywords=("победить лидера", "выжить в течение", "уничтожить"),
    ),
    ScreenProfile(
        state=GameState.VICTORY_SCREEN,
        priority=15,
        groups=((Anchor("victory_banner", 0.84, (0.0, 0.25, 1.0, 0.55)),),),
        keywords=("задание выполнено", "победа", "поздравляем с победой"),
    ),
    ScreenProfile(
        state=GameState.QUESTS_MENU,
        priority=16,
        # Экран ежедневных заданий: сверху красный заголовок «Задание»,
        # ниже вкладки «Задание роста» / «Ежедневное задание». Якорь
        # специфичнее, чем зелёные кнопки «Забрать награды», иначе экран
        # ошибочно классифицируется как popup_reward.
        groups=((Anchor("title_daily_quests", 0.85, (0.35, 0.04, 0.65, 0.10)),),),
        keywords=("ежедневное задание", "задание роста"),
    ),
    ScreenProfile(
        state=GameState.POPUP_OFFER,
        priority=18,
        # Модалка "Разблокировать золотой молот" и похожие предложения.
        groups=((Anchor("modal_golden_hammer", 0.70, (0.0, 0.10, 1.0, 0.95)),),),
        keywords=("разблокировать", "золотой молот", "купить"),
    ),
    ScreenProfile(
        state=GameState.POPUP_GENERIC,
        priority=19,
        # Системный диалог "Выйти из игры?" — закрываем только через Back/Нет.
        groups=((Anchor("title_exit_confirm", 0.72, (0.30, 0.42, 0.70, 0.50)),),),
        keywords=("выйти из игры", "выход из игры"),
    ),
    ScreenProfile(
        state=GameState.POPUP_REWARD,
        priority=20,
        groups=(
            (
                Anchor("banner_reward_received", 0.80, (0.0, 0.05, 1.0, 0.95)),
                Anchor("btn_claim_green", 0.84, (0.0, 0.10, 1.0, 0.98)),
                Anchor("btn_collect_reward", 0.84, (0.0, 0.10, 1.0, 0.98)),
            ),
        ),
        keywords=("награда за задание", "получена награда", "награды главы"),
    ),
    ScreenProfile(
        state=GameState.STAGE_BRIEFING,
        priority=25,
        groups=(
            (
                Anchor("btn_challenge", 0.78, (0.0, 0.55, 1.0, 1.0)),
                Anchor("btn_stage_challenge", 0.78, (0.0, 0.55, 1.0, 1.0)),
                Anchor("btn_green_confirm", 0.84, (0.0, 0.55, 1.0, 1.0)),
            ),
        ),
        keywords=("испытание", "награда за первое прохождение"),
    ),
    ScreenProfile(
        state=GameState.CAMPAIGN_MAP,
        priority=26,
        # Полноэкранная карта кампании «Спаситель порядка». Заголовок
        # в верхней панели уникален: баннер города выглядит иначе.
        # Приоритет ниже, чем у брифинга: модалка этапа рисуется поверх
        # карты и сохраняет заголовок «Спаситель порядка».
        groups=((Anchor("title_campaign", 0.80, (0.30, 0.03, 0.70, 0.10)),),),
        keywords=("спаситель порядка", "испытания доблести", "просмотр навыка"),
    ),
    ScreenProfile(
        state=GameState.ROYAL_ROAD,
        priority=22,
        # Заголовок модалки виден всегда, тогда как нижняя навигация
        # просвечивает сквозь затемнение и даёт ложный city_view.
        groups=((Anchor("title_royal_road", 0.72, (0.20, 0.12, 0.80, 0.21)),),),
        keywords=("королевская дорога", "забрать награды", "глава"),
    ),
    ScreenProfile(
        state=GameState.ALLIANCE_MENU,
        priority=27,
        groups=((Anchor("title_alliance", 0.70, (0.35, 0.04, 0.65, 0.10)),),),
        keywords=("создать альянс", "присоединиться", "альянс"),
    ),
    ScreenProfile(
        state=GameState.QUESTS_MENU,
        priority=26,
        groups=((Anchor("title_quests", 0.75, (0.35, 0.04, 0.65, 0.09)),),),
        keywords=("задание роста", "ежедневное задание", "текущие очки"),
    ),
    ScreenProfile(
        state=GameState.BUILDING_UPGRADE,
        priority=23,
        # Панель улучшения: блок «Требуется» и синяя кнопка подтверждения.
        groups=((Anchor("title_upgrade_required", 0.78, (0.30, 0.56, 0.70, 0.64)),),),
        keywords=("требуется", "улучшить", "уровень здания"),
    ),
    ScreenProfile(
        state=GameState.BUILDING_MENU,
        priority=24,
        # Радиальное меню выбранного здания (Детали / Улучшить / Оборона).
        groups=((Anchor("btn_building_upgrade", 0.80, (0.42, 0.50, 0.68, 0.61)),),),
        keywords=("детали", "улучшить", "оборона"),
    ),
    ScreenProfile(
        state=GameState.CITY_VIEW,
        priority=30,
        # Нижняя навигация — самый надёжный признак города: верхнюю панель
        # часто перекрывают системные уведомления и всплывающие баннеры.
        groups=(_BOTTOM_NAV_ANCHOR,),
        keywords=("замок", "событие", "альянс"),
    ),
    ScreenProfile(
        state=GameState.CAMPAIGN_MAP,
        priority=40,
        # "Спаситель порядка" встречается и в баннере города, поэтому
        # опираемся на более специфичные надписи карты кампании.
        keywords=("испытания доблести", "просмотр навыка", "карта кампании"),
    ),
    ScreenProfile(
        state=GameState.ROYAL_ROAD,
        priority=45,
        keywords=("королевская дорога", "забрать награды", "глава", "развитие технологий"),
    ),
    ScreenProfile(
        state=GameState.QUESTS_MENU,
        priority=50,
        keywords=("задание роста", "ежедневное задание", "текущие очки"),
    ),
    ScreenProfile(
        state=GameState.TRAINING_MENU,
        priority=55,
        keywords=("нанять", "тренировать", "войска"),
    ),
    ScreenProfile(
        state=GameState.BUILDING_MENU,
        priority=60,
        keywords=("улучшить", "требования", "уровень здания"),
    ),
    ScreenProfile(
        state=GameState.STORY_DIALOGUE,
        priority=65,
        keywords=("белла", "пропустить"),
    ),
)


# ---------------------------------------------------------------------------
# Классификатор экранов
# ---------------------------------------------------------------------------


class ScreenClassifier:
    def __init__(
        self,
        vision: Vision,
        profiles: Sequence[ScreenProfile] = DEFAULT_PROFILES,
        *,
        ocr_scale: float = 0.5,
        allow_ocr: bool = True,
        match_scale: float = 0.5,
        city_nav_min_brightness: float = 45.0,
    ):
        self.vision = vision
        self.profiles = sorted(profiles, key=lambda item: item.priority)
        self.ocr_scale = ocr_scale
        self.allow_ocr = allow_ocr
        self.match_scale = match_scale
        self.city_nav_min_brightness = city_nav_min_brightness
        self.last_result: Optional[Classification] = None

    @staticmethod
    def _bottom_nav_brightness(screen: np.ndarray) -> float:
        """Средняя яркость нижней навигации.

        В городе она ~90, а под затемнением модального окна ~17.
        TM_CCOEFF_NORMED нечувствителен к линейному изменению яркости,
        поэтому шаблоны навигации продолжают "совпадать" даже сквозь
        полупрозрачный оверлей. Яркость закрывает эту дыру.
        """
        height = screen.shape[0]
        strip = screen[int(height * 0.90) : height]
        if strip.size == 0:
            return 0.0
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    # -- anchors ----------------------------------------------------------
    def _match_groups(self, screen: np.ndarray, profile: ScreenProfile) -> Tuple[bool, float, str]:
        height, width = screen.shape[:2]
        matched_confidences: List[float] = []
        evidence: List[str] = []

        for group in profile.groups:
            best: Optional[Tuple[str, float]] = None
            for anchor in group:
                if not self.vision.has_template(anchor.template):
                    continue
                region = to_pixels(anchor.region, width, height)
                match = self.vision.find_template(
                    screen,
                    anchor.template,
                    threshold=anchor.threshold,
                    region=region,
                    scale=self.match_scale,
                )
                if match and (best is None or match[2] > best[1]):
                    best = (anchor.template, match[2])
            if best is None:
                return False, 0.0, ""
            matched_confidences.append(best[1])
            evidence.append(f"{best[0]}={best[1]:.2f}")

        confidence = sum(matched_confidences) / len(matched_confidences) if matched_confidences else 0.0
        return True, confidence, ", ".join(evidence)

    # -- OCR --------------------------------------------------------------
    def _read_screen(self, screen: np.ndarray) -> OcrResult:
        return read_text(screen, scale=self.ocr_scale, min_confidence=0.42)

    def _match_keywords(self, screen: np.ndarray) -> Tuple[Optional[ScreenProfile], float, OcrResult, List[Tuple[str, float]]]:
        result = self._read_screen(screen)
        text = result.text
        best_profile: Optional[ScreenProfile] = None
        best_score = 0.0
        scored: List[Tuple[str, float]] = []

        for profile in self.profiles:
            # Город определяется только якорями: в тексте модалок часто
            # встречается слово "замок" (например, в списке улучшений).
            if not profile.keywords or profile.state == GameState.CITY_VIEW:
                continue
            score = max(fuzzy_score(text, keyword) for keyword in profile.keywords)
            if score > 0:
                scored.append((profile.state, score))
            if score >= profile.keyword_threshold and score > best_score:
                best_profile = profile
                best_score = score

        scored.sort(key=lambda item: item[1], reverse=True)
        return best_profile, best_score, result, scored[:3]

    # -- public -----------------------------------------------------------
    def classify(self, screen: np.ndarray, *, allow_ocr: Optional[bool] = None) -> Classification:
        if screen is None or screen.size == 0:
            return Classification(state=GameState.UNKNOWN)

        nav_brightness: Optional[float] = None
        for profile in self.profiles:
            if not profile.groups:
                continue
            matched, confidence, evidence = self._match_groups(screen, profile)
            if not matched:
                continue
            if profile.state == GameState.CITY_VIEW:
                if nav_brightness is None:
                    nav_brightness = self._bottom_nav_brightness(screen)
                if nav_brightness < self.city_nav_min_brightness:
                    # Навигация видна, но притемнена модальным окном.
                    logger.debug(
                        f"[Perception] Навигация притемнена (brightness={nav_brightness:.0f}), "
                        "город отклонён — ищем модальный экран через OCR."
                    )
                    continue
            result = Classification(
                state=profile.state,
                confidence=confidence,
                source="anchor",
                evidence=evidence,
            )
            self.last_result = result
            return result

        use_ocr = self.allow_ocr if allow_ocr is None else allow_ocr
        if use_ocr:
            profile, score, ocr_result, near_misses = self._match_keywords(screen)
            if profile is not None:
                result = Classification(
                    state=profile.state,
                    confidence=score,
                    source="ocr",
                    evidence=f"keyword score={score:.2f}",
                    ocr_text=ocr_result.text,
                    ocr_elapsed=ocr_result.elapsed,
                    near_misses=near_misses,
                )
                self.last_result = result
                return result
            unknown = Classification(
                state=GameState.UNKNOWN,
                source="ocr",
                evidence="no profile matched",
                ocr_text=ocr_result.text,
                ocr_elapsed=ocr_result.elapsed,
                near_misses=near_misses,
            )
            self.last_result = unknown
            return unknown

        result = Classification(state=GameState.UNKNOWN)
        self.last_result = result
        return result


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------


@dataclass
class HudState:
    lord_level: Optional[int] = None
    energy: Optional[int] = None
    food: Optional[int] = None
    wood: Optional[int] = None
    power: Optional[int] = None
    gold: Optional[int] = None
    gems: Optional[int] = None
    vip_level: Optional[int] = None
    raw: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Optional[int]]:
        return {
            "lord_level": self.lord_level,
            "energy": self.energy,
            "food": self.food,
            "wood": self.wood,
            "power": self.power,
            "gold": self.gold,
            "gems": self.gems,
            "vip_level": self.vip_level,
        }

    @property
    def parsed_any(self) -> bool:
        return any(value is not None for value in self.as_dict().values())


class HudParser:
    """Читает верхнюю панель города одним проходом OCR.

    Значения группируются по двум строкам и сортируются по X, а не
    привязываются к жёстким координатам: игра меняет отступы между
    версиями, и жёсткие зоны быстро устаревают.
    """

    def __init__(self, ocr_scale: float = 1.2, region: RelRegion = (0.0, 0.02, 1.0, 0.14)):
        self.ocr_scale = ocr_scale
        self.region = region

    def parse(self, screen: np.ndarray) -> HudState:
        height, width = screen.shape[:2]
        pixel_region = to_pixels(self.region, width, height)
        result = read_text(screen, pixel_region, scale=self.ocr_scale, min_confidence=0.30)

        state = HudState()
        if not result.items:
            return state

        # Системная плашка "Нажмите, чтобы отключить отладку..." перекрывает HUD.
        joined = result.text.lower()
        if "отключить отладку" in joined or "otkjiouwtb" in joined or "otnaaky" in joined:
            state.raw["overlay"] = "debug_banner"
            return state

        first_row: List[Tuple[int, int, str]] = []
        second_row: List[Tuple[int, int, str]] = []

        for item in result.items:
            text = item.text.strip()
            lowered = text.lower()

            if "vip" in lowered:
                digits = re.sub(r"\D", "", text)
                if digits:
                    state.vip_level = int(digits)
                    state.raw["vip"] = text
                continue

            # Уровень лорда — маленькое число слева под аватаром.
            if item.x < int(width * 0.10) and int(height * 0.085) <= item.y <= int(height * 0.13):
                value = parse_compact_number(text)
                if value is not None and state.lord_level is None:
                    state.lord_level = value
                    state.raw["lord_level"] = text
                continue

            value = parse_compact_number(text)
            if value is None:
                continue

            if item.y < int(height * 0.068):
                first_row.append((item.x, value, text))
            elif item.y < int(height * 0.093):
                second_row.append((item.x, value, text))

        first_row.sort(key=lambda entry: entry[0])
        second_row.sort(key=lambda entry: entry[0])

        # Первая строка: энергия, еда, дерево. Иногда энергия не читается.
        if len(first_row) >= 3:
            state.energy = first_row[0][1]
            state.food = first_row[1][1]
            state.wood = first_row[2][1]
            state.raw["energy"], state.raw["food"], state.raw["wood"] = (
                first_row[0][2],
                first_row[1][2],
                first_row[2][2],
            )
        elif len(first_row) == 2:
            state.food, state.wood = first_row[0][1], first_row[1][1]
            state.raw["food"], state.raw["wood"] = first_row[0][2], first_row[1][2]

        # Вторая строка: сила, золото, самоцветы.
        if len(second_row) >= 3:
            state.power, state.gold, state.gems = (
                second_row[0][1],
                second_row[1][1],
                second_row[2][1],
            )
            state.raw["power"], state.raw["gold"], state.raw["gems"] = (
                second_row[0][2],
                second_row[1][2],
                second_row[2][2],
            )
        elif len(second_row) == 2:
            state.gold, state.gems = second_row[0][1], second_row[1][1]
            state.raw["gold"], state.raw["gems"] = second_row[0][2], second_row[1][2]

        # Маленький бейдж уровня лорда под аватаром плохо читается в общем
        # проходе — дочитываем его отдельным увеличением.
        if state.lord_level is None:
            lord_region = to_pixels((0.0, 0.085, 0.09, 0.135), width, height)
            lord_result = read_text(screen, lord_region, scale=3.0, min_confidence=0.20)
            for item in lord_result.items:
                match = re.search(r"\d+", item.text)
                if match:
                    state.lord_level = int(match.group())
                    state.raw["lord_level"] = item.text
                    break

        return state


# ---------------------------------------------------------------------------
# Опасные состояния
# ---------------------------------------------------------------------------

_PAYMENT_KEYWORDS = (
    "купить", "оплатить", "платеж", "пополнить", "цена", "счет", "usd", "rub",
    "карта", "подписка", "спецпредложение",
)


def looks_like_payment(screen: np.ndarray, scale: float = 0.5) -> bool:
    """Грубая проверка на экран оплаты. Используется как аварийный тормоз."""
    height, width = screen.shape[:2]
    # Проверяем нижнюю треть: там обычно кнопки покупки и цена.
    region = to_pixels((0.0, 0.55, 1.0, 1.0), width, height)
    result = read_text(screen, region, scale=scale, min_confidence=0.4)
    text = result.text.lower()
    return any(keyword in text for keyword in _PAYMENT_KEYWORDS)
