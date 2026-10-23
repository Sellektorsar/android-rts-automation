"""
Быстрая проверка экранов без OCR (FastScreen).

Адаптация идей из `sonpiaz/4x-game-agent` (`games/kingshot/screen_analyzer.py`):
быстрая пиксельная классификация игровых экранов — контраст нижней
навигации, затемнение оверлея, наличие диалоговой панели, сравнение кадров.

OCR в проекте занимает ~2 c на полный проход; для простых визуальных сигналов
(наличие навигации, затемнение оверлея, появление панели) достаточно средних
яркостей по заранее известным ROI. Это ускоряет боевой цикл: детекция
оверлеев, подтверждение появления панелей и сравнение кадров после тапа
не требуют полного OCR.

Функции работают с любым BGR-кадром (OpenCV), разрешение не привязано
к конкретному устройству (все ROI — доли кадра).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class FastScreenAnalyzer:
    """Пиксельные индикаторы игровых экранов без OCR."""

    def __init__(self) -> None:
        self._stats = {"fast": 0, "fallback": 0}

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Индикаторы экранов
    # ------------------------------------------------------------------
    def has_bottom_nav(self, img: Optional[np.ndarray]) -> bool:
        """Есть ли нижняя навигационная панель (городской экран)."""
        if img is None or img.size == 0:
            return False
        height, width = img.shape[:2]
        above = img[int(0.85 * height) : int(0.91 * height), :]
        nav = img[int(0.92 * height) : int(0.99 * height), :]
        if above.size == 0 or nav.size == 0:
            return False
        above_brightness = float(above.mean())
        nav_brightness = float(nav.mean())
        contrast = nav_brightness - above_brightness
        if contrast > 50 and nav_brightness > 120:
            return True
        if nav_brightness > 130:
            b, g, r = cv2.split(nav)
            r_avg, g_avg, b_avg = float(r.mean()), float(g.mean()), float(b.mean())
            if r_avg > g_avg > b_avg and r_avg > 130:
                return True
        return False

    def overlay_level(self, img: Optional[np.ndarray]) -> float:
        """Доля затемнения оверлея: 0..1. Popup: тёмные края + светлый центр."""
        if img is None or img.size == 0:
            return 0.0
        height, width = img.shape[:2]
        edges = [
            img[int(0.12 * height) : int(0.18 * height), 0:int(0.08 * width)],
            img[int(0.12 * height) : int(0.18 * height), int(0.92 * width) : width],
            img[int(0.82 * height) : int(0.88 * height), 0:int(0.08 * width)],
            img[int(0.82 * height) : int(0.88 * height), int(0.92 * width) : width],
        ]
        edge_values = [float(edge.mean()) for edge in edges if edge.size > 0]
        if not edge_values:
            return 0.0
        avg_edge = float(np.mean(edge_values))
        center = img[int(0.35 * height) : int(0.65 * height), int(0.2 * width) : int(0.8 * width)]
        center_value = float(center.mean()) if center.size > 0 else avg_edge
        if avg_edge < 50 and center_value > avg_edge + 30:
            return min(1.0, (center_value - avg_edge) / 80.0)
        return 0.0

    def has_dialog_panel(self, img: Optional[np.ndarray]) -> bool:
        """Есть ли светлая диалоговая панель в нижней части экрана."""
        if img is None or img.size == 0:
            return False
        height, width = img.shape[:2]
        panel = img[int(0.60 * height) : int(0.85 * height), int(0.1 * width) : int(0.9 * width)]
        city = img[int(0.15 * height) : int(0.45 * height), int(0.1 * width) : int(0.9 * width)]
        if panel.size == 0:
            return False
        brightness = float(panel.mean())
        city_brightness = float(city.mean()) if city.size > 0 else brightness
        panel_std = float(panel.std())
        if brightness > 120 and brightness > city_brightness + 15 and panel_std < 70:
            return True
        if brightness > 160 and panel_std < 60:
            return True
        return False

    def classify(self, img: Optional[np.ndarray]) -> str:
        """Быстрая классификация по пикселям по пикселям: city / popup / dialog / unknown."""
        if img is None or img.size == 0:
            return "unknown"
        self._stats["fast"] += 1
        if self.overlay_level(img) > 0.5:
            return "popup"
        if self.has_dialog_panel(img):
            return "dialog"
        if self.has_bottom_nav(img):
            return "city"
        return "unknown"

    # ------------------------------------------------------------------
    # Сравнение кадров и появление элементов
    # ------------------------------------------------------------------
    def screen_changed(
        self,
        before: Optional[np.ndarray],
        after: Optional[np.ndarray],
        threshold: float = 0.92,
    ) -> bool:
        """Изменился ли экран достаточно сильно (сравнение в мелком масштабе)."""
        try:
            if before is None or after is None:
                return True
            size = (180, 320)
            a_small = cv2.resize(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
            b_small = cv2.resize(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(a_small, b_small, cv2.TM_CCOEFF_NORMED)
            return float(result[0][0]) < threshold
        except Exception:
            return True

    def menu_opened(
        self,
        before: Optional[np.ndarray],
        after: Optional[np.ndarray],
    ) -> bool:
        """Появилась ли после тапа диалоговая панель (меню/оверлей)."""
        if not self.screen_changed(before, after):
            return False
        return self.has_dialog_panel(after)
