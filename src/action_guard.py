"""
Проверка действий «наблюдение → действие → подтверждение».

Идея взята из esp32-android-agent: перед вводом нужно убедиться, что кадр
свежий и не тёмный, а после ввода — дождаться стабилизации экрана и
проверить наблюдаемый результат. Это защищает от кликов по устаревшему
кадру и от «успешных» действий, которые на самом деле не открыли меню.

Модуль не знает про конкретную игру: он работает с шаблонами, областями
кадра и разницей изображений. Все координаты — в пикселях исходного кадра.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from src.vision import Vision

Region = Tuple[int, int, int, int]


@dataclass
class GuardResult:
    """Результат проверяемого действия."""

    ok: bool
    reason: str
    before: Optional[np.ndarray] = None
    after: Optional[np.ndarray] = None
    elapsed: float = 0.0
    match: Optional[Tuple[int, int, float, str]] = None


class ActionGuard:
    """Небольшой набор проверок вокруг одного действия.

    :param client: AdbClient (или совместимый объект с get_screenshot/tap).
    :param vision: Vision для поиска шаблонов.
    :param poll_interval: пауза между повторными кадрами, секунды.
    :param change_threshold: минимальная средняя разница кадров (0..1).
    """

    def __init__(
        self,
        client,
        vision: Vision,
        *,
        poll_interval: float = 0.35,
        change_threshold: float = 0.01,
        dark_threshold: float = 2.0,
    ):
        self.client = client
        self.vision = vision
        self.poll_interval = max(0.05, float(poll_interval))
        self.change_threshold = max(0.0, float(change_threshold))
        self.dark_threshold = float(dark_threshold)

    # ------------------------------------------------------------------
    # Свежесть и пригодность кадра
    # ------------------------------------------------------------------
    def capture(self, *, fresh: bool = True) -> Optional[np.ndarray]:
        try:
            return self.client.get_screenshot(fresh=fresh)
        except Exception as exc:  # noqa: BLE001 - ADB может отвалиться в любой момент
            logger.debug(f"[Guard] Не удалось получить кадр: {exc}")
            return None

    def frame_age(self) -> float:
        """Возраст последнего кадра в секундах (inf, если кадра ещё не было)."""
        getter = getattr(self.client, "frame_age", None)
        if callable(getter):
            try:
                return float(getter())
            except Exception:  # noqa: BLE001
                return float("inf")
        return 0.0

    def is_actionable(
        self,
        frame: Optional[np.ndarray],
        *,
        max_age: float = 8.0,
    ) -> Tuple[bool, str]:
        """Можно ли действовать по этому кадру.

        Порог 8 секунд выбран с учётом реального цикла: захват кадра ~1.8 с,
        классификация с OCR может занять ещё 2–4 с. Более строгий порог
        отменял бы безопасные городские действия из-за самой задержки
        восприятия. Для боевых сценариев порог нужно задавать отдельно.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False, "no_frame"
        if float(frame.mean()) < self.dark_threshold:
            return False, "dark_frame"
        if max_age > 0:
            age = self.frame_age()
            if age > max_age:
                return False, f"stale_frame:{age:.1f}s"
        return True, "ok"

    def require_actionable(self, frame: Optional[np.ndarray], *, max_age: float = 8.0) -> bool:
        ok, reason = self.is_actionable(frame, max_age=max_age)
        if not ok:
            logger.warning(f"[Guard] Действие отменено: {reason}")
        return ok

    # ------------------------------------------------------------------
    # Ожидание шаблонов
    # ------------------------------------------------------------------
    @staticmethod
    def _as_names(names: Iterable[str] | str) -> list[str]:
        return [names] if isinstance(names, str) else list(names)

    def wait_for_match(
        self,
        names: Iterable[str] | str,
        *,
        region: Optional[Region] = None,
        threshold: float = 0.80,
        timeout: float = 4.0,
        scale: float = 1.0,
    ) -> GuardResult:
        """Ждёт появления шаблона и возвращает кадр, на котором он найден."""
        started = time.monotonic()
        wanted = [name for name in self._as_names(names) if self.vision.has_template(name)]
        if not wanted:
            logger.debug("[Guard] Нет доступных шаблонов для ожидания")
            return GuardResult(False, "template_missing", elapsed=time.monotonic() - started)

        deadline = started + max(0.0, float(timeout))
        while True:
            frame = self.capture(fresh=True)
            ok, reason = self.is_actionable(frame, max_age=max(3.0, timeout))
            if ok:
                match = self.vision.find_best(
                    frame, wanted, threshold=threshold, region=region, scale=scale
                )
                if match:
                    return GuardResult(
                        True,
                        "found",
                        before=frame,
                        after=frame,
                        elapsed=time.monotonic() - started,
                        match=match,
                    )
            else:
                logger.debug(f"[Guard] Кадр не пригоден для проверки: {reason}")

            if time.monotonic() >= deadline:
                return GuardResult(
                    False,
                    "timeout",
                    after=frame,
                    elapsed=time.monotonic() - started,
                )
            time.sleep(self.poll_interval)

    def wait_for(
        self,
        names: Iterable[str] | str,
        *,
        region: Optional[Region] = None,
        threshold: float = 0.80,
        timeout: float = 4.0,
        scale: float = 1.0,
    ) -> Optional[Tuple[int, int, float, str]]:
        """Ждёт появления любого из шаблонов и возвращает совпадение."""
        result = self.wait_for_match(
            names,
            region=region,
            threshold=threshold,
            timeout=timeout,
            scale=scale,
        )
        return result.match if result.ok else None

    def wait_until_gone(
        self,
        names: Iterable[str] | str,
        *,
        region: Optional[Region] = None,
        threshold: float = 0.80,
        timeout: float = 4.0,
        scale: float = 1.0,
    ) -> bool:
        """Ждёт исчезновения всех указанных шаблонов."""
        wanted = [name for name in self._as_names(names) if self.vision.has_template(name)]
        if not wanted:
            logger.debug("[Guard] Нет доступных шаблонов для проверки исчезновения")
            return False

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            frame = self.capture(fresh=True)
            ok, _ = self.is_actionable(frame, max_age=max(3.0, timeout))
            if ok:
                match = self.vision.find_best(
                    frame, wanted, threshold=threshold, region=region, scale=scale
                )
                if match is None:
                    return True

            if time.monotonic() >= deadline:
                return False
            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Стабильность и изменение кадра
    # ------------------------------------------------------------------
    def wait_stable(
        self,
        *,
        timeout: float = 4.0,
        tolerance: float = 0.004,
        region: Optional[Region] = None,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Ждёт, пока два соседних кадра перестанут заметно отличаться."""
        previous = self.capture(fresh=True)
        if previous is None:
            return None, False

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            if time.monotonic() >= deadline:
                return previous, False
            time.sleep(self.poll_interval)
            current = self.capture(fresh=True)
            if current is None:
                continue
            if Vision.diff_score(previous, current, region=region) < tolerance:
                return current, True
            previous = current

    def wait_for_change(
        self,
        before: Optional[np.ndarray],
        *,
        region: Optional[Region] = None,
        timeout: float = 3.0,
        min_diff: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Ждёт визуального изменения относительно ``before``."""
        if before is None:
            return self.capture(fresh=True), False

        threshold = self.change_threshold if min_diff is None else float(min_diff)
        deadline = time.monotonic() + max(0.0, float(timeout))
        latest = before
        while True:
            current = self.capture(fresh=True)
            if current is not None:
                latest = current
                if Vision.diff_score(before, current, region=region) >= threshold:
                    return current, True

            if time.monotonic() >= deadline:
                return latest, False
            time.sleep(self.poll_interval)

    def tap(
        self,
        x: int,
        y: int,
        *,
        jitter_radius: int = 6,
        expect_change: bool = False,
        region: Optional[Region] = None,
        timeout: float = 3.0,
    ) -> GuardResult:
        """Тап с проверкой свежести кадра и (опционально) изменения экрана."""
        started = time.monotonic()
        before = self.capture(fresh=True)
        ok, reason = self.is_actionable(before)
        if not ok:
            return GuardResult(False, reason, before, None, time.monotonic() - started)

        self.client.tap(x, y, jitter_radius=jitter_radius)
        if not expect_change:
            return GuardResult(True, "submitted", before, None, time.monotonic() - started)

        after, changed = self.wait_for_change(
            before, region=region, timeout=timeout
        )
        return GuardResult(
            changed,
            "changed" if changed else "no_change",
            before,
            after,
            time.monotonic() - started,
        )
