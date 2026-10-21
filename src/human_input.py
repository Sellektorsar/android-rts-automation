"""
Модуль имитации действий человека (Human-like Input Controller).
Обеспечивает гауссово распределение координат клика,
задержки между действиями и построение кривых Безье для свайпов.
"""

import math
import random
import time
from typing import List, Tuple


def sleep_human(min_sec: float = 0.5, max_sec: float = 1.3) -> None:
    """Случайная задержка для имитации человеческой реакции."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)


def jitter_point(x: int, y: int, radius: int = 6) -> Tuple[int, int]:
    """
    Добавляет случайное гауссово смещение к координатам клика,
    чтобы клики не попадали точно в один и тот же пиксель.
    """
    if radius <= 0:
        return x, y
    offset_x = int(random.gauss(0, radius / 2))
    offset_y = int(random.gauss(0, radius / 2))
    
    # Ограничиваем максимальное отклонение радиусом
    offset_x = max(-radius, min(radius, offset_x))
    offset_y = max(-radius, min(radius, offset_y))
    return x + offset_x, y + offset_y


def generate_bezier_curve(
    start: Tuple[int, int],
    end: Tuple[int, int],
    points_count: int = 25
) -> List[Tuple[int, int]]:
    """
    Генерирует естественную траекторию свайпа по кубической кривой Безье.
    Человек никогда не ведет палец по идеально прямой линии.
    """
    x0, y0 = start
    x3, y3 = end

    # Расстояние между точками
    dist = math.hypot(x3 - x0, y3 - y0)
    jitter = max(20, int(dist * 0.15))

    # Случайные контрольные точки отклонения
    ctrl1_x = int(x0 + (x3 - x0) * 0.33 + random.randint(-jitter, jitter))
    ctrl1_y = int(y0 + (y3 - y0) * 0.33 + random.randint(-jitter, jitter))

    ctrl2_x = int(x0 + (x3 - x0) * 0.66 + random.randint(-jitter, jitter))
    ctrl2_y = int(y0 + (y3 - y0) * 0.66 + random.randint(-jitter, jitter))

    trajectory: List[Tuple[int, int]] = []
    for i in range(points_count + 1):
        t = i / float(points_count)
        # Формула кубической кривой Безье: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        bx = (
            (1 - t) ** 3 * x0
            + 3 * (1 - t) ** 2 * t * ctrl1_x
            + 3 * (1 - t) * t ** 2 * ctrl2_x
            + t ** 3 * x3
        )
        by = (
            (1 - t) ** 3 * y0
            + 3 * (1 - t) ** 2 * t * ctrl1_y
            + 3 * (1 - t) * t ** 2 * ctrl2_y
            + t ** 3 * y3
        )
        trajectory.append((int(bx), int(by)))

    return trajectory
