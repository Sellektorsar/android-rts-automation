"""
Модуль SIFT-гомографии для компенсации дрейфа камеры города (DistrictAligner).

Реализация ADR-022:
Город в War & Order — изометрическая сцена. При перезапусках, свайпах или
автоматическом скролле камера смещается, из-за чего фиксированные экранные
координаты попадают в другие здания или на пустую землю.

DistrictAligner:
1. Хранит эталонный снимок квартала (assets/reference/district_ref.png).
2. Вычисляет гомографию H между эталоном и текущим кадром с маскированием
   статичного HUD (верхняя панель ресурсов, нижняя навигация, чат).
3. Переносит точки из эталонной системы координат в текущую экранную через
   cv2.perspectiveTransform.
4. Если гомография не сошлась (inliers < min_inliers или вырожденная матрица),
   возвращает None. Слепой клик по старым координатам СТРОГО ЗАПРЕЩЁН.
5. Кэширует H для одного и того же кадра, избегая повторного запуска SIFT
   при проверке нескольких точек подряд.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class DistrictAligner:
    """Выравнивание координат квартала через SIFT + RANSAC гомографию."""

    DEFAULT_REF_PATH = Path("assets/reference/district_ref.png")

    def __init__(
        self,
        ref_path: Optional[Path | str] = None,
        min_inliers: int = 15,
        contrast_threshold: float = 0.006,
        nfeatures: int = 6000,
        ransac_reproj_threshold: float = 4.0,
        ratio_test: float = 0.70,
    ):
        self.ref_path = Path(ref_path or self.DEFAULT_REF_PATH)
        self.min_inliers = min_inliers
        self.contrast_threshold = contrast_threshold
        self.nfeatures = nfeatures
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ratio_test = ratio_test

        self._sift = cv2.SIFT_create(
            nfeatures=self.nfeatures, contrastThreshold=self.contrast_threshold
        )
        self._matcher = cv2.BFMatcher()

        self._ref_img: Optional[np.ndarray] = None
        self._ref_kp: Optional[list] = None
        self._ref_des: Optional[np.ndarray] = None

        # Кэш последнего вычисленного кадра
        self._cached_frame_id: Optional[int] = None
        self._cached_H: Optional[np.ndarray] = None
        self._cached_inliers: int = 0

    def _ensure_ref_loaded(self) -> bool:
        """Загружает эталонный снимок и предрассчитывает ключевые точки."""
        if self._ref_des is not None and self._ref_kp is not None:
            return True

        if not self.ref_path.exists():
            logger.warning(
                f"[DistrictAligner] Эталонный кадр не найден: {self.ref_path}"
            )
            return False

        img = cv2.imread(str(self.ref_path))
        if img is None:
            logger.warning(
                f"[DistrictAligner] Не удалось прочитать эталон: {self.ref_path}"
            )
            return False

        self._ref_img = img
        mask = self.make_ref_mask(img.shape[:2])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self._sift.detectAndCompute(gray, mask)
        if des is None or len(kp) < self.min_inliers:
            logger.warning(
                f"[DistrictAligner] Мало ключевых точек в эталоне: {len(kp) if kp else 0}"
            )
            return False

        self._ref_kp = kp
        self._ref_des = des
        logger.debug(
            f"[DistrictAligner] Загружен эталон {self.ref_path.name}: {len(kp)} SIFT точек."
        )
        return True

    @staticmethod
    def make_ref_mask(shape: Tuple[int, int]) -> np.ndarray:
        """Маска для эталонного кадра: вырезает HUD и оставляет квартал."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Область квартала (560..1680 по вертикали, 150..1040 по горизонтали)
        y1, y2 = min(560, h), min(1680, h)
        x1, x2 = min(150, w), min(1040, w)
        mask[y1:y2, x1:x2] = 255
        # Исключаем левый верхний угол (аватарка героя, квест)
        mask[560:min(780, h), 0:min(270, w)] = 0
        return mask

    @staticmethod
    def make_live_mask(shape: Tuple[int, int]) -> np.ndarray:
        """Маска для текущего кадра: вырезает HUD, нижнюю навигацию и чат."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = min(560, h), min(1940, h)
        x1, x2 = min(150, w), min(1040, w)
        mask[y1:y2, x1:x2] = 255
        # Исключаем нижний правый угол (кнопки меню, рюкзак, почта)
        mask[min(1600, h):min(1940, h), min(850, w):min(1080, w)] = 0
        # Исключаем нижний левый угол (чат)
        mask[min(1600, h):min(1940, h), 0:min(150, w)] = 0
        return mask

    def get_homography(self, live_frame: np.ndarray) -> Optional[np.ndarray]:
        """Возвращает гомографию ref -> live с кэшированием на объект кадра."""
        if live_frame is None:
            return None

        frame_id = id(live_frame)
        if frame_id == self._cached_frame_id:
            return self._cached_H

        H, inliers = self._compute_homography(live_frame)
        self._cached_frame_id = frame_id
        self._cached_H = H
        self._cached_inliers = inliers
        return H

    def _compute_homography(
        self, live_frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int]:
        """Вычисляет матрицу гомографии ref -> live через SIFT + RANSAC."""
        if not self._ensure_ref_loaded():
            return None, 0

        assert self._ref_des is not None
        assert self._ref_kp is not None

        live_mask = self.make_live_mask(live_frame.shape[:2])
        live_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        live_kp, live_des = self._sift.detectAndCompute(live_gray, live_mask)

        if live_des is None or len(live_kp) < self.min_inliers:
            logger.debug(
                f"[DistrictAligner] Недостаточно точек на живом кадре: {len(live_kp) if live_kp else 0}"
            )
            return None, 0

        matches = self._matcher.knnMatch(self._ref_des, live_des, k=2)
        good = [
            m for m, n in matches if m.distance < self.ratio_test * n.distance
        ]

        if len(good) < self.min_inliers:
            logger.debug(
                f"[DistrictAligner] Мало хороших совпадений: {len(good)} < {self.min_inliers}"
            )
            return None, 0

        src_pts = np.float32([self._ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold
        )
        if H is None:
            logger.warning("[DistrictAligner] cv2.findHomography вернул None")
            return None, 0

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_inliers:
            logger.warning(
                f"[DistrictAligner] Недостаточно инлаеров: {inliers} < {self.min_inliers}"
            )
            return None, inliers

        # Проверка адекватности матрицы гомографии (детерминант и масштаб)
        det = np.linalg.det(H[:2, :2])
        if det <= 0.2 or det >= 5.0:
            logger.warning(
                f"[DistrictAligner] Вырожденный детерминант H: det={det:.3f}"
            )
            return None, inliers

        logger.debug(
            f"[DistrictAligner] Гомография найдена: inliers={inliers}/{len(good)}, det={det:.3f}"
        )
        return H, inliers

    def transform_point(
        self,
        point: Tuple[int, int] | Sequence[int],
        live_frame: np.ndarray,
        check_bounds: bool = True,
    ) -> Optional[Tuple[int, int]]:
        """Трансформирует точку из системы координат эталона в систему живого кадра.

        Возвращает (live_x, live_y) или None, если выравнивание не сошлось
        или точка вышла за границы экрана.
        """
        H = self.get_homography(live_frame)
        if H is None:
            return None

        p = np.float32([[point[:2]]]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(p, H)[0, 0]
        tx = int(round(float(q[0])))
        ty = int(round(float(q[1])))

        if check_bounds and live_frame is not None:
            h, w = live_frame.shape[:2]
            if tx < 0 or tx >= w or ty < 0 or ty >= h:
                logger.warning(
                    f"[DistrictAligner] Точка {point} спроецирована за границы экрана: ({tx}, {ty})"
                )
                return None

        return tx, ty
