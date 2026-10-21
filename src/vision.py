"""
Модуль компьютерного зрения (Computer Vision).
Отвечает за поиск шаблонов (кнопок, иконок, текста) на экране
с помощью алгоритмов сопоставления шаблонов OpenCV (Template Matching).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from loguru import logger


class Vision:
    def __init__(self, templates_dir: str = "assets"):
        self.templates_dir = Path(templates_dir)
        self.templates_cache: Dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Загрузка всех PNG-шаблонов из папки assets в память."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in self.templates_dir.rglob("*.png"):
            template_name = file_path.stem
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is not None:
                self.templates_cache[template_name] = img
                logger.debug(f"Загружен шаблон UI: {template_name} ({img.shape[1]}x{img.shape[0]})")
            else:
                logger.warning(f"Не удалось загрузить шаблон: {file_path}")

    def add_template(self, name: str, img: np.ndarray) -> None:
        """Динамическое добавление или обновление шаблона."""
        self.templates_cache[name] = img

    def find_template(
        self,
        screen: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Поиск одного вхождения шаблона на экране.
        :param screen: Кадр экрана (OpenCV BGR).
        :param template_name: Имя шаблона из кэша (без расширения).
        :param threshold: Порог уверенности совпадения (0.0 .. 1.0).
        :param region: Ограничение области поиска (x, y, w, h).
        :return: (center_x, center_y, confidence) или None, если не найден.
        """
        template = self.templates_cache.get(template_name)
        if template is None:
            logger.debug(f"Шаблон '{template_name}' отсутствует в кэше")
            return None

        search_screen = screen
        offset_x, offset_y = 0, 0
        if region:
            rx, ry, rw, rh = region
            search_screen = screen[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry

        # Проверка размеров
        th, tw = template.shape[:2]
        sh, sw = search_screen.shape[:2]
        if th > sh or tw > sw:
            return None

        # Сопоставление с нормализованной взаимной корреляцией
        res = cv2.matchTemplate(search_screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            center_x = offset_x + max_loc[0] + tw // 2
            center_y = offset_y + max_loc[1] + th // 2
            return center_x, center_y, float(max_val)

        return None

    def find_all(
        self,
        screen: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        min_distance: int = 20,
    ) -> List[Tuple[int, int, float]]:
        """
        Поиск всех совпадений шаблона на экране с подавлением немаксимумов (NMS).
        Полезно для поиска множественных пузырей ресурсов или плиток.
        """
        template = self.templates_cache.get(template_name)
        if template is None:
            return []

        th, tw = template.shape[:2]
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        y_locs, x_locs = np.where(res >= threshold)

        candidates = []
        for x, y in zip(x_locs, y_locs):
            candidates.append((x + tw // 2, y + th // 2, float(res[y, x])))

        # Сортировка по уверенности совпадения
        candidates.sort(key=lambda item: item[2], reverse=True)

        filtered: List[Tuple[int, int, float]] = []
        for cand in candidates:
            cx, cy, conf = cand
            # Проверка расстояния до уже добавленных точек
            too_close = False
            for fcx, fcy, _ in filtered:
                if (cx - fcx) ** 2 + (cy - fcy) ** 2 < min_distance ** 2:
                    too_close = True
                    break
            if not too_close:
                filtered.append(cand)

        return filtered

    def crop(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Вырезание фрагмента изображения для сохранения шаблона."""
        return image[y : y + h, x : x + w]

    @staticmethod
    def diff_score(img1: Optional[np.ndarray], img2: Optional[np.ndarray]) -> float:
        """
        Оценка визуальной разницы между двумя кадрами (от 0.0 до 1.0).
        Использует уменьшенную копию для мгновенного расчета (паттерн android-harness).
        """
        if img1 is None or img2 is None or img1.shape != img2.shape:
            return 1.0
        # Ресайз до быстрой сетки (360x812) для экономии процессорного времени
        small1 = cv2.resize(img1, (360, 812), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(img2, (360, 812), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(small1, small2)
        return float(np.mean(diff) / 255.0)

    @staticmethod
    def is_screen_stable(img1: Optional[np.ndarray], img2: Optional[np.ndarray], tolerance: float = 0.015) -> bool:
        """
        Проверка стабилизации экрана и завершения анимаций перехода (паттерн android-harness).
        Возвращает True, если разница между последовательными кадрами меньше tolerance.
        """
        return Vision.diff_score(img1, img2) < tolerance
