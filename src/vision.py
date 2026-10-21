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
        self._scaled_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Загрузка всех PNG-шаблонов из папки assets в память."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in self.templates_dir.rglob("*.png"):
            # Эталонные скриншоты — не шаблоны: они большие и только шумят в кэше.
            if "screens" in {part.lower() for part in file_path.parts}:
                continue
            template_name = file_path.stem
            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is not None:
                self.templates_cache[template_name] = img
                logger.debug(f"Загружен шаблон UI: {template_name} ({img.shape[1]}x{img.shape[0]})")
            else:
                logger.warning(f"Не удалось загрузить шаблон: {file_path}")

    def has_template(self, name: str) -> bool:
        """Есть ли шаблон в кэше (без шумных предупреждений в цикле)."""
        return name in self.templates_cache

    def find_best(
        self,
        screen: np.ndarray,
        template_names: List[str],
        threshold: float = 0.8,
        region: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
    ) -> Optional[Tuple[int, int, float, str]]:
        """Ищет лучший из нескольких шаблонов и возвращает (x, y, conf, name)."""
        best: Optional[Tuple[int, int, float, str]] = None
        for name in template_names:
            if not self.has_template(name):
                continue
            match = self.find_template(screen, name, threshold=threshold, region=region, scale=scale)
            if match and (best is None or match[2] > best[2]):
                best = (match[0], match[1], match[2], name)
        return best

    def add_template(self, name: str, img: np.ndarray) -> None:
        """Динамическое добавление или обновление шаблона."""
        self.templates_cache[name] = img

    def _scaled_template(self, name: str, scale: float) -> Optional[np.ndarray]:
        template = self.templates_cache.get(name)
        if template is None:
            return None
        if scale == 1.0:
            return template
        key = (name, scale)
        cached = self._scaled_cache.get(key)
        if cached is None:
            cached = cv2.resize(
                template,
                (max(1, int(template.shape[1] * scale)), max(1, int(template.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            self._scaled_cache[key] = cached
        return cached

    def find_template(
        self,
        screen: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        region: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Поиск одного вхождения шаблона на экране.
        :param screen: Кадр экрана (OpenCV BGR).
        :param template_name: Имя шаблона из кэша (без расширения).
        :param threshold: Порог уверенности совпадения (0.0 .. 1.0).
        :param region: Ограничение области поиска (x, y, w, h) в пикселях исходного кадра.
        :param scale: Масштаб поиска (0.5 = в 4 раза быстрее, координаты пересчитываются).
        :return: (center_x, center_y, confidence) или None, если не найден.
        """
        template = self._scaled_template(template_name, scale)
        if template is None:
            logger.debug(f"Шаблон '{template_name}' отсутствует в кэше")
            return None

        search_screen = screen
        if scale != 1.0:
            search_screen = cv2.resize(
                screen,
                (max(1, int(screen.shape[1] * scale)), max(1, int(screen.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        offset_x, offset_y = 0, 0
        if region:
            rx, ry, rw, rh = region
            if scale != 1.0:
                rx, ry = int(rx * scale), int(ry * scale)
                rw, rh = int(rw * scale), int(rh * scale)
            search_screen = search_screen[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry

        th, tw = template.shape[:2]
        sh, sw = search_screen.shape[:2]
        if th > sh or tw > sw or th < 2 or tw < 2:
            return None

        res = cv2.matchTemplate(search_screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            center_x = int(round((offset_x + max_loc[0] + tw / 2) / scale))
            center_y = int(round((offset_y + max_loc[1] + th / 2) / scale))
            return center_x, center_y, float(max_val)

        return None

    def find_all(
        self,
        screen: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
        min_distance: int = 20,
        region: Optional[Tuple[int, int, int, int]] = None,
        scale: float = 1.0,
    ) -> List[Tuple[int, int, float]]:
        """
        Поиск всех совпадений шаблона на экране с подавлением немаксимумов (NMS).
        Полезно для поиска множественных пузырей ресурсов или плиток.
        """
        template = self._scaled_template(template_name, scale)
        if template is None:
            return []

        search_screen = screen
        if scale != 1.0:
            search_screen = cv2.resize(
                screen,
                (max(1, int(screen.shape[1] * scale)), max(1, int(screen.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        offset_x, offset_y = 0, 0
        if region:
            rx, ry, rw, rh = region
            if scale != 1.0:
                rx, ry = int(rx * scale), int(ry * scale)
                rw, rh = int(rw * scale), int(rh * scale)
            search_screen = search_screen[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry

        th, tw = template.shape[:2]
        sh, sw = search_screen.shape[:2]
        if th > sh or tw > sw or th < 2 or tw < 2:
            return []

        res = cv2.matchTemplate(search_screen, template, cv2.TM_CCOEFF_NORMED)
        y_locs, x_locs = np.where(res >= threshold)

        candidates = []
        for x, y in zip(x_locs, y_locs):
            candidates.append((x + tw // 2, y + th // 2, float(res[y, x])))

        candidates.sort(key=lambda item: item[2], reverse=True)

        min_distance_scaled = max(1, int(min_distance * scale))
        filtered: List[Tuple[int, int, float]] = []
        for cx, cy, conf in candidates:
            too_close = False
            for fcx, fcy, _ in filtered:
                if (cx - fcx) ** 2 + (cy - fcy) ** 2 < min_distance_scaled ** 2:
                    too_close = True
                    break
            if not too_close:
                filtered.append((cx, cy, conf))

        return [
            (int(round((offset_x + cx) / scale)), int(round((offset_y + cy) / scale)), conf)
            for cx, cy, conf in filtered
        ]

    def crop(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Вырезание фрагмента изображения для сохранения шаблона."""
        return image[y : y + h, x : x + w]

    @staticmethod
    def diff_score(
        img1: Optional[np.ndarray],
        img2: Optional[np.ndarray],
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """
        Оценка визуальной разницы между двумя кадрами (от 0.0 до 1.0).
        Использует уменьшенную копию для мгновенного расчета (паттерн android-harness).

        :param region: необязательная область (x, y, w, h). Сравнение только
            внутри неё уменьшает ложные срабатывания от анимаций в других
            частях экрана (например, при проверке закрытия панели).
        """
        if img1 is None or img2 is None or img1.shape != img2.shape:
            return 1.0

        if region is not None:
            rx, ry, rw, rh = (int(v) for v in region)
            h, w = img1.shape[:2]
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = max(0, min(rw, w - rx)), max(0, min(rh, h - ry))
            if rw < 2 or rh < 2:
                return 1.0
            img1 = img1[ry : ry + rh, rx : rx + rw]
            img2 = img2[ry : ry + rh, rx : rx + rw]

        # Ресайз до быстрой сетки для экономии процессорного времени
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
