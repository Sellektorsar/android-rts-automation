"""
Клиент для взаимодействия с Android через ADB.
Поддерживает прямое чтение скриншотов в память (OpenCV),
эмуляцию нажатий, свайпов и управление приложением.
"""

import subprocess
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from loguru import logger
import socket
from concurrent.futures import ThreadPoolExecutor

from src.human_input import generate_bezier_curve, jitter_point, sleep_human


class AdbClient:
    def __init__(self, adb_path: str, ip: str, port: int, timeout: int = 10):
        self.adb_path = adb_path
        self.device_addr = f"{ip}:{port}"
        self.timeout = timeout
        self.screen_size: Optional[Tuple[int, int]] = None

    def _run_cmd(self, args: list, check: bool = False, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Запуск ADB-команды."""
        cmd = [self.adb_path, "-s", self.device_addr] + args
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                check=check,
                timeout=timeout or self.timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Команда ADB превысила таймаут ({self.timeout}с): {' '.join(cmd)}")
            raise

    def connect(self, retries: int = 3, retry_delay: float = 1.0) -> bool:
        """Подключение к устройству по Wi-Fi с поддержкой повторных попыток."""
        logger.info(f"Подключение к устройству {self.device_addr}...")
        for attempt in range(1, retries + 1):
            try:
                res = subprocess.run(
                    [self.adb_path, "connect", self.device_addr],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                logger.debug(f"ADB connect (попытка {attempt}) output: {res.stdout.strip()}")
                if "connected to" in res.stdout or "already connected" in res.stdout:
                    self._update_screen_size()
                    logger.success(f"Успешно подключено к {self.device_addr}! Разрешение: {self.screen_size}")
                    return True
                else:
                    logger.warning(f"Попытка {attempt}/{retries} подключения не удалась: {res.stdout.strip()} {res.stderr.strip()}")
                    if attempt < retries:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Исключение при подключении к ADB (попытка {attempt}): {e}")
                if attempt < retries:
                    time.sleep(retry_delay)

        logger.warning(f"Не удалось подключиться к {self.device_addr} на порту {self.port}. Пробуем авто-сканирование открытых портов...")
        new_port = self._scan_and_find_port()
        if new_port:
            logger.info(f"Найден новый порт ADB: {new_port}. Переподключаемся...")
            self.port = new_port
            self.device_addr = f"{self.ip}:{self.port}"
            self.device_serial = self.device_addr
            res = subprocess.run([self.adb_path, "connect", self.device_addr], capture_output=True, text=True, timeout=self.timeout)
            if "connected to" in res.stdout or "already connected" in res.stdout:
                self._update_screen_size()
                logger.success(f"Автоматически восстановлено подключение к {self.device_addr}!")
                return True

        logger.error(f"Не удалось подключиться к {self.device_addr} после {retries} попыток")
        return False

    def _scan_and_find_port(self) -> Optional[int]:
        """Автоматический поиск открытого порта беспроводной отладки Android (30000..50000)."""
        def check_p(p):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.25)
                res = s.connect_ex((self.ip, p))
                s.close()
                if res == 0:
                    return p
            except Exception:
                pass
            return None

        ports = [5555] + list(range(30000, 50000))
        found = []
        with ThreadPoolExecutor(max_workers=150) as ex:
            for p in ex.map(check_p, ports):
                if p:
                    found.append(p)

        for p in found:
            res = subprocess.run([self.adb_path, "connect", f"{self.ip}:{p}"], capture_output=True, text=True, timeout=3)
            if "connected to" in res.stdout or "already connected" in res.stdout:
                return p
        return None

    def is_connected(self) -> bool:
        """Проверка доступности устройства."""
        try:
            res = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            for line in res.stdout.splitlines():
                if self.device_addr in line and "device" in line:
                    return True
            return False
        except Exception:
            return False

    def _update_screen_size(self) -> None:
        """Определение физического разрешения экрана устройства."""
        res = self._run_cmd(["shell", "wm", "size"])
        out = res.stdout.decode("utf-8", errors="ignore")
        # Physical size: 1080x2436
        for line in out.splitlines():
            if "size:" in line:
                parts = line.split(":")[-1].strip().split("x")
                if len(parts) == 2:
                    self.screen_size = (int(parts[0]), int(parts[1]))
                    break

    def get_screenshot(self) -> Optional[np.ndarray]:
        """
        Захват экрана напрямую в память без промежуточной записи на диск.
        Возвращает изображение в формате OpenCV (numpy BGR array).
        """
        try:
            res = self._run_cmd(["exec-out", "screencap", "-p"], timeout=15)
            if not res.stdout:
                logger.warning("Пустой буфер скриншота от ADB")
                return None
            image = cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Ошибка при захвате скриншота: {e}")
            return None

    def tap(self, x: int, y: int, jitter_radius: int = 6) -> None:
        """Клик по координатам с добавлением гауссова шума."""
        real_x, real_y = jitter_point(x, y, radius=jitter_radius)
        self._run_cmd(["shell", "input", "tap", str(real_x), str(real_y)])
        logger.debug(f"Клик по ({real_x}, {real_y}) [исходные: ({x}, {y})]")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 350,
        smooth: bool = False
    ) -> None:
        """Свайп от (x1, y1) к (x2, y2)."""
        if not smooth:
            self._run_cmd(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
        else:
            # Сглаженный свайп по кривой Безье
            points = generate_bezier_curve((x1, y1), (x2, y2), points_count=10)
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                self._run_cmd(["shell", "input", "swipe", str(p1[0]), str(p1[1]), str(p2[0]), str(p2[1]), "30"])
        logger.debug(f"Свайп ({x1}, {y1}) -> ({x2}, {y2})")

    def press_back(self) -> None:
        """Нажатие системной кнопки 'Назад'."""
        self._run_cmd(["shell", "input", "keyevent", "4"])
        logger.debug("Нажата кнопка 'Назад'")

    def release_touch(self, x: int = 540, y: int = 1200) -> None:
        """
        Аварийный сброс зажатого тача через motionevent UP (паттерн esp32-android-agent).
        Предотвращает зависание 'зажатого пальца' при прерывании свайпа или паузе.
        """
        self._run_cmd(["shell", "input", "touchscreen", "motionevent", "UP", str(x), str(y)])
        logger.debug(f"Аварийный сброс тача motionevent UP ({x}, {y})")

    def launch_game(self, package_name: str, activity_name: str) -> bool:
        """Запуск приложения игры."""
        logger.info(f"Запуск игры {package_name}...")
        self._run_cmd(["shell", "am", "start", "-n", f"{package_name}/{activity_name}"])
        time.sleep(3)
        return self.is_game_foreground(package_name)

    def is_game_foreground(self, package_name: str) -> bool:
        """Проверка, открыта ли игра на переднем плане."""
        res = self._run_cmd(["shell", "dumpsys", "window", "windows"])
        out = res.stdout.decode("utf-8", errors="ignore")
        return package_name in out
