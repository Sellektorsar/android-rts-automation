"""
Клиент для взаимодействия с Android через ADB.

Поддерживает чтение скриншотов напрямую в память (OpenCV), эмуляцию
нажатий и свайпов, переподключение по Wi-Fi и кэширование кадров.

Кэш кадров важен для автономного цикла: один и тот же кадр используется
классификатором экрана, HUD-парсером и задачами, вместо трёх-четырёх
последовательных `screencap` (каждый ~0.5-1.5 с).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from src.human_input import jitter_point, sleep_human

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class AdbClient:
    def __init__(self, adb_path: str, ip: str, port: int, timeout: int = 10, frame_ttl: float = 1.5):
        self.adb_path = adb_path
        self.ip = ip
        self.port = int(port)
        self.device_addr = f"{ip}:{self.port}"
        self.timeout = timeout
        self.screen_size: Optional[Tuple[int, int]] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._frame_ttl: float = frame_ttl
        self._remote_frame = "/data/local/tmp/wao_frame.png"
        self._local_frame = Path(tempfile.gettempdir()) / "wao_frame.png"
        self._orig_screen_off_timeout: Optional[str] = None
        self._orig_stay_on: Optional[str] = None
        self._last_port_scan: float = 0.0

    # ------------------------------------------------------------------
    # Низкий уровень
    # ------------------------------------------------------------------
    def _run_cmd(
        self,
        args: List[str],
        check: bool = False,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Запуск ADB-команды без лишних окон консоли на Windows."""
        cmd = [self.adb_path, "-s", self.device_addr] + args
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                check=check,
                timeout=timeout or self.timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Команда ADB превысила таймаут ({timeout or self.timeout}с): {' '.join(cmd)}")
            raise

    @staticmethod
    def _decode(data: Optional[bytes]) -> str:
        if not data:
            return ""
        return data.decode("utf-8", errors="ignore")

    # ------------------------------------------------------------------
    # Подключение
    # ------------------------------------------------------------------
    def connect(self, retries: int = 3, retry_delay: float = 1.5, allow_discovery: bool = True) -> bool:
        """Подключение к устройству по Wi-Fi с повторами и поиском через mDNS."""
        logger.info(f"Подключение к устройству {self.device_addr}...")
        for attempt in range(1, retries + 1):
            if self._try_connect(self.device_addr):
                self._update_screen_size()
                logger.success(f"Успешно подключено к {self.device_addr}! Разрешение: {self.screen_size}")
                return True
            logger.warning(f"Попытка {attempt}/{retries} подключения к {self.device_addr} не удалась")
            if attempt < retries:
                time.sleep(retry_delay)

        if allow_discovery:
            discovered = self.discover_wireless_addresses()
            for address in discovered:
                if address == self.device_addr:
                    continue
                logger.info(f"Найден ADB-сервис через mDNS: {address}. Пробуем подключиться...")
                if self._try_connect(address):
                    self._switch_address(address)
                    self._update_screen_size()
                    logger.success(f"Подключение восстановлено через mDNS: {self.device_addr}")
                    return True

            # mDNS часто молчит после перезагрузки/сна. Порт беспроводной
            # отладки случайный, поэтому ищем открытый TCP-порт в диапазоне
            # Android. Делаем это не чаще раза в минуту.
            now = time.time()
            if now - self._last_port_scan >= 60:
                self._last_port_scan = now
                for port in self.scan_wireless_ports():
                    address = f"{self.ip}:{port}"
                    if address == self.device_addr:
                        continue
                    if self._try_connect(address):
                        self._switch_address(address)
                        self._update_screen_size()
                        logger.success(f"Подключение восстановлено по сканированию портов: {self.device_addr}")
                        return True

        logger.error(f"Не удалось подключиться к {self.device_addr} после {retries} попыток")
        return False

    def scan_wireless_ports(
        self,
        start: int = 30000,
        end: int = 50001,
        max_workers: int = 500,
        timeout: float = 0.09,
    ) -> List[int]:
        """Быстрый поиск открытых портов беспроводной отладки на IP устройства."""
        logger.info(f"[ADB] Сканируем порты {start}-{end - 1} на {self.ip}...")

        def check(port: int) -> Optional[int]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                return port if sock.connect_ex((self.ip, port)) == 0 else None
            except OSError:
                return None
            finally:
                sock.close()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            found = [port for port in executor.map(check, range(start, end)) if port]

        if found:
            logger.info(f"[ADB] Открытые порты: {found}")
        else:
            logger.warning("[ADB] Открытых портов в диапазоне не найдено")
        return found

    def _try_connect(self, address: str) -> bool:
        try:
            result = subprocess.run(
                [self.adb_path, "connect", address],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
            output = f"{result.stdout} {result.stderr}".lower()
            if "cannot connect" in output or "failed to connect" in output:
                return False
            # Не доверяем только тексту: `adb devices` — источник истины.
            if f"{address}\tdevice" in self._list_devices():
                return True
            if "connected to" in output or "already connected" in output:
                time.sleep(0.3)
                return f"{address}\tdevice" in self._list_devices()
        except Exception as exc:  # noqa: BLE001 - сеть/ADB могут падать по-разному
            logger.debug(f"Ошибка подключения к {address}: {exc}")
        # Убираем мусорную запись offline, иначе `adb devices` разрастается.
        if f"{address}\toffline" in self._list_devices():
            self._disconnect_address(address)
        return False

    def _disconnect_address(self, address: str) -> None:
        try:
            subprocess.run(
                [self.adb_path, "disconnect", address],
                capture_output=True,
                timeout=5,
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception:  # noqa: BLE001
            pass

    def _switch_address(self, address: str) -> None:
        host, _, port = address.rpartition(":")
        self.ip = host or self.ip
        try:
            self.port = int(port)
        except ValueError:
            pass
        self.device_addr = f"{self.ip}:{self.port}"

    def discover_wireless_addresses(self) -> List[str]:
        """Список Wi-Fi ADB-сервисов из `adb mdns services`."""
        try:
            result = subprocess.run(
                [self.adb_path, "mdns", "services"],
                capture_output=True,
                text=True,
                timeout=max(5, self.timeout),
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"mDNS-обнаружение недоступно: {exc}")
            return []

        addresses: List[str] = []
        for line in result.stdout.splitlines():
            if "_adb-tls-connect._tcp" not in line and "_adb._tcp" not in line:
                continue
            match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3}):(\d{2,5})", line)
            if match:
                address = f"{match.group(1)}:{match.group(2)}"
                if address not in addresses:
                    addresses.append(address)
        if addresses:
            logger.info(f"mDNS нашёл ADB-устройства: {', '.join(addresses)}")
        return addresses

    def _scan_and_find_port(self) -> Optional[int]:
        """Резервный перебор портов. По умолчанию не используется: медленно и шумно."""
        logger.warning("Запущен перебор портов 30000..50000. Это может занять несколько минут.")

        def check_port(port: int) -> Optional[int]:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.2)
                result = sock.connect_ex((self.ip, port))
                sock.close()
                return port if result == 0 else None
            except Exception:  # noqa: BLE001
                return None

        ports = [5555] + list(range(30000, 50000))
        with ThreadPoolExecutor(max_workers=200) as executor:
            found = [port for port in executor.map(check_port, ports) if port]

        for port in found:
            address = f"{self.ip}:{port}"
            if self._try_connect(address):
                return port
        return None

    def _list_devices(self) -> str:
        try:
            result = subprocess.run(
                [self.adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
            return result.stdout or ""
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"adb devices не выполнилась: {exc}")
            return ""

    def is_connected(self) -> bool:
        """Проверка доступности устройства."""
        return f"{self.device_addr}\tdevice" in self._list_devices()

    def disconnect(self) -> None:
        try:
            subprocess.run(
                [self.adb_path, "disconnect", self.device_addr],
                capture_output=True,
                timeout=self.timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
        except Exception:  # noqa: BLE001
            pass

    def restart_server(self) -> None:
        """Перезапуск локального ADB-сервера — лечит зависшие транспорты."""
        try:
            logger.warning("[ADB] Перезапуск ADB-сервера...")
            subprocess.run(
                [self.adb_path, "kill-server"],
                capture_output=True,
                timeout=self.timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
            time.sleep(1.0)
            subprocess.run(
                [self.adb_path, "start-server"],
                capture_output=True,
                timeout=self.timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
            time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось перезапустить ADB-сервер: {exc}")

    def _update_screen_size(self) -> None:
        """Определение физического разрешения экрана устройства."""
        try:
            result = self._run_cmd(["shell", "wm", "size"], timeout=8)
            output = self._decode(result.stdout)
            # Physical size: 1080x2436 / Override size: ...
            for line in output.splitlines():
                if "physical size:" in line.lower():
                    parts = line.split(":")[-1].strip().lower().split("x")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        self.screen_size = (int(parts[0]), int(parts[1]))
                        return
            for line in output.splitlines():
                if "size:" in line:
                    parts = line.split(":")[-1].strip().lower().split("x")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        self.screen_size = (int(parts[0]), int(parts[1]))
                        return
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось определить разрешение экрана: {exc}")

    # ------------------------------------------------------------------
    # Кадры
    # ------------------------------------------------------------------
    def invalidate_frame(self) -> None:
        self._frame_ts = 0.0

    def frame_age(self) -> float:
        """Возраст последнего кадра в секундах.

        Нужен проверкам перед действием: клик по кадру, которому больше
        нескольких секунд, может попасть уже в другой экран.
        """
        if self._frame is None:
            return float("inf")
        return max(0.0, time.time() - self._frame_ts)

    def get_screenshot(self, fresh: bool = False, attempts: int = 2) -> Optional[np.ndarray]:
        """Захват экрана в память. По умолчанию отдаёт кадр из короткого кэша.

        На Wi-Fi ADB `exec-out screencap -p` отдаёт кадр 3.5-4.5 с (PNG
        кодируется на устройстве и идёт через pipe). Быстрее сделать
        снимок в файл на устройстве и забрать его через `adb pull`
        (1.8-2.2 с). Если pull недоступен — откатываемся на exec-out.
        """
        now = time.time()
        if not fresh and self._frame is not None and (now - self._frame_ts) <= self._frame_ttl:
            return self._frame

        for attempt in range(1, attempts + 1):
            image = self._capture_via_pull()
            if image is None:
                image = self._capture_via_exec()
            if image is None:
                logger.warning(f"Не удалось получить скриншот (попытка {attempt}/{attempts})")
                time.sleep(0.3)
                continue

            self._frame = image
            self._frame_ts = time.time()
            if self.screen_size is None:
                self.screen_size = (image.shape[1], image.shape[0])
            return image
        return None

    def _capture_via_pull(self) -> Optional[np.ndarray]:
        try:
            cap = self._run_cmd(["shell", "screencap", "-p", self._remote_frame], timeout=15)
            if cap.returncode != 0:
                return None
            pull = subprocess.run(
                [self.adb_path, "-s", self.device_addr, "pull", self._remote_frame, str(self._local_frame)],
                capture_output=True,
                timeout=20,
                creationflags=_CREATE_NO_WINDOW,
            )
            if pull.returncode != 0 or not self._local_frame.exists():
                return None
            image = cv2.imread(str(self._local_frame), cv2.IMREAD_COLOR)
            return image
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Захват через pull не удался: {exc}")
            return None

    def _capture_via_exec(self) -> Optional[np.ndarray]:
        try:
            result = self._run_cmd(["exec-out", "screencap", "-p"], timeout=20)
            if not result.stdout:
                return None
            return cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Ошибка при захвате скриншота: {exc}")
            return None

    def save_screenshot(self, path: str, fresh: bool = True) -> bool:
        image = self.get_screenshot(fresh=fresh)
        if image is None:
            return False
        try:
            return bool(cv2.imwrite(path, image))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Не удалось сохранить кадр в {path}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Ввод
    # ------------------------------------------------------------------
    def tap(self, x: int, y: int, jitter_radius: int = 6) -> None:
        """Клик по координатам с гауссовым шумом."""
        real_x, real_y = jitter_point(x, y, radius=jitter_radius)
        self._run_cmd(["shell", "input", "tap", str(real_x), str(real_y)])
        self.invalidate_frame()
        logger.debug(f"Клик по ({real_x}, {real_y}) [исходные: ({x}, {y})]")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 350,
        smooth: bool = True,
    ) -> None:
        """Свайп от (x1, y1) к (x2, y2).

        `input swipe` — это уже один непрерывный жест, который Android
        интерполирует сам. Дробить его на десять коротких свайпов (как
        было раньше) медленно и неестественно, поэтому smooth добавляет
        только лёгкий джиттер концов.
        """
        if smooth:
            x1, y1 = jitter_point(x1, y1, radius=3)
            x2, y2 = jitter_point(x2, y2, radius=3)
        self._run_cmd(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
        self.invalidate_frame()
        logger.debug(f"Свайп ({x1}, {y1}) -> ({x2}, {y2}) за {duration_ms}мс")

    def press_back(self) -> None:
        self._run_cmd(["shell", "input", "keyevent", "4"])
        self.invalidate_frame()
        logger.debug("Нажата кнопка 'Назад'")

    def press_home(self) -> None:
        self._run_cmd(["shell", "input", "keyevent", "3"])
        self.invalidate_frame()
        logger.debug("Нажата системная кнопка 'Домой'")

    def release_touch(self, x: int = 540, y: int = 1200) -> None:
        """Аварийный сброс зажатого тача через motionevent UP."""
        self._run_cmd(["shell", "input", "touchscreen", "motionevent", "UP", str(x), str(y)])
        self.invalidate_frame()
        logger.debug(f"Аварийный сброс тача motionevent UP ({x}, {y})")

    def collapse_statusbar(self) -> None:
        """Сворачивает шторку уведомлений, если она перекрывает HUD."""
        try:
            self._run_cmd(["shell", "cmd", "statusbar", "collapse"], timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def is_screen_on(self) -> bool:
        """Включён ли экран устройства (важно для долгого автономного прогона)."""
        try:
            result = self._run_cmd(["shell", "dumpsys", "power"], timeout=6)
            text = result.stdout or ""
            if "mWakefulness=Awake" in text:
                return True
            if "mWakefulness=Asleep" in text or "mWakefulness=Dozing" in text:
                return False
            # Резерв для других версий Android.
            return "state=ON" in text
        except Exception:  # noqa: BLE001
            return True

    def wake_screen(self) -> bool:
        """Будит экран и снимает простой keyguard (если нет PIN/графического ключа)."""
        if self.is_screen_on():
            return True
        logger.info("[ADB] Экран выключен — будим устройство.")
        try:
            self._run_cmd(["shell", "input", "keyevent", "KEYCODE_WAKEUP"], timeout=5)
            time.sleep(0.6)
            self._run_cmd(["shell", "wm", "dismiss-keyguard"], timeout=5)
            time.sleep(0.6)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось разбудить экран: {exc}")
        self.invalidate_frame()
        return self.is_screen_on()

    def keep_screen_awake(self, enabled: bool = True) -> None:
        """Не даёт экрану гаснуть на время автономной работы.

        Wi-Fi ADB отваливается, когда устройство уходит в сон, и разбудить
        его удалённо уже нельзя. Поэтому на старте фиксируем экран включённым:
        `svc power stayon` + максимальный `screen_off_timeout`.
        """
        try:
            if enabled:
                if self._orig_screen_off_timeout is None:
                    result = self._run_cmd(
                        ["shell", "settings", "get", "system", "screen_off_timeout"], timeout=5
                    )
                    self._orig_screen_off_timeout = self._decode(result.stdout).strip() or "30000"
                if self._orig_stay_on is None:
                    result = self._run_cmd(
                        ["shell", "settings", "get", "global", "stay_on_while_plugged_in"], timeout=5
                    )
                    self._orig_stay_on = self._decode(result.stdout).strip() or "0"

                self._run_cmd(["shell", "svc", "power", "stayon", "true"], timeout=5)
                # 3 = AC | USB; на многих прошивках этого достаточно.
                self._run_cmd(
                    ["shell", "settings", "put", "global", "stay_on_while_plugged_in", "3"], timeout=5
                )
                # Максимальный таймаут — страховка, если зарядки нет.
                self._run_cmd(
                    ["shell", "settings", "put", "system", "screen_off_timeout", "2147483647"],
                    timeout=5,
                )
                logger.info("[ADB] Экран закреплён включённым (stayon + screen_off_timeout).")
            else:
                self._run_cmd(["shell", "svc", "power", "stayon", "false"], timeout=5)
                if self._orig_stay_on is not None:
                    self._run_cmd(
                        ["shell", "settings", "put", "global", "stay_on_while_plugged_in",
                         self._orig_stay_on], timeout=5,
                    )
                if self._orig_screen_off_timeout is not None:
                    self._run_cmd(
                        ["shell", "settings", "put", "system", "screen_off_timeout",
                         self._orig_screen_off_timeout], timeout=5,
                    )
                logger.info("[ADB] Настройки сна экрана восстановлены.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось закрепить экран включённым: {exc}")

    def dismiss_heads_up(self) -> None:
        """Убирает системную heads-up плашку (например, 'Отладка по Wi-Fi')."""
        try:
            # Короткий свайп вверх по центру плашки закрывает её, не открывая настройки.
            self._run_cmd(["shell", "input", "swipe", "540", "220", "540", "10", "120"], timeout=5)
            self._run_cmd(["shell", "cmd", "statusbar", "collapse"], timeout=5)
            self.invalidate_frame()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Не удалось убрать heads-up уведомление: {exc}")

    # ------------------------------------------------------------------
    # Приложение
    # ------------------------------------------------------------------
    def launch_game(self, package_name: str, activity_name: str) -> bool:
        logger.info(f"Запуск игры {package_name}...")
        self._run_cmd(["shell", "am", "start", "-n", f"{package_name}/{activity_name}"])
        self.invalidate_frame()
        time.sleep(3)
        return self.is_game_foreground(package_name)

    def force_stop(self, package_name: str) -> bool:
        """Принудительно останавливает приложение.

        Используется только как аварийное восстановление, когда экран
        боя/выбора навыка не реагирует на штатные кнопки. Данные игры не
        изменяются; текущий бой может быть прерван.
        """
        logger.warning(f"Принудительная остановка {package_name}...")
        try:
            self._run_cmd(["shell", "am", "force-stop", package_name])
            self.invalidate_frame()
            time.sleep(1.0)
            return not self.is_game_foreground(package_name)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Не удалось остановить {package_name}: {exc}")
            return False

    def is_game_foreground(self, package_name: str) -> bool:
        """Проверяет, что игра находится именно в активном фокусе (на переднем плане)."""
        try:
            result = self._run_cmd(["shell", "dumpsys", "window"], timeout=max(6, self.timeout))
            output = self._decode(result.stdout)
            focused_lines = [
                line for line in output.splitlines()
                if "mCurrentFocus" in line or "mFocusedApp" in line
            ]
            if focused_lines:
                return any(package_name in line for line in focused_lines)
            return package_name in output
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Не удалось проверить foreground: {exc}")
            return False

    def is_phone_call_active(self) -> bool:
        """Проверяет наличие активного телефонного вызова (звонок, разговор).

        Критическая защита: если пользователь разговаривает по телефону,
        любые автоматические клики и переключения приложений блокируются,
        чтобы не сбросить звонок и не мешать разговору.
        """
        try:
            result = self._run_cmd(
                ["shell", "dumpsys", "telephony.registry"], timeout=max(5, self.timeout)
            )
            output = self._decode(result.stdout)
            for line in output.splitlines():
                if "mCallState=" in line:
                    val = line.split("mCallState=")[-1].strip()
                    # 1 = CALL_STATE_RINGING, 2 = CALL_STATE_OFFHOOK (активный разговор)
                    if val.isdigit() and int(val) in (1, 2):
                        return True
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Не удалось проверить статус звонка: {exc}")
        return False

    # ------------------------------------------------------------------
    # Удобные обёртки
    # ------------------------------------------------------------------
    def human_pause(self, min_sec: float = 0.6, max_sec: float = 1.4) -> None:
        sleep_human(min_sec, max_sec)
