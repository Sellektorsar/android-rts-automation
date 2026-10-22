"""
Быстрый поток кадров с Android-устройства.

Зачем: `adb pull` готового PNG занимает ~1.8 с, этого мало для боя.
Здесь используется штатный `screenrecord` (H.264) и локальный `ffmpeg`,
который декодирует поток в кадры BGR без промежуточных файлов.

Модуль не запускает бой и не нажимает кнопки. Это только транспорт кадров.
Если `screenrecord`/`ffmpeg` недоступны, вызывающий код должен откатиться
на `AdbClient.get_screenshot()`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class FastCaptureStats:
    frames: int = 0
    started_at: float = 0.0
    last_frame_at: float = 0.0
    restarts: int = 0

    @property
    def fps(self) -> float:
        elapsed = self.last_frame_at - self.started_at
        return self.frames / elapsed if elapsed > 0 else 0.0


class FastCapture:
    """Поток кадров через `adb exec-out screenrecord` + `ffmpeg`."""

    def __init__(
        self,
        adb_path: str,
        serial: str,
        *,
        width: int = 540,
        height: int = 1218,
        record_time_limit: int = 180,
        ffmpeg_path: Optional[str] = None,
    ):
        self.adb_path = adb_path
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.record_time_limit = int(record_time_limit)
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
        self._adb_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._pending = b""
        self.stats = FastCaptureStats()

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg_path) and shutil.which(self.adb_path) is not None

    def start(self) -> bool:
        """Запуск фонового потока. Возвращает False, если нет инструментов."""
        if not self.available:
            logger.warning(
                "[FastCapture] Нет ffmpeg или adb — быстрый поток недоступен."
            )
            return False
        self.stop()

        adb_cmd = [
            self.adb_path,
            "-s",
            self.serial,
            "exec-out",
            "screenrecord",
            "--output-format=h264",
            f"--time-limit={self.record_time_limit}",
            "-",
        ]
        # ВАЖНО: не использовать `-fflags nobuffer` для raw H.264:
        # ffmpeg в этом режиме не отдаёт ни одного кадра (проверено на
        # локальном файле и на живом потоке этого устройства).
        ffmpeg_cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-flags",
            "low_delay",
            "-probesize",
            "500000",
            "-analyzeduration",
            "0",
            "-i",
            "pipe:0",
            "-vf",
            f"scale={self.width}:-2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._adb_proc = subprocess.Popen(
                adb_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            assert self._adb_proc.stdout is not None
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self._adb_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            # ffmpeg владеет копией stdout ADB; закрываем свою ссылку,
            # чтобы не блокировать пайп.
            self._adb_proc.stdout.close()
            self._pending = b""
            self.stats = FastCaptureStats(started_at=time.time())
            logger.info(
                f"[FastCapture] Поток запущен: {self.width}x{self.height}, "
                f"screenrecord до {self.record_time_limit}с."
            )
            return True
        except Exception as exc:
            logger.error(f"[FastCapture] Не удалось запустить поток: {exc}")
            self.stop()
            return False

    def read(self, timeout_sec: float = 2.0) -> Optional[np.ndarray]:
        """Прочитать один кадр. None — поток закончился/не запущен."""
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
            return None

        deadline = time.time() + max(0.1, timeout_sec)
        while len(self._pending) < self.frame_bytes and time.time() < deadline:
            chunk = self._ffmpeg_proc.stdout.read(self.frame_bytes - len(self._pending))
            if not chunk:
                return None
            self._pending += chunk

        if len(self._pending) < self.frame_bytes:
            # Частичный кадр сохраняем до следующего вызова, чтобы не
            # сбить выравнивание потока.
            return None

        raw = self._pending[: self.frame_bytes]
        self._pending = self._pending[self.frame_bytes :]
        frame = np.frombuffer(raw, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        self.stats.frames += 1
        self.stats.last_frame_at = time.time()
        return frame

    def restart(self) -> bool:
        """Перезапуск после `--time-limit` или обрыва потока."""
        self.stats.restarts += 1
        return self.start()

    def stop(self) -> None:
        for proc in (self._ffmpeg_proc, self._adb_proc):
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._ffmpeg_proc = None
        self._adb_proc = None

    def __enter__(self) -> "FastCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
