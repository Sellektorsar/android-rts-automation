"""
Калибровка шаблонов: снимок экрана или готовый кадр -> вырезка -> проверка.

Примеры:
    # Снять текущий экран и сохранить полный кадр
    python tools/capture_template.py --screenshot-only

    # Вырезать шаблон из свежего кадра устройства
    python tools/capture_template.py --name title_alliance --box 462 132 166 72

    # Вырезать из уже сохранённого кадра (устройство не нужно)
    python tools/capture_template.py --frame logs/inspect/alliance.png \
        --name title_alliance --box 462 132 166 72

    # Дополнительно проверить, что шаблон не срабатывает на других экранах
    python tools/capture_template.py --frame logs/inspect/alliance.png \
        --name title_alliance --box 462 132 166 72 --test logs/inspect/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient


def capture_from_device(require_game: bool = True) -> np.ndarray | None:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dev_cfg = config["device"]
    client = AdbClient(
        adb_path=dev_cfg["adb_path"],
        ip=dev_cfg["ip"],
        port=dev_cfg["port"],
        timeout=dev_cfg.get("timeout_sec", 10),
        frame_ttl=0.0,
    )
    if not client.connect(allow_discovery=True):
        logger.error("Не удалось подключиться к устройству.")
        return None
    client.wake_screen()
    client.collapse_statusbar()

    # Приватность: не сохраняем кадры чужих приложений.
    if require_game and not client.is_game_foreground(config["game"]["package"]):
        logger.error("Игра не на переднем плане — снимок не делается.")
        return None

    frame = client.get_screenshot(fresh=True)
    if frame is None:
        logger.error("Не удалось получить скриншот.")
    return frame


def match_score(screen: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return float(max_val), max_loc


def main() -> None:
    parser = argparse.ArgumentParser(description="Захват и проверка шаблонов UI")
    parser.add_argument("--screenshot-only", action="store_true", help="Только сохранить полный кадр")
    parser.add_argument("--frame", type=str, help="Готовый кадр вместо захвата с устройства")
    parser.add_argument("--name", type=str, help="Имя шаблона (без .png)")
    parser.add_argument(
        "--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
        help="Прямоугольник вырезки в пикселях кадра: x y w h",
    )
    parser.add_argument("--out-dir", type=str, default="assets/templates", help="Папка шаблонов")
    parser.add_argument(
        "--test", type=str, nargs="*", default=[],
        help="Кадры для проверки ложных срабатываний (glob/пути)",
    )
    parser.add_argument(
        "--allow-other-app", action="store_true",
        help="Диагностика: разрешить снимок другого приложения (не сохраняется)",
    )
    args = parser.parse_args()

    if args.frame:
        screen = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if screen is None:
            logger.error(f"Не удалось открыть кадр: {args.frame}")
            return
        from_device = False
    else:
        screen = capture_from_device(require_game=not args.allow_other_app)
        if screen is None:
            return
        from_device = True

    if from_device and args.allow_other_app:
        logger.warning("Снимок другого приложения не сохраняется.")
        return

    stamp = Path("logs/inspect")
    if from_device:
        stamp.mkdir(parents=True, exist_ok=True)
        full_path = stamp / "template_source.png"
        cv2.imwrite(str(full_path), screen)
        logger.info(f"Полный кадр: {full_path} ({screen.shape[1]}x{screen.shape[0]})")
    else:
        logger.info(f"Исходный кадр: {args.frame} ({screen.shape[1]}x{screen.shape[0]})")

    if args.screenshot_only:
        return
    if not args.name or not args.box:
        logger.info("Укажите --name и --box <x> <y> <w> <h> для вырезки шаблона.")
        return

    x, y, w, h = args.box
    if x < 0 or y < 0 or x + w > screen.shape[1] or y + h > screen.shape[0]:
        logger.error("Прямоугольник выходит за границы кадра.")
        return

    crop = screen[y: y + h, x: x + w]
    if crop.size == 0:
        logger.error("Пустая вырезка.")
        return

    out_path = Path(args.out_dir) / f"{args.name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    logger.success(f"Шаблон сохранён: {out_path} ({w}x{h} px)")

    # Проверка на исходном кадре: шаблон обязан найтись сам в себе.
    self_score, self_loc = match_score(screen, crop)
    logger.info(f"Исходный кадр: {self_score:.3f} @ {self_loc}")

    # Проверка на соседних экранах: там шаблон не должен давать высокий score.
    for pattern in args.test:
        if any(ch in pattern for ch in "*?["):
            candidates = sorted(Path().glob(pattern))
        else:
            candidates = [Path(pattern)]
        for path in candidates:
            if not path.exists() or path.resolve() == Path(args.frame or "").resolve():
                continue
            other = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if other is None:
                continue
            score, loc = match_score(other, crop)
            level = "!! ЛОЖНОЕ" if score >= 0.75 else ("? проверь" if score >= 0.65 else "ok")
            logger.info(f"{path.name}: {score:.3f} @ {loc} {level}")


if __name__ == "__main__":
    main()
