"""Живой прогон CityBuildTask: застройка пустых ячеек в правом нижнем квартале.

Запускает только задачу постройки и закрыватель попапов, поэтому остальные
городские задачи не мешают калибровке. Каждый успешный акт пишется в
memory_bank/built_plots.json.

Пример:
    .venv/Scripts/python.exe tools/run_city_build.py --duration 540
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_engine import BotEngine  # noqa: E402
from src.tasks.city_build import CityBuildTask  # noqa: E402
from src.tasks.popup_closer import PopupCloserTask  # noqa: E402


def _setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        logs_dir / "city_build_live.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Живой прогон постройки зданий")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=int, default=540, help="Лимит времени, секунды")
    parser.add_argument("--interval", type=float, default=3.0, help="Интервал между попытками")
    args = parser.parse_args()

    _setup_logging()

    engine = BotEngine(config_path=args.config)
    engine.tasks = [
        task for task in engine.tasks if isinstance(task, (CityBuildTask, PopupCloserTask))
    ]
    build = next((task for task in engine.tasks if isinstance(task, CityBuildTask)), None)
    if build is None:
        logger.error("CityBuildTask не создана: проверьте tasks.city_build.enabled в конфиге.")
        return
    build.interval_sec = args.interval

    logger.info(
        f"Живой прогон постройки: ячеек в конфиге {len(build.config_plots)}, "
        f"уже построено {len(build.built)}; лимит {args.duration}с."
    )
    engine.start(duration_sec=args.duration)


if __name__ == "__main__":
    main()
