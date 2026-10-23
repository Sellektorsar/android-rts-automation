"""Живой прогон BuildingUpgradeTask по 19 зданиям квартала.

Запускает только улучшение зданий и закрыватель попапов, поэтому
строительство и остальные городские задачи не мешают. Тип и уровни
каждой точки пишутся в memory_bank/building_spots.json.

Пример:
    .venv/Scripts/python.exe tools/run_building_upgrade.py --duration 900
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot_engine import BotEngine  # noqa: E402
from src.tasks.building_upgrade import BuildingUpgradeTask  # noqa: E402
from src.tasks.city_harvest import CityHarvestTask  # noqa: E402
from src.tasks.popup_closer import PopupCloserTask  # noqa: E402


def _setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        logs_dir / "building_upgrade_live.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Живой прогон улучшения зданий")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=int, default=900, help="Лимит времени, секунды")
    parser.add_argument("--interval", type=float, default=2.5, help="Интервал между попытками")
    args = parser.parse_args()

    _setup_logging()

    engine = BotEngine(config_path=args.config)
    engine.tasks = [
        task
        for task in engine.tasks
        if isinstance(task, (BuildingUpgradeTask, PopupCloserTask, CityHarvestTask))
    ]
    upgrade = next(
        (task for task in engine.tasks if isinstance(task, BuildingUpgradeTask)), None
    )
    if upgrade is None:
        logger.error("BuildingUpgradeTask не создана: проверьте tasks.building_upgrade.enabled.")
        return
    upgrade.interval_sec = args.interval

    logger.info(
        f"Живой прогон улучшения: точек в конфиге {len(upgrade.spots)}; "
        f"лимит {args.duration}с. Паспорта: {upgrade.KNOWLEDGE_PATH}."
    )
    engine.start(duration_sec=args.duration)


if __name__ == "__main__":
    main()
