"""
Точка входа для запуска бота War & Order.

Примеры:
    python main.py --observe              # только смотреть и классифицировать
    python main.py --observe --duration 60
    python main.py --once                 # один снимок состояния + HUD
    python main.py                        # обычный автономный режим
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.bot_engine import BotEngine


def _setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        logs_dir / "bot.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Автономный бот War & Order")
    parser.add_argument("--config", default="config.yaml", help="Путь к конфигурации")
    parser.add_argument("--observe", action="store_true", help="Только наблюдение, без действий")
    parser.add_argument("--once", action="store_true", help="Сделать один снимок состояния и выйти")
    parser.add_argument("--duration", type=int, default=0, help="Ограничить время работы, секунды")
    args = parser.parse_args()

    _setup_logging()

    if args.once:
        engine = BotEngine(config_path=args.config, observe_only=True)
        print(engine.dump_snapshot())
        return

    engine = BotEngine(config_path=args.config, observe_only=args.observe)
    engine.start(duration_sec=args.duration)


if __name__ == "__main__":
    main()
