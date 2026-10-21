"""
Точка входа для запуска бота War & Order.
"""

from loguru import logger
from src.bot_engine import BotEngine


def main():
    logger.add(
        "bot.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )
    engine = BotEngine(config_path="config.yaml")
    engine.start()


if __name__ == "__main__":
    main()
