"""
Модуль саморефлексии и обучения на ошибках (Self-Reflection).
Основан на паттернах CRADLE и Reflexion из фреймворка 4x-game-agent.
Фиксирует успешные и неудачные действия бота, позволяет адаптировать координаты
и стратегии при повторении задач.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger


class ReflectionLog:
    """Хранилище истории успехов и неудач с адаптивным анализом."""

    def __init__(self, filepath: str = "memory_bank/reflections.json", max_entries: int = 300):
        self.filepath = Path(filepath)
        self.max_entries = max_entries
        self.entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось прочитать {self.filepath}: {e}. Инициализация нового списка.")
                return []
        return []

    def _save(self) -> None:
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения {self.filepath}: {e}")

    def record_success(self, task: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Запись успешно выполненного действия."""
        entry = {
            "type": "success",
            "task": task,
            "details": details or {},
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self._save()
        logger.debug(f"[Reflection] Успех: {task}")

    def record_failure(
        self,
        task: str,
        error: str,
        analysis: str,
        fix_applied: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Запись ошибки, причины сбоя и примененного исправления."""
        entry = {
            "type": "failure",
            "task": task,
            "error": error,
            "analysis": analysis,
            "fix_applied": fix_applied,
            "details": details or {},
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self._save()
        logger.warning(f"[Reflection] Сбой: {task} | Ошибка: {error} | Анализ: {analysis}")

    def get_recent_failures(self, task_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Получить последние ошибки для конкретной задачи или типа задачи."""
        results = []
        for entry in reversed(self.entries):
            if entry.get("type") == "failure":
                if entry.get("task", "").startswith(task_name):
                    results.append(entry)
                    if len(results) >= limit:
                        break
        return results

    def success_rate(self, task_prefix: str, window: int = 30) -> float:
        """Расчет процента успешности задачи за последние N вызовов."""
        return self.task_stats(task_prefix, window=window)["rate"]

    def task_stats(self, task_prefix: str, window: int = 30) -> Dict[str, Any]:
        """Статистика задачи: всего записей, успехи, сбои и доля успехов.

        Нужна для адаптивного интервала: если задача стабильно падает,
        движок увеличивает паузу вместо повторения одного и того же сбоя.
        """
        relevant = [
            e for e in self.entries[-window:] if e.get("task", "").startswith(task_prefix)
        ]
        total = len(relevant)
        successes = sum(1 for e in relevant if e.get("type") == "success")
        failures = total - successes
        return {
            "total": total,
            "successes": successes,
            "failures": failures,
            "rate": (successes / total) if total else 1.0,
        }
