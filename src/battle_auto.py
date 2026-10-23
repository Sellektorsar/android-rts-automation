"""
Боевая логика автопилота (BattleAuto).

Чистый контроллер фаз боя: в каком состоянии находится герой, когда
берём управление джойстиком на себя, когда нужен выбор/подтверждение
навыка, когда бой завершён исходом.

Здесь нет Android-зависимостей и CV-алгоритмов — решения принимаются
по состояниям классификатора экрана и времени. Визуальные действия
(поиск кнопок, тапы) остаются в задаче (`src/tasks/battle.py`), чтобы
автопилот оставался легко тестируемым.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from src.fsm import GameState


class BattlePhase:
    """Фазы боя, через которые проходит автопилот."""

    IN_BATTLE = "in_battle"
    SKILL_SELECTION = "skill_selection"
    SKILL_CONFIRMED = "skill_confirmed"
    VICTORY = "victory"
    DEFEAT = "defeat"
    EXIT = "exit"


# (x1, y1, x2, y2, duration_ms) — круговой кайт вокруг джойстика.
DEFAULT_MOVE_CYCLE: List[Tuple[int, int, int, int, int]] = [
    (540, 1980, 540, 1720, 1100),
    (540, 1980, 780, 1850, 1100),
    (540, 1980, 780, 2100, 1100),
    (540, 1980, 540, 2240, 1100),
    (540, 1980, 300, 2100, 1100),
    (540, 1980, 300, 1850, 1100),
]


class BattleAuto:
    """Контроллер последовательности действий в бою."""

    EXIT_STATES = frozenset(
        {
            GameState.CITY_VIEW,
            GameState.WORLD_MAP,
            GameState.CAMPAIGN_MAP,
            GameState.ROYAL_ROAD,
            GameState.QUESTS_MENU,
            GameState.STAGE_BRIEFING,
            GameState.LOADING,
            GameState.ALLIANCE_MENU,
            GameState.BUILDING_MENU,
            GameState.BUILDING_UPGRADE,
            GameState.TRAINING_MENU,
            GameState.RESEARCH_MENU,
            GameState.STORY_DIALOGUE,
        }
    )

    def __init__(
        self,
        move_cycle: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        self.phase: str = BattlePhase.IN_BATTLE
        self.move_cycle: List[List[int]] = [
            [int(value) for value in step] for step in (move_cycle or DEFAULT_MOVE_CYCLE)
        ]
        self.move_index: int = 0
        self.last_move_at: float = float("-inf")
        self.move_cooldown_sec: float = 0.8
        self.skill_confirmed: bool = False

    # ------------------------------------------------------------------
    # Фазы
    # ------------------------------------------------------------------
    def observe(self, state: str) -> None:
        """Обновить текущую фазу по классифицированному экрану."""
        if state == GameState.TUTORIAL_BATTLE:
            self.phase = BattlePhase.IN_BATTLE
            self.skill_confirmed = False
        elif state == GameState.SKILL_SELECTION:
            self.phase = BattlePhase.SKILL_SELECTION
        elif state == GameState.VICTORY_SCREEN:
            self.phase = BattlePhase.VICTORY
        elif state in self.EXIT_STATES:
            self.phase = BattlePhase.EXIT

    def handle_outcome(self, outcome: str) -> None:
        """Зафиксировать исход боя (victory/defeat/exit)."""
        if outcome == "victory":
            self.phase = BattlePhase.VICTORY
        elif outcome == "defeat":
            self.phase = BattlePhase.DEFEAT
        elif outcome == "exit":
            self.phase = BattlePhase.EXIT

    # ------------------------------------------------------------------
    # Действия
    # ------------------------------------------------------------------
    def next_move(
        self, now: float = 0.0
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """Следующий свайп джойстика (круговой кайт). None — cooldown или не в фазе."""
        if self.phase != BattlePhase.IN_BATTLE:
            return None
        if now - self.last_move_at < self.move_cooldown_sec:
            return None
        step = self.move_cycle[self.move_index % len(self.move_cycle)]
        self.move_index += 1
        self.last_move_at = now
        return (int(step[0]), int(step[1]), int(step[2]), int(step[3]), int(step[4]))

    def mark_skill_sent(self) -> None:
        """Навык выбран и отправлен — возвращаемся к управлению джойстиком."""
        self.skill_confirmed = True
        self.phase = BattlePhase.IN_BATTLE

    def reset(self) -> None:
        """Сброс состояния автопилота (например, перед повтором этапа)."""
        self.phase = BattlePhase.IN_BATTLE
        self.move_index = 0
        self.last_move_at = float("-inf")
        self.skill_confirmed = False
