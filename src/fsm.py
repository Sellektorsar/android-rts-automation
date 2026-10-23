"""
Конечный автомат состояний игры (FSM) с поддержкой стека всплывающих окон (Popup Interrupt Stack).
Основан на архитектуре 4x-game-agent.
Позволяет боту точно знать текущий экран игры, автоматически изолировать всплывающие акции,
сохранять предыдущее состояние в стек и возвращаться к нему после закрытия модалки.
"""

import time
from typing import List, Optional, Set
from loguru import logger


class GameState:
    """Константы состояний экранов War & Order."""
    UNKNOWN = "unknown"
    LOADING = "loading"
    
    # Основные экраны
    CITY_VIEW = "city_view"             # Главный экран замка и города
    WORLD_MAP = "world_map"             # Карта королевства (фермы, монстры, враги)
    CAMPAIGN_MAP = "campaign_map"       # Карта этапов кампании ("Спаситель порядка")
    ROYAL_ROAD = "royal_road"           # Список сюжетных заданий главы (Королевская Дорога)
    QUESTS_MENU = "quests_menu"         # Ежедневные и сюжетные задания
    STAGE_BRIEFING = "stage_briefing"   # Карточка этапа с кнопкой "Испытание"
    TUTORIAL_BATTLE = "tutorial_battle" # Бой на арене с волнами монстров
    SKILL_SELECTION = "skill_selection" # Выбор карт навыков при повышении уровня
    VICTORY_SCREEN = "victory_screen"   # Экран победы ("Задание выполнено")
    STORY_DIALOGUE = "story_dialogue"   # Диалоговый свиток (Белла / советник)
    
    # Модальные окна и всплывающие акции
    POPUP_OFFER = "popup_offer"         # Донатный/акционный баннер
    POPUP_GENERIC = "popup_generic"     # Любое модальное окно с крестиком закрытия
    POPUP_REWARD = "popup_reward"       # Окно получения сундука/награды
    
    # Внутренние меню
    BUILDING_MENU = "building_menu"     # Радиальное меню выбранного здания
    BUILDING_UPGRADE = "building_upgrade"  # Панель улучшения (требования + кнопка)
    PLOT_BUILD = "plot_build"           # Экран постройки на пустой ячейке
    RESEARCH_MENU = "research_menu"     # Институт / дерево исследований
    TRAINING_MENU = "training_menu"     # Казармы / найм войск
    ALLIANCE_MENU = "alliance_menu"     # Меню альянса / помощь


POPUP_STATES: Set[str] = {
    GameState.POPUP_OFFER,
    GameState.POPUP_GENERIC,
    GameState.POPUP_REWARD,
}


class GameFSM:
    """Управление состояниями экранов и стеком прерываний."""

    def __init__(self):
        self.current_state: str = GameState.UNKNOWN
        self.state_stack: List[str] = []
        self.state_history: List[dict] = []
        self.stuck_counter: int = 0
        self.last_state_change: float = time.time()

    def update(self, new_state: str) -> None:
        """Обновление текущего состояния экрана."""
        old_state = self.current_state

        if new_state == old_state:
            self.stuck_counter += 1
            return

        self.stuck_counter = 0
        self.last_state_change = time.time()

        self.state_history.append({
            "from": old_state,
            "to": new_state,
            "timestamp": time.time(),
        })
        if len(self.state_history) > 30:
            self.state_history.pop(0)

        # Если появилось всплывающее окно — сохраняем исходное состояние в стек
        if new_state in POPUP_STATES and old_state not in POPUP_STATES and old_state != GameState.UNKNOWN:
            self.state_stack.append(old_state)
            logger.info(f"[FSM] POPUP обнаружен: {new_state} (фоновое состояние: {old_state} сохранено в стек)")
        # Если модалка закрыта и вернулись к нормальному экрану — снимаем из стека
        elif old_state in POPUP_STATES and new_state not in POPUP_STATES:
            expected = self.state_stack.pop() if self.state_stack else None
            logger.info(f"[FSM] Popup закрыт. Возврат к: {new_state} (ожидалось: {expected})")
        else:
            logger.info(f"[FSM] Переход: {old_state} -> {new_state}")

        self.current_state = new_state

    def is_popup(self) -> bool:
        """Находится ли бот сейчас на всплывающем модальном окне."""
        return self.current_state in POPUP_STATES

    def is_stuck(self, threshold: int = 8) -> bool:
        """Проверка зависания в одном и том же состоянии."""
        return self.stuck_counter >= threshold

    def get_resuming_state(self) -> Optional[str]:
        """Получить состояние, к которому нужно вернуться после закрытия окна."""
        if self.state_stack:
            return self.state_stack[-1]
        return None
