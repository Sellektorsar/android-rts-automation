"""Боевой автопилот (BattleTask) — выживание на этапе «Спаситель порядка»."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

from src.battle_auto import BattleAuto
from src.fsm import GameState
from src.fast_screen import FastScreenAnalyzer
from src.perception import ScreenClassifier
from src.tasks.base_task import BaseTask


class BattleTask(BaseTask):
    """Автопилот боя: джойстик, выбор навыка, экран победы.

    Сценарий этапа 7 («Выжить в течение 120 сек.»), проверенный зондами:

    1. Детекция боя через классификатор (`battle_pause` / `battle_pause_fast`).
    2. В фазе боя — джойстик по круговому кайту (`BattleAuto`).
    3. Оверлей выбора навыка (`skill_selection`): сначала зелёная
       `btn_skill_confirm`, иначе первая карточка → подтверждение.
    4. Экран победы фиксируется как исход боя; выход из боя и повтор
       этапа — через обычную навигацию движка (без «Повторить»).

    Задача не нажимает Back в бою и не трогает кнопки оплаты/покупок.
    """

    SKILL_CONFIRM_TEMPLATE = "btn_skill_confirm"
    BATTLE_EXIT_TEMPLATE = "btn_battle_exit"
    VICTORY_TEMPLATE = "victory_banner"

    # Регион и параметры подтверждения навыка (доли кадра + масштаб).
    SKILL_CONFIRM_REGION = (0.25, 0.65, 0.55, 0.20)
    SKILL_CONFIRM_SCALE = 0.5
    SKILL_CONFIRM_THRESHOLD = 0.90
    # Первая карточка навыка (бесплатный выбор) — центр карточки.
    SKILL_CARD_X_FRAC = 0.17
    SKILL_CARD_Y_FRAC = 0.45
    SKILL_TIMEOUT_SEC = 7.0

    def __init__(
        self,
        client,
        vision,
        interval_sec: int = 30,
        reflection=None,
        battle: Optional[BattleAuto] = None,
        world=None,
    ):
        super().__init__(
            "BattleAuto",
            client,
            vision,
            interval_sec=interval_sec,
            reflection=reflection,
        )
        self.classifier = ScreenClassifier(vision, allow_ocr=False)
        self.fast = FastScreenAnalyzer()
        self.battle = battle if battle is not None else BattleAuto()
        self.world = world

    def execute(self, screen: np.ndarray) -> bool:
        if not self.require_templates(
            [
                self.SKILL_CONFIRM_TEMPLATE,
                self.BATTLE_EXIT_TEMPLATE,
                self.VICTORY_TEMPLATE,
            ]
        ):
            return False
        screen = self.ensure_screen(screen)
        if screen is None:
            return False

        state = self.classifier.classify(screen, allow_ocr=False).state
        self.battle.observe(state)

        if state == GameState.SKILL_SELECTION:
            return self._handle_skill_selection(screen)

        if state == GameState.TUTORIAL_BATTLE:
            if self.fast.overlay_level(screen) > 0.5:
                # Оверлей поверх боя (выбор/результат) — джойстик не трогаем.
                return False
            move = self.battle.next_move(now=time.time())
            if move is None:
                return False
            x1, y1, x2, y2, duration_ms = move
            logger.info(f"[BattleAuto] Кайт: ({x1}, {y1}) -> ({x2}, {y2}) за {duration_ms} мс.")
            self.client.swipe(x1, y1, x2, y2, duration_ms=duration_ms)
            self.mark_executed()
            return True

        if state == GameState.VICTORY_SCREEN:
            # Экран победы фиксируется как исход боя; выход из боя и
            # повтор этапа оформляет обычную навигацию движка (без «Повторить»).
            self.battle.handle_outcome("victory")
            if self.world is not None:
                self.world.record_event("battle:victory", "stage7")
            return False

        return False

    # ------------------------------------------------------------------
    # Выбор навыка поверх боя
    # ------------------------------------------------------------------
    def _handle_skill_selection(self, screen: np.ndarray) -> bool:
        """Подтверждает выбор навыка; Back не нажимаем, пока не выбран."""
        started = time.time()
        card_tapped = False
        while time.time() - started < self.SKILL_TIMEOUT_SEC:
            height, width = screen.shape[:2]
            region = (
                int(width * self.SKILL_CONFIRM_REGION[0]),
                int(height * self.SKILL_CONFIRM_REGION[1]),
                int(width * self.SKILL_CONFIRM_REGION[2]),
                int(height * self.SKILL_CONFIRM_REGION[3]),
            )
            match = self.vision.find_template(
                screen,
                self.SKILL_CONFIRM_TEMPLATE,
                threshold=self.SKILL_CONFIRM_THRESHOLD,
                region=region,
                scale=self.SKILL_CONFIRM_SCALE,
            )
            if match is not None:
                x, y, confidence = int(match[0]), int(match[1]), float(match[2])
                logger.info(
                    f"[BattleAuto] Подтверждаем навык в оверлее ({x}, {y}), conf={confidence:.2f}."
                )
                self.client.tap(x, y)
                time.sleep(1.5)
                self.battle.mark_skill_sent()
                self.mark_executed()
                return True

            if not card_tapped:
                card_x = int(width * self.SKILL_CARD_X_FRAC)
                card_y = int(height * self.SKILL_CARD_Y_FRAC)
                logger.info(f"[BattleAuto] Выбираем первую карточку навыка ({card_x}, {card_y}).")
                self.client.tap(card_x, card_y)
                card_tapped = True
                time.sleep(0.8)
                screen = self.client.get_screenshot(fresh=True)
                if screen is None:
                    return False
                continue

            time.sleep(0.4)

        self.report_failure("skill_selection_unhandled", "не удалось подтвердить выбор навыка за 7 сек")
        return False
