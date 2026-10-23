"""
Главный движок бота (BotEngine).

Реальный автономный цикл:
1. Подключение и переподключение ADB.
2. Захват кадра и классификация текущего экрана.
3. Обновление FSM, WorldModel и HUD.
4. Обработка всплывающих окон и наград.
5. Запуск задач, разрешённых для текущего экрана.
6. Watchdog: восстановление при зависании и аварийная остановка
   на экранах оплаты.

Режим `observe_only` ничего не нажимает: только смотрит, классифицирует
и пишет лог. Это безопасный способ проверить восприятие на живом устройстве.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml
from loguru import logger

from src.adb_client import AdbClient
from src.fast_capture import FastCapture
from src.fsm import GameFSM, GameState
from src.human_input import sleep_human
from src.perception import HudParser, HudState, ScreenClassifier, looks_like_payment
from src.reflection import ReflectionLog
from src.tasks.alliance import AllianceTask
from src.tasks.base_task import BaseTask
from src.tasks.battle import BattleTask
from src.tasks.building_upgrade import BuildingUpgradeTask
from src.tasks.chapter_quest import ChapterQuestTask
from src.tasks.city_build import CityBuildTask
from src.tasks.city_gifts import CityGiftsTask
from src.tasks.city_harvest import CityHarvestTask
from src.tasks.popup_closer import PopupCloserTask
from src.tasks.quest_rewards import QuestRewardsTask
from src.tasks.research import ResearchTask
from src.tasks.troop_training import TroopTrainingTask
from src.vision import Vision
from src.workflow_engine import WorkflowEngine
from src.world_model import WorldModel


class BotEngine:
    def __init__(self, config_path: str = "config.yaml", observe_only: bool = False):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        dev_cfg = self.config["device"]
        self.client = AdbClient(
            adb_path=dev_cfg["adb_path"],
            ip=dev_cfg["ip"],
            port=dev_cfg["port"],
            timeout=dev_cfg.get("timeout_sec", 10),
            frame_ttl=dev_cfg.get("frame_ttl_sec", 1.5),
        )
        self.keep_screen_on = dev_cfg.get("keep_screen_on", True)

        self.fast_capture: Optional[FastCapture] = None
        if dev_cfg.get("fast_capture", True):
            try:
                self.fast_capture = FastCapture(
                    adb_path=self.client.adb_path,
                    serial=f"{self.client.ip}:{self.client.port}",
                    width=540,
                    height=1218,
                    record_time_limit=int(dev_cfg.get("fast_capture_time_limit", 180)),
                )
            except Exception as exc:
                logger.warning(f"[FastCapture] Не удалось подготовить быстрый поток: {exc}")
                self.fast_capture = None

        self.vision = Vision(templates_dir="assets")
        perception_cfg = self.config.get("perception", {})
        self.classifier = ScreenClassifier(
            self.vision,
            ocr_scale=perception_cfg.get("ocr_scale", 0.5),
            allow_ocr=perception_cfg.get("allow_ocr", True),
            match_scale=perception_cfg.get("match_scale", 0.5),
            city_nav_min_brightness=perception_cfg.get("city_nav_min_brightness", 45.0),
        )
        self.hud_parser = HudParser()
        self.hud_interval = perception_cfg.get("hud_interval_sec", 60)

        self.fsm = GameFSM()
        self.world = WorldModel()
        self.reflection = ReflectionLog()
        self.workflow = WorkflowEngine(
            self.client,
            self.vision,
            self.fsm,
            self.world,
            self.reflection,
        )

        policy = self.config.get("policy_and_safety", {})
        self.observe_only = observe_only or policy.get("mode") == "observe_only"
        self.on_unknown = policy.get("on_unknown", "recover")
        self.stop_on_payment = policy.get("stop_on_payment_screen", True)
        self.unknown_timeout = policy.get("unknown_timeout_sec", 25)
        self.max_back_attempts = policy.get("max_back_attempts", 2)

        recovery = self.config.get("recovery", {})
        self.recovery_enabled = recovery.get("enabled", True)
        self.relaunch_after_failures = recovery.get("relaunch_after_failures", 3)
        self.reconnect_interval = recovery.get("reconnect_interval_sec", 30)

        logging_cfg = self.config.get("logging", {})
        self.artifacts_dir = Path(logging_cfg.get("dir", "logs"))
        self.save_unknown_frames = logging_cfg.get("save_failure_frames", True)

        self.tasks: List[BaseTask] = []
        self._init_tasks()

        self.running = False
        self.started_at = 0.0
        self.last_frame_at = 0.0
        self.last_hud_at = 0.0
        self.last_state_change_at = time.time()
        self.last_reconnect_at = 0.0
        self.reconnect_failures = 0
        self.last_heartbeat_at = 0.0
        self.unknown_since: Optional[float] = None
        self.unknown_back_attempts = 0
        self.nav_attempts = 0
        self.consecutive_failures = 0
        self.last_hud: Optional[HudState] = None
        self.stats: Dict[str, int] = {"ticks": 0, "actions": 0, "unknown": 0, "recoveries": 0}

    # ------------------------------------------------------------------
    # Конфигурация
    # ------------------------------------------------------------------
    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _init_tasks(self) -> None:
        task_cfg = self.config.get("tasks", {})

        if task_cfg.get("popup_closer", {}).get("enabled", True):
            self.tasks.append(
                PopupCloserTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("popup_closer", {}).get("interval_sec", 5),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("alliance_help", {}).get("enabled", True):
            self.tasks.append(
                AllianceTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("alliance_help", {}).get("interval_sec", 60),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("chapter_quest", {}).get("enabled", True):
            self.tasks.append(
                ChapterQuestTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("chapter_quest", {}).get("interval_sec", 15),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("building_upgrade", {}).get("enabled", True):
            upgrade_cfg = task_cfg.get("building_upgrade", {})
            self.tasks.append(
                BuildingUpgradeTask(
                    self.client,
                    self.vision,
                    interval_sec=upgrade_cfg.get("interval_sec", 20),
                    spots=upgrade_cfg.get("spots"),
                    reflection=self.reflection,
                    world=self.world,
                    sweep=upgrade_cfg.get("sweep"),
                    castle=upgrade_cfg.get("castle"),
                    align_district=upgrade_cfg.get("align_district", True),
                )
            )

        if task_cfg.get("battle_auto", {}).get("enabled", False):
            self.tasks.append(
                BattleTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("battle_auto", {}).get("interval_sec", 30),
                    reflection=self.reflection,
                    world=self.world,
                )
            )

        if task_cfg.get("research_auto", {}).get("enabled", False):
            self.tasks.append(
                ResearchTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("research_auto", {}).get("interval_sec", 180),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("barracks_train", {}).get("enabled", False):
            self.tasks.append(
                TroopTrainingTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("barracks_train", {}).get("interval_sec", 120),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("quest_rewards", {}).get("enabled", False):
            self.tasks.append(
                QuestRewardsTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("quest_rewards", {}).get("interval_sec", 45),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("city_harvest", {}).get("enabled", True):
            self.tasks.append(
                CityHarvestTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("city_harvest", {}).get("interval_sec", 180),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("city_build", {}).get("enabled", False):
            build_cfg = task_cfg.get("city_build", {})
            self.tasks.append(
                CityBuildTask(
                    self.client,
                    self.vision,
                    interval_sec=build_cfg.get("interval_sec", 15),
                    plots=build_cfg.get("plots"),
                    reflection=self.reflection,
                )
            )

        if task_cfg.get("city_gifts", {}).get("enabled", True):
            self.tasks.append(
                CityGiftsTask(
                    self.client,
                    self.vision,
                    interval_sec=task_cfg.get("city_gifts", {}).get("interval_sec", 900),
                    reflection=self.reflection,
                )
            )

        logger.info(f"Инициализировано задач: {len(self.tasks)} ({', '.join(t.name for t in self.tasks)})")

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def start(self, duration_sec: int = 0) -> None:
        logger.info("Запуск бота War & Order...")
        if self.observe_only:
            logger.warning("Режим OBSERVE: бот не выполняет действия, только классифицирует экраны.")

        if not self.client.connect(allow_discovery=True):
            logger.error("Критическая ошибка: невозможно подключиться к устройству.")
            return

        self._start_fast_capture()
        self.client.wake_screen()
        if self.keep_screen_on:
            self.client.keep_screen_awake(True)
        self._ensure_game_foreground(initial=True)
        self.client.collapse_statusbar()
        self.client.dismiss_heads_up()

        self.running = True
        self.started_at = time.time()
        deadline = self.started_at + duration_sec if duration_sec else 0

        try:
            while self.running:
                if deadline and time.time() >= deadline:
                    logger.info(f"Достигнут лимит времени ({duration_sec}с), завершаем работу.")
                    break
                self.tick()
                sleep_human(1.0, 2.0)
        except KeyboardInterrupt:
            logger.info("Остановка бота пользователем (Ctrl+C).")
        finally:
            self.running = False
            if self.fast_capture is not None:
                self.fast_capture.stop()
            self.world.save()
            if self.keep_screen_on:
                self.client.keep_screen_awake(False)
            self._log_summary()

    def stop(self) -> None:
        self.running = False

    def _log_summary(self) -> None:
        uptime = time.time() - self.started_at if self.started_at else 0
        logger.info(
            "Итоги: "
            f"uptime={uptime:.0f}с, ticks={self.stats['ticks']}, "
            f"actions={self.stats['actions']}, unknown={self.stats['unknown']}, "
            f"recoveries={self.stats['recoveries']}"
        )

    # ------------------------------------------------------------------
    # Один такт
    # ------------------------------------------------------------------
    def _start_fast_capture(self) -> None:
        if self.fast_capture is None:
            return
        self.fast_capture.serial = f"{self.client.ip}:{self.client.port}"
        if not self.fast_capture.start():
            self.fast_capture = None

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Кадр для задач: быстрый поток, при недоступности — обычный скриншот."""
        if self.fast_capture is not None:
            frame = self.fast_capture.read(timeout_sec=0.9)
            if frame is not None:
                screen_size = self.client.screen_size
                if screen_size is not None and frame.shape[1::-1] != screen_size:
                    frame = cv2.resize(frame, screen_size, interpolation=cv2.INTER_LINEAR)
                return frame
        return self.client.get_screenshot(fresh=False)

    def tick(self) -> None:
        self.stats["ticks"] += 1

        if not self.client.is_connected():
            self._reconnect()
            return

        if not self.client.is_screen_on():
            if not self.client.wake_screen():
                logger.warning("[Screen] Экран остаётся выключенным (возможно, PIN/графический ключ). Пауза.")
                time.sleep(3)
            return

        if getattr(self.client, "is_phone_call_active", None) and self.client.is_phone_call_active():
            logger.info("[Safety] Активен телефонный вызов. Ожидание завершения разговора...")
            time.sleep(3)
            return

        if not self.client.is_game_foreground(self._package):
            self._ensure_game_foreground()
            return

        frame = self._capture_frame()
        if frame is None:
            logger.warning("Нет кадра с устройства, пауза.")
            time.sleep(2)
            return
        self.last_frame_at = time.time()

        # Иногда dumpsys сообщает Awake, а фактический кадр чёрный (локскрин
        # или погасший дисплей). Такой кадр нельзя отдавать задачам.
        if float(frame.mean()) < 2.0:
            logger.warning("[Screen] Получен чёрный кадр — будим экран и пропускаем такт.")
            self.client.wake_screen()
            time.sleep(1.0)
            return

        classification = self.classifier.classify(frame)
        state = classification.state
        if state != self.fsm.current_state:
            self.last_state_change_at = time.time()

        log_line = (
            f"{state} ({classification.source}, conf={classification.confidence:.2f})"
            if classification.is_known
            else f"UNKNOWN (ocr={classification.ocr_elapsed:.2f}с)"
        )
        logger.debug(f"[Tick {self.stats['ticks']}] {log_line}")

        self.fsm.update(state)
        for event in self.world.clean_expired_timers():
            logger.success(f"[WorldModel] Завершено событие: {event}")
        self.world.state["ui_state"] = state

        if classification.is_known:
            self.unknown_since = None
            self.unknown_back_attempts = 0
            self.consecutive_failures = 0
        else:
            self.stats["unknown"] += 1
        self._maybe_update_hud(frame, state)

        if self.observe_only:
            self._heartbeat(state)
            return

        # Аварийный тормоз: экран оплаты (проверяем только окна и неизвестные экраны,
        # чтобы не гонять OCR нижней трети на каждом такте в городе).
        if classification.is_popup and self.stop_on_payment and looks_like_payment(frame):
            logger.error("Обнаружен экран оплаты. Аварийная остановка бота.")
            self.stop()
            return

        if classification.is_popup:
            self._handle_popup(classification.state, frame)
            return

        if not classification.is_known:
            self._handle_unknown(frame, classification)
            return

        self._run_tasks(state, frame)
        self._navigate_home_if_needed(state, frame)
        self._heartbeat(state)

    # ------------------------------------------------------------------
    # Задачи
    # ------------------------------------------------------------------
    def _allowed_states(self, task: BaseTask) -> set[str]:
        # Задачи города работают только в городе; окна открывают они сами.
        if isinstance(task, BattleTask):
            # Боевой автопилот: бой, оверлей навыка и экран победы.
            # Выход из боя и повтор этапа — через обычную навигацию движка.
            return {
                GameState.TUTORIAL_BATTLE,
                GameState.SKILL_SELECTION,
                GameState.VICTORY_SCREEN,
            }
        if isinstance(task, BuildingUpgradeTask):
            return {GameState.CITY_VIEW, GameState.BUILDING_MENU, GameState.BUILDING_UPGRADE}
        if isinstance(task, CityBuildTask):
            # Постройка начинаетcя в городе и сама открывает экран ячейки.
            return {GameState.CITY_VIEW, GameState.PLOT_BUILD}
        if isinstance(task, ResearchTask):
            # Автопилот науки: меню Институт / дерево исследований.
            return {GameState.RESEARCH_MENU}
        if isinstance(task, (ChapterQuestTask, CityHarvestTask, CityGiftsTask, AllianceTask)):
            return {GameState.CITY_VIEW}
        if isinstance(task, PopupCloserTask):
            return {GameState.CITY_VIEW, GameState.UNKNOWN}
        return {
            GameState.CITY_VIEW,
            GameState.ROYAL_ROAD,
            GameState.CAMPAIGN_MAP,
            GameState.STAGE_BRIEFING,
        }

    def _run_tasks(self, state: str, frame) -> None:
        for task in self.tasks:
            if state not in self._allowed_states(task):
                continue
            if not task.can_run():
                continue

            try:
                did_action = task.execute(frame)
            except Exception as exc:  # noqa: BLE001 - задача не должна ронять движок
                self.consecutive_failures += 1
                self.reflection.record_failure(
                    task=task.name,
                    error=str(exc),
                    analysis=f"Исключение в задаче: {type(exc).__name__}",
                )
                logger.exception(f"Ошибка в задаче {task.name}: {exc}")
                if self.consecutive_failures >= 3:
                    self._recover("серия ошибок в задачах")
                continue

            if did_action:
                self.stats["actions"] += 1
                self.consecutive_failures = 0
                self.client.invalidate_frame()
                # Даём игре доиграть анимацию, следующий такт возьмёт новый кадр.
                time.sleep(0.8)
                return

    # ------------------------------------------------------------------
    # Всплывающие окна и награды
    # ------------------------------------------------------------------
    def _handle_popup(self, state: str, frame) -> None:
        logger.info(f"[Popup] Обработка окна: {state}")

        # 1. Крестик закрытия, если для него уже есть шаблон.
        close = self.vision.find_best(
            frame,
            [
                "btn_close_x",
                "btn_close_round",
                "btn_close_gold",
                "btn_close_event_x",
            ],
            threshold=0.72,
        )
        if close:
            x, y, confidence, template = close
            logger.info(f"[Popup] Закрываем '{template}' ({x}, {y}), conf={confidence:.2f}")
            self.client.tap(x, y)
            self.stats["actions"] += 1
            return

        # 2. Награда: забираем только по проверенному шаблону кнопки.
        if state == GameState.POPUP_REWARD:
            claim = self.vision.find_best(
                frame,
                ["btn_claim_reward", "btn_collect_reward", "btn_claim_green"],
                threshold=0.78,
            )
            if claim:
                x, y, confidence, template = claim
                logger.info(f"[Popup] Забираем награду '{template}' ({x}, {y}), conf={confidence:.2f}")
                self.client.tap(x, y)
                time.sleep(1.0)
                self.stats["actions"] += 1
                return

        # 3. Безопасный откат: системный Back не совершает покупок.
        logger.info("[Popup] Крестик/кнопка награды не найдены — нажимаем Back.")
        self.client.press_back()
        self.stats["actions"] += 1

    # ------------------------------------------------------------------
    # Навигация
    # ------------------------------------------------------------------
    # Радиальное меню и панель улучшения закрывает сама BuildingUpgradeTask:
    # после апгрейда панель исчезает, а кадр такта остаётся старым. Слепой тап
    # «стрелки» по такому кадру попадает в профиль лорда в городе.
    MENU_STATES = {
        GameState.ROYAL_ROAD,
        GameState.QUESTS_MENU,
        GameState.CAMPAIGN_MAP,
        GameState.STAGE_BRIEFING,
        GameState.TRAINING_MENU,
        GameState.RESEARCH_MENU,
        GameState.ALLIANCE_MENU,
        GameState.STORY_DIALOGUE,
    }

    def _navigate_home_if_needed(self, state: str, frame) -> None:
        """Если задача оставила бота в меню — мягко возвращаемся в город."""
        if state == GameState.CITY_VIEW:
            self.nav_attempts = 0
            return
        if state not in self.MENU_STATES:
            return
        if time.time() - self.last_state_change_at < 6:
            return

        if self.nav_attempts >= 3:
            self._recover(f"не удаётся выйти из экрана {state}")
            return

        self.nav_attempts += 1
        logger.info(f"[Nav] Возврат в город из {state} (попытка {self.nav_attempts}/3)")

        # В части меню аппаратный Back не закрывает экран (например, вкладка
        # «Задание»). Сначала ищем штатную стрелку назад в левом верхнем углу.
        height, width = frame.shape[:2]
        back_region = (0, 0, int(width * 0.15), int(height * 0.12))
        back = self.vision.find_template(frame, "btn_back_arrow", threshold=0.78, region=back_region)
        if back:
            bx, by, _ = back
            logger.debug(f"[Nav] Нажимаем штатную стрелку назад ({bx}, {by}).")
            self.client.tap(bx, by)
        elif state == GameState.CAMPAIGN_MAP:
            self.client.tap(75, 120)
        else:
            self.client.press_back()
        self.last_state_change_at = time.time()
        self.stats["actions"] += 1

    # ------------------------------------------------------------------
    # Неизвестный экран и восстановление
    # ------------------------------------------------------------------
    def _handle_unknown(self, frame, classification) -> None:
        now = time.time()
        if self.unknown_since is None:
            self.unknown_since = now
            logger.warning(f"[Recovery] Неизвестный экран. OCR: '{classification.ocr_text[:120]}'")

        # Проверяем оплату сразу, не дожидаясь таймаута: на таком экране
        # лучше остановиться, чем через 25 секунд нажать Back/тап.
        if self.stop_on_payment and looks_like_payment(frame):
            logger.error("[Recovery] Неизвестный экран похож на оплату. Останавливаемся.")
            self.stop()
            return

        elapsed = now - self.unknown_since
        if elapsed < self.unknown_timeout:
            return

        if self.on_unknown == "stop":
            logger.error("[Recovery] Политика on_unknown=stop: останавливаем бота.")
            self.stop()
            return

        if not self.recovery_enabled:
            logger.warning("[Recovery] Восстановление отключено в конфиге.")
            self.unknown_since = time.time()
            return

        self._save_artifact(frame, "unknown")
        self.stats["recoveries"] += 1

        if self.unknown_back_attempts < self.max_back_attempts:
            self.unknown_back_attempts += 1
            logger.info(f"[Recovery] Нажимаем Back ({self.unknown_back_attempts}/{self.max_back_attempts})...")
            self.client.press_back()
            self.unknown_since = time.time()
            return

        self._recover("неизвестный экран не закрывается")

    def _recover(self, reason: str) -> None:
        logger.warning(f"[Recovery] Полное восстановление: {reason}")
        self.stats["recoveries"] += 1
        self.client.release_touch()
        self.client.press_back()
        time.sleep(1.0)
        self.client.press_home()
        time.sleep(1.5)
        self._ensure_game_foreground()
        self.unknown_since = None
        self.unknown_back_attempts = 0
        self.consecutive_failures = 0

    def _reconnect(self) -> None:
        now = time.time()
        if now - self.last_reconnect_at < self.reconnect_interval:
            time.sleep(2)
            return
        self.last_reconnect_at = now
        logger.warning("[ADB] Соединение потеряно, переподключаемся...")
        if self.client.connect(allow_discovery=True):
            logger.success("[ADB] Соединение восстановлено.")
            self.reconnect_failures = 0
            self.client.wake_screen()
            if self.keep_screen_on:
                self.client.keep_screen_awake(True)
            self.client.collapse_statusbar()
            self._start_fast_capture()
        else:
            self.reconnect_failures += 1
            logger.error(
                f"[ADB] Переподключиться не удалось ({self.reconnect_failures}), ждём."
            )
            # Если устройство точно включено, зависший ADB-транспорт лечится
            # перезапуском сервера. Делаем это не чаще, чем раз в 5 неудач.
            if self.reconnect_failures % 5 == 0:
                self.client.restart_server()

    def _ensure_game_foreground(self, initial: bool = False) -> None:
        package = self._package
        if self.client.is_game_foreground(package):
            return
        logger.info("Игра не на переднем плане. Запускаем...")
        self.client.launch_game(package, self._activity)
        sleep_human(8.0, 12.0) if initial else sleep_human(4.0, 6.0)

    # ------------------------------------------------------------------
    # HUD и артефакты
    # ------------------------------------------------------------------
    def _maybe_update_hud(self, frame, state: str) -> None:
        if state != GameState.CITY_VIEW:
            return
        now = time.time()
        if now - self.last_hud_at < self.hud_interval:
            return
        self.last_hud_at = now
        try:
            hud = self.hud_parser.parse(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось разобрать HUD: {exc}")
            return

        if not hud.parsed_any:
            if hud.raw.get("overlay") == "debug_banner":
                logger.info("[HUD] Верхнюю панель перекрывает системная плашка — убираем её.")
                self.client.dismiss_heads_up()
                self.last_hud_at = 0.0
            else:
                logger.debug("[HUD] Значения не распознаны (возможно, панель перекрыта).")
            return

        self.last_hud = hud
        account = self.world.state.setdefault("account", {})
        if hud.lord_level is not None:
            account["lord_level"] = hud.lord_level
        if hud.power is not None:
            account["power"] = hud.power
        if hud.vip_level is not None:
            account["vip_level"] = hud.vip_level
        if hud.gems is not None:
            account["gems"] = hud.gems

        resources = self.world.state.setdefault("resources", {})
        for key in ("food", "wood"):
            value = getattr(hud, key)
            if value is not None:
                resources[key] = value

        self.world.save()
        logger.info(
            "[HUD] "
            f"Лорд={hud.lord_level}, сила={hud.power}, "
            f"еда={hud.food}, дерево={hud.wood}, золото={hud.gold}, гемы={hud.gems}"
        )

    def _save_artifact(self, frame, kind: str) -> None:
        if not self.save_unknown_frames:
            return
        try:
            folder = self.artifacts_dir / kind
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}.png"
            cv2.imwrite(str(path), frame)
            logger.info(f"[Artifact] Кадр сохранён: {path}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Не удалось сохранить кадр: {exc}")

    def _heartbeat(self, state: str) -> None:
        now = time.time()
        if now - self.last_heartbeat_at < 60:
            return
        self.last_heartbeat_at = now
        uptime = now - self.started_at
        hud = self.last_hud.as_dict() if self.last_hud else {}
        logger.info(
            f"[Heartbeat] uptime={uptime:.0f}с state={state} "
            f"actions={self.stats['actions']} unknown={self.stats['unknown']} "
            f"hud={hud} world={self.world.get_summary()}"
        )

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def _package(self) -> str:
        return self.config["game"]["package"]

    @property
    def _activity(self) -> str:
        return self.config["game"]["activity"]

    def snapshot(self) -> Dict[str, object]:
        """Один снимок состояния — для отладки и CLI."""
        if not self.client.connect(allow_discovery=True):
            return {"error": "adb_connect_failed"}
        self.client.wake_screen()
        if not self.client.is_game_foreground(self._package):
            return {"error": "game_not_foreground"}
        frame = self.client.get_screenshot(fresh=True)
        if frame is None:
            return {"error": "screenshot_failed"}
        classification = self.classifier.classify(frame)
        hud = self.hud_parser.parse(frame)
        return {
            "screen": classification.state,
            "confidence": round(classification.confidence, 3),
            "source": classification.source,
            "evidence": classification.evidence,
            "ocr_text": classification.ocr_text,
            "ocr_elapsed": round(classification.ocr_elapsed, 3),
            "near_misses": classification.near_misses,
            "hud": hud.as_dict(),
            "hud_raw": hud.raw,
        }

    def dump_snapshot(self) -> str:
        data = self.snapshot()
        return json.dumps(data, ensure_ascii=False, indent=2)
