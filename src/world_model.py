"""
Модель игрового мира (World Model) для War & Order.
Основана на архитектуре 4x-game-agent.
Хранит персистентное состояние аккаунта, таймеры строительства, исследований и найма,
баланс ресурсов, а также адаптивно запоминает координаты зданий на карте города.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from loguru import logger


MODEL_FILE = "memory_bank/world_model.json"


class WorldModel:
    """Персистентное представление игрового мира с прогнозированием таймеров."""

    def __init__(self, filepath: str = MODEL_FILE):
        self.filepath = Path(filepath)
        self.state: Dict[str, Any] = self._load()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "account": {
                "castle_level": 1,
                "power": 0,
                "lord_level": 1,
                "gems": 0,
                "vip_level": 0,
                "has_second_builder": False,
            },
            "builders": {
                "builder_1": {"busy": False, "target": None, "timer_end": None},
                "builder_2": {"busy": False, "target": None, "timer_end": None},
            },
            "buildings": {
                "castle": {"level": 1, "upgrading": False, "timer_end": None},
                "barracks": {"level": 0, "upgrading": False, "timer_end": None},
                "farm": {"level": 0, "upgrading": False, "timer_end": None},
                "sawmill": {"level": 0, "upgrading": False, "timer_end": None},
            },
            "research": {
                "active": False,
                "target": None,
                "timer_end": None,
            },
            "training": {
                "infantry": {"active": False, "timer_end": None},
                "cavalry": {"active": False, "timer_end": None},
                "archers": {"active": False, "timer_end": None},
                "mages": {"active": False, "timer_end": None},
            },
            "resources": {
                "food": 0,
                "wood": 0,
                "stone": 0,
                "iron": 0,
            },
            "cached_coordinates": {},  # name -> {"x": int, "y": int, "hits": int, "misses": int}
            "last_updated": time.time(),
        }

    def _load(self) -> Dict[str, Any]:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                default = self._default_state()
                for key in default:
                    if key not in data:
                        data[key] = default[key]
                return data
            except Exception as e:
                logger.warning(f"Ошибка загрузки {self.filepath}: {e}. Использование дефолтного состояния.")
        return self._default_state()

    def save(self) -> None:
        """Сохранение состояния модели на диск."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.state["last_updated"] = time.time()
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения WorldModel в {self.filepath}: {e}")

    # =========================================================
    # ТАЙМЕРЫ И СТРОИТЕЛИ
    # =========================================================

    def clean_expired_timers(self) -> list[str]:
        """Сброс завершившихся таймеров строительства, исследования и найма.

        Возвращает описания завершённых событий — их удобно писать в лог
        и использовать для планирования следующего действия.
        """
        now = time.time()
        completed: list[str] = []

        for b_id, b_data in self.state["builders"].items():
            if b_data["busy"] and b_data["timer_end"] and now >= b_data["timer_end"]:
                target = b_data["target"]
                logger.info(f"[WorldModel] Таймер строителя {b_id} ({target}) завершился!")
                b_data["busy"] = False
                b_data["target"] = None
                b_data["timer_end"] = None
                if target in self.state["buildings"]:
                    self.state["buildings"][target]["upgrading"] = False
                    self.state["buildings"][target]["timer_end"] = None
                    self.state["buildings"][target]["level"] += 1
                    completed.append(f"building:{target}")
                else:
                    completed.append(f"builder:{b_id}")

        research = self.state["research"]
        if research["active"] and research["timer_end"] and now >= research["timer_end"]:
            logger.info(f"[WorldModel] Исследование {research['target']} завершено!")
            completed.append(f"research:{research['target']}")
            research["active"] = False
            research["target"] = None
            research["timer_end"] = None

        for name, training in self.state.get("training", {}).items():
            if training["active"] and training["timer_end"] and now >= training["timer_end"]:
                training["active"] = False
                training["timer_end"] = None
                completed.append(f"training:{name}")

        if completed:
            self.save()
        return completed

    def builders_available(self) -> int:
        """Сколько строителей свободно прямо сейчас."""
        self.clean_expired_timers()
        free = 0
        if not self.state["builders"]["builder_1"]["busy"]:
            free += 1
        if (
            self.state["account"].get("has_second_builder")
            and not self.state["builders"]["builder_2"]["busy"]
        ):
            free += 1
        return free

    def is_builder_free(self) -> bool:
        """Доступен ли хотя бы один бесплатный строитель."""
        return self.builders_available() > 0

    def start_upgrade(self, building_name: str, duration_sec: float) -> bool:
        """Регистрация начатого улучшения здания.

        Возвращает False, если все доступные слоты строителей заняты:
        лучше не отмечать здание улучшающимся, чем показывать ложный таймер.
        """
        self.clean_expired_timers()
        if not self.state["builders"]["builder_1"]["busy"]:
            b_slot = "builder_1"
        elif (
            self.state["account"].get("has_second_builder")
            and not self.state["builders"]["builder_2"]["busy"]
        ):
            b_slot = "builder_2"
        else:
            logger.warning(
                f"[WorldModel] Нельзя начать улучшение {building_name}: все строители заняты."
            )
            return False

        now = time.time()
        timer_end = now + max(0.0, float(duration_sec))
        self.state["builders"][b_slot] = {
            "busy": True,
            "target": building_name,
            "timer_end": timer_end,
        }
        if building_name not in self.state["buildings"]:
            self.state["buildings"][building_name] = {
                "level": 1,
                "upgrading": True,
                "timer_end": timer_end,
            }
        else:
            self.state["buildings"][building_name]["upgrading"] = True
            self.state["buildings"][building_name]["timer_end"] = timer_end

        logger.info(
            f"[WorldModel] Улучшение {building_name} начато на слоте {b_slot}. "
            f"Окончание через {duration_sec}с."
        )
        self.save()
        return True

    # =========================================================
    # ПЛАНИРОВАНИЕ ПО ТАЙМЕРАМ
    # =========================================================

    def next_event(self) -> Optional[Tuple[float, str]]:
        """Ближайшее завершение таймера: (unix_time, описание) или None."""
        self.clean_expired_timers()
        now = time.time()
        events: list[Tuple[float, str]] = []

        for b_id, data in self.state["builders"].items():
            if data["busy"] and data.get("timer_end") and data["timer_end"] > now:
                events.append((float(data["timer_end"]), f"builder:{b_id}:{data.get('target')}"))

        research = self.state["research"]
        if research["active"] and research.get("timer_end") and research["timer_end"] > now:
            events.append((float(research["timer_end"]), f"research:{research.get('target')}"))

        for name, training in self.state.get("training", {}).items():
            if training["active"] and training.get("timer_end") and training["timer_end"] > now:
                events.append((float(training["timer_end"]), f"training:{name}"))

        if not events:
            return None
        events.sort(key=lambda item: item[0])
        return events[0]

    def seconds_until_next_event(self) -> float:
        """Сколько секунд до ближайшего таймера (inf, если таймеров нет)."""
        event = self.next_event()
        return max(0.0, event[0] - time.time()) if event else float("inf")

    def get_summary(self) -> str:
        """Короткая строка состояния для heartbeat/логов."""
        account = self.state["account"]
        nxt = self.next_event()
        if nxt:
            remaining = max(0, int(nxt[0] - time.time()))
            next_text = f", next={nxt[1]} in {remaining // 60}м"
        else:
            next_text = ", next=none"
        return (
            f"castle={account.get('castle_level', '?')}, "
            f"lord={account.get('lord_level', '?')}, "
            f"power={account.get('power', 0)}, "
            f"builders={self.builders_available()}{next_text}"
        )

    # =========================================================
    # АДАПТИВНЫЕ КООРДИНАТЫ ЗДАНИЙ
    # =========================================================

    def record_coord_hit(self, name: str, x: int, y: int) -> None:
        """Запоминание или подтверждение координат объекта."""
        entry = self.state["cached_coordinates"].get(name, {"x": x, "y": y, "hits": 0, "misses": 0})
        entry["x"] = x
        entry["y"] = y
        entry["hits"] += 1
        self.state["cached_coordinates"][name] = entry
        self.save()

    def record_coord_miss(self, name: str) -> None:
        """Фиксация промаха по координате (для запуска адаптивного поиска)."""
        if name in self.state["cached_coordinates"]:
            self.state["cached_coordinates"][name]["misses"] += 1
            self.save()

    def get_cached_coord(self, name: str) -> Optional[Tuple[int, int]]:
        """Получить подтвержденные координаты объекта."""
        if name in self.state["cached_coordinates"]:
            item = self.state["cached_coordinates"][name]
            return item["x"], item["y"]
        return None
