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

    def clean_expired_timers(self) -> None:
        """Сброс завершившихся таймеров строительства и найма."""
        now = time.time()
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

        if self.state["research"]["active"] and self.state["research"]["timer_end"] and now >= self.state["research"]["timer_end"]:
            logger.info(f"[WorldModel] Исследование {self.state['research']['target']} завершено!")
            self.state["research"]["active"] = False
            self.state["research"]["target"] = None
            self.state["research"]["timer_end"] = None

    def is_builder_free(self) -> bool:
        """Доступен ли хотя бы один бесплатный строитель."""
        self.clean_expired_timers()
        if not self.state["builders"]["builder_1"]["busy"]:
            return True
        if self.state["account"]["has_second_builder"] and not self.state["builders"]["builder_2"]["busy"]:
            return True
        return False

    def start_upgrade(self, building_name: str, duration_sec: float) -> None:
        """Регистрация начатого улучшения здания."""
        now = time.time()
        timer_end = now + duration_sec
        b_slot = "builder_1" if not self.state["builders"]["builder_1"]["busy"] else "builder_2"
        self.state["builders"][b_slot] = {
            "busy": True,
            "target": building_name,
            "timer_end": timer_end,
        }
        if building_name not in self.state["buildings"]:
            self.state["buildings"][building_name] = {"level": 1, "upgrading": True, "timer_end": timer_end}
        else:
            self.state["buildings"][building_name]["upgrading"] = True
            self.state["buildings"][building_name]["timer_end"] = timer_end

        logger.info(f"[WorldModel] Улучшение {building_name} начато на слоте {b_slot}. Окончание через {duration_sec}с.")
        self.save()

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
