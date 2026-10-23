"""
Организация эталонного датасета скриншотов War & Order для OpenCV.

Что делает:
1. Сканирует все скриншоты проекта (logs/inspect, logs/unknown, assets/screens).
2. Классифицирует каждый файл по состоянию игры (city/radial/panel/...).
3. Полные кадры (1080x2436) и кадры FastCapture (540x1218) конвертирует
   в JPEG q90 в assets/dataset/<категория>/; обрезки копирует как PNG
   в assets/dataset/crops/.
4. Генерирует assets/dataset/manifest.json — разметку для OpenCV:
   состояние, камера, подписи зданий (из labels.json), описание.
5. labels.json — ручные/проверенные подписи; при перегенерации они
   не затираются, а вливаются в манифест.

Идемпотентность: повторный запуск добавляет только новые файлы;
существующие JPEG не перезаписываются.

Использование:
    python tools/dataset_organize.py            # полная генерация
    python tools/dataset_organize.py --force    # перезаписать JPEG
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "assets" / "dataset"
MANIFEST = DATASET / "manifest.json"
LABELS = DATASET / "labels.json"

FULL_SIZE = (1080, 2436)   # (w, h) скриншот ADB
FAST_SIZE = (540, 1218)    # (w, h) кадр FastCapture

# Категория = состояние игры (совпадает с профилями ScreenClassifier там,
# где экран распознаётся ботом). Порядок папок — порядок в манифесте.
CATEGORIES = [
    "city",        # city_view без выделения здания
    "radial",      # радиальное меню выбранного здания
    "panel",       # панель улучшения/постройки («Требуется»)
    "plot",        # пустая ячейка, экран новой постройки
    "battle",      # бой, выбор навыка, пауза, победа
    "campaign",    # карта кампании, брифинг, диалоги
    "royal_road",  # Королевская Дорога
    "quests",      # задания
    "alliance",    # альянс
    "popups",      # события, награды, офферы, системные диалоги
    "misc",        # восстановление, heads-up, тесты захвата
    "crops",       # обрезки-кандидаты шаблонов
]

# Точная карта: имя файла -> (категория, состояние, описание).
# Составлена по живым сессиям 2026-10-21/22 (см. memory_bank/development_log.md).
FILE_MAP: dict[str, tuple[str, str, str]] = {
    # ---------------- city ----------------
    "after_double_back": ("city", "city_view", "Город после двойного Back (проверка выхода)"),
    "after_globe_back": ("city", "city_view", "Город после возврата из чата мира"),
    "after_pinch": ("city", "city_view", "Город после пинча (тест камеры)"),
    "after_revert": ("city", "city_view", "Город после возврата камеры"),
    "after_worldmap_tap": ("city", "city_view", "После тапа по мини-городу: карта мира не открылась"),
    "after_worldmap_return": ("city", "city_view", "Возврат в город"),
    "before_globe_tap": ("city", "city_view", "Город перед тапом по глобусу"),
    "builder_tap2": ("city", "city_view", "Город после тапа по слоту строителя (меню не открылось)"),
    "after_builder_tap": ("city", "city_view", "Город после тапа по слоту строителя"),
    "castle_tap": ("city", "city_view", "Город после тапа по замку (радиальное меню замка, не верифицировано)"),
    "city_after_briefing": ("city", "city_view", "Город после брифинга этапа"),
    "city_after_probe3": ("city", "city_view", "Город после третьего зонда камеры"),
    "city_after_relaunch": ("city", "city_view", "Город после перезапуска игры"),
    "city_after_stage7_exit": ("city", "city_view", "Город после выхода из этапа 7"),
    "city_after_user_pan": ("city", "city_view", "Город после ручного пана камеры"),
    "city_before_tracker_tap": ("city", "city_view", "Город перед тапом по трекеру квеста"),
    "city_before_worldmap": ("city", "city_view", "Город перед попыткой карты мира"),
    "city_check_now": ("city", "city_view", "Проверка города"),
    "city_current": ("city", "city_view", "Город, камера по умолчанию (2026-10-21)"),
    "city_deselect": ("city", "city_view", "Город после снятия выделения здания"),
    "city_live": ("city", "city_view", "Живой город"),
    "city_now": ("city", "city_view", "Город, последовательная проверка 1"),
    "city_now2": ("city", "city_view", "Город, последовательная проверка 2"),
    "city_pan_left": ("city", "city_view", "Город после пана влево"),
    "city_probe6": ("city", "city_view", "Зонд камеры города 6"),
    "city_probe11": ("city", "city_view", "Зонд камеры города 11 (портал вне экрана)"),
    "city_relaunch_stable": ("city", "city_view", "Стабильный город после перезапуска"),
    "city_template_fresh": ("city", "city_view", "Свежий кадр для шаблонов HUD"),
    "check_probe_now": ("city", "city_view", "Проверка состояния после зонда"),
    "current_state": ("city", "city_view", "Текущее состояние (город)"),
    "last": ("city", "city_view", "Последний кадр сессии (город)"),
    "screen_now_check": ("city", "city_view", "Проверка экрана (город)"),
    "live_building_scan": ("city", "city_view", "Сканирование города перед стройкой: видны пустые ячейки"),
    "after_forge_build": ("city", "city_view", "Город сразу после постройки кузницы (камера перецентрирована)"),
    "after_forge_lv2": ("city", "city_view", "Город после улучшения кузницы до LV.2"),
    "forge_after_lv4": ("city", "city_view", "Город после улучшения кузницы до LV.4"),
    "forge_lv5_timer_started": ("city", "city_view", "Город с запущенным таймером кузницы LV.5 (09:54)"),
    "probe_bld_no_radial_wall": ("city", "city_view", "Тап по стене не открыл радиальное меню"),
    "relaunch_0": ("city", "city_view", "Кадр перезапуска игры 0"),
    "relaunch_state": ("city", "city_view", "Состояние после перезапуска"),
    "post_recovery_check": ("city", "city_view", "Город после аварийного восстановления"),
    "post_recovery_check2": ("city", "city_view", "Город после восстановления, проверка 2"),
    "after_recovery_check": ("city", "city_view", "Проверка после восстановления"),
    "after_recovery_check2": ("city", "city_view", "Проверка после восстановления 2"),
    "recover_final": ("city", "city_view", "Финальный кадр восстановления (город)"),
    "ds_city_now": ("city", "city_view", "Город, текущая камера (сброс-камера) перед датасет-сканом"),
    "ds_castle_radial": ("city", "city_view", "Тап по зоне замка не открыл меню — остался город"),
    "ds_sawmill_radial": ("city", "city_view", "Тап по зоне пилорамы не открыл меню — остался город"),
    "ds_tap_540_850": ("city", "city_view", "Город после тапа (540,850), меню не открылось"),
    "grid_hit_200_650_radial": ("radial", "city_view", "Радиальное меню стены из грид-зонда (200,650), conf 0.817"),
    "grid_hit_200_1150_radial": ("radial", "city_view", "Радиальное меню стены из грид-зонда (200,1150), conf 0.954"),
    "grid_hit_600_1150_radial": ("radial", "city_view", "Радиальное меню стены из грид-зонда (600,1150), conf 0.954"),
    "grid_hit_400_1400_radial_wall": ("radial", "city_view", "Радиальное меню стены из грид-зонда (400,1400), conf 0.954"),
    "grid_hit_800_1850_build": ("plot", "plot_build", "Экран постройки Фермы с пустой ячейки (800,1850), btn_build_free 0.996"),
    # ---------------- radial ----------------
    "forge_radial": ("radial", "city_view", "Радиальное меню кузницы LV.1 после постройки"),
    "forge_selected": ("radial", "city_view", "Кузница выбрана (центр здания)"),
    "forge_selected_base": ("radial", "city_view", "Основание кузницы выбрано (кликабельный хитбокс)"),
    "forge_lv4_radial": ("radial", "city_view", "Радиальное меню кузницы LV.4: Детали/Улучшить/Коваль/Склад"),
    "radial_open_now": ("radial", "city_view", "Открытое радиальное меню (сброс-камера)"),
    "radial_fast_0": ("radial", "city_view", "Радиальное меню в быстром потоке, кадр 0 (540x1218)"),
    "radial_fast_1": ("radial", "city_view", "Радиальное меню в быстром потоке, кадр 1 (540x1218)"),
    "builder_queue_verified": ("radial", "city_view", "Радиальное меню стены: Детали/Улучшить/Оборона/Найм"),
    "builder_timer_sync": ("radial", "city_view", "Радиальное меню стены + таймер кузницы 07:08 в HUD"),
    "probe_bld_radial_wall": ("radial", "city_view", "Радиальное меню стены (сброс-камера), conf 0.984"),
    "building_menu_wall": ("radial", "city_view", "Меню стены (2026-10-21)"),
    # ---------------- panel ----------------
    "forge_upgrade_panel": ("panel", "building_upgrade", "Панель улучшения кузницы LV.1→LV.2"),
    "forge_lv2_upgrade_panel": ("panel", "building_upgrade", "Панель кузницы LV.2→LV.3"),
    "forge_lv3_upgrade_panel": ("panel", "building_upgrade", "Панель кузницы LV.3→LV.4"),
    "forge_lv4_upgrade_panel": ("panel", "building_upgrade", "Панель кузницы LV.4→LV.5, таймер 09:54"),
    "forge_before_lv4": ("panel", "building_upgrade", "Панель кузницы LV.3→LV.4 перед подтверждением"),
    "building_upgrade_wall": ("panel", "building_upgrade", "Эталонная панель стены LV.4→LV.5: якорь «Требуется»"),
    "probe_bld_panel_wall": ("panel", "building_upgrade", "Панель стены LV.5→LV.6 (сброс-камера), таймер 4277 с"),
    "probe_bld_panel_live": ("panel", "building_upgrade", "Панель стены, живая проверка OCR"),
    # ---------------- plot ----------------
    "plot_probe_690_330": ("plot", "plot_build", "Экран постройки кузницы с пустой ячейки (690,330)"),
    # ---------------- battle ----------------
    "battle_move_before": ("battle", "tutorial_battle", "Бой до движения (тест джойстика)"),
    "battle_move_after": ("battle", "tutorial_battle", "Бой после движения"),
    "battle_observe_full": ("battle", "tutorial_battle", "Полный кадр наблюдения боя"),
    "battle_observe_1": ("battle", "tutorial_battle", "Наблюдение боя 1 (540x1218)"),
    "battle_observe_2": ("battle", "tutorial_battle", "Наблюдение боя 2 (540x1218)"),
    "battle_observe_3": ("battle", "tutorial_battle", "Наблюдение боя 3 (540x1218)"),
    "battle_observe_4": ("battle", "tutorial_battle", "Наблюдение боя 4 (540x1218)"),
    "battle_observe_5": ("battle", "tutorial_battle", "Наблюдение боя 5 (540x1218)"),
    "battle_observe_6": ("battle", "tutorial_battle", "Наблюдение боя 6 (540x1218)"),
    "battle_observe_last": ("battle", "tutorial_battle", "Последний кадр наблюдения (540x1218)"),
    "battle_fast_540": ("battle", "tutorial_battle", "Кадр быстрого потока в бою (540x1218)"),
    "stage7_battle_end": ("battle", "tutorial_battle", "Конец боя этапа 7"),
    "stage7_battle_full": ("battle", "tutorial_battle", "Полный кадр боя этапа 7"),
    "stage7_battle_pull": ("battle", "tutorial_battle", "Кадр боя из pull_fallback"),
    "stage7_battle_fast": ("battle", "tutorial_battle", "Быстрый кадр боя этапа 7 (540x1218)"),
    "stage7_pause_menu_1": ("battle", "battle_pause", "Меню паузы боя этапа 7"),
    "after_skill_card_tap": ("battle", "skill_selection", "Выбор навыка после тапа по первой карточке"),
    "after_skill_confirm": ("battle", "skill_selection", "Выбор навыка после «Подтвердить»"),
    "current_skill_selection": ("battle", "skill_selection", "Оверлей выбора навыка в бою"),
    # ---------------- campaign ----------------
    "after_back_campaign_dialogue": ("campaign", "campaign_dialogue", "Диалог кампании после Back"),
    "campaign_after_stage7_exit": ("campaign", "campaign_map", "Карта кампании после выхода из этапа 7"),
    "campaign_after_stage7_exit_fast": ("campaign", "campaign_map", "Карта кампании после выхода (540x1218)"),
    "campaign_after_tap_dialogue": ("campaign", "campaign_dialogue", "Диалог после тапа на карте"),
    "campaign_dialogue_current": ("campaign", "campaign_dialogue", "Текущий диалог кампании"),
    "campaign_map_probe": ("campaign", "campaign_map", "Карта кампании (зонд)"),
    "campaign_map_probe_fast_540": ("campaign", "campaign_map", "Карта кампании (540x1218)"),
    "campaign_probe_back1": ("campaign", "campaign_map", "Карта после Back"),
    "campaign_probe_crown": ("campaign", "campaign_map", "Тап по короне на карте"),
    "campaign_probe_current": ("campaign", "campaign_map", "Текущая карта кампании"),
    "campaign_reopened": ("campaign", "campaign_map", "Повторно открытая карта кампании"),
    "campaign_stage7_probe": ("campaign", "campaign_map", "Затемнённая карта, этап 7 (сброс-камера)"),
    "campaign_stage7_probe_fast_540": ("campaign", "campaign_map", "Затемнённая карта этапа 7 (540x1218)"),
    "stage7_after_exit_1": ("campaign", "campaign_map", "После выхода из этапа 7"),
    "stage7_briefing_dark": ("campaign", "stage_briefing", "Брифинг этапа 7"),
    "stage7_after_challenge": ("campaign", "hero_test_dialogue", "Диалог героизма после «Испытание»"),
    # ---------------- royal_road ----------------
    "royal_road": ("royal_road", "royal_road", "Королевская Дорога, глава 4"),
    "royal_road_check": ("royal_road", "royal_road", "Королевская Дорога, проверка"),
    "royal_road_clean": ("royal_road", "royal_road", "Королевская Дорога без оверлеев"),
    "royal_road_forward": ("royal_road", "royal_road", "Королевская Дорога после «Вперёд»"),
    "royal_road_probe": ("royal_road", "royal_road", "Зонд Королевской Дороги"),
    "royal_road_fast_540": ("royal_road", "royal_road", "Королевская Дорога (540x1218)"),
    "step_royal": ("royal_road", "royal_road", "Шаг по Королевской Дороге"),
    "step_forward_1": ("royal_road", "royal_road", "Шаг «Вперёд» 1"),
    "step_forward_2": ("royal_road", "royal_road", "Шаг «Вперёд» 2"),
    "step_forward_3": ("royal_road", "royal_road", "Шаг «Вперёд» 3"),
    "step_forward_4": ("royal_road", "royal_road", "Шаг «Вперёд» 4"),
    "step_forward_5": ("royal_road", "royal_road", "Шаг «Вперёд» 5"),
    # ---------------- quests ----------------
    "quest_claim_after": ("quests", "quests_menu", "Задания после получения награды"),
    "quests_after_claims": ("quests", "quests_menu", "Задания после забора наград"),
    "quests_after_claims2": ("quests", "quests_menu", "Задания после забора наград 2"),
    "quests_all_claims_done": ("quests", "quests_menu", "Все награды получены"),
    "quests_claims_round3": ("quests", "quests_menu", "Третий раунд получения наград"),
    "quests_menu": ("quests", "quests_menu", "Экран заданий: 6 зелёных кнопок"),
    "quests_menu_now": ("quests", "quests_menu", "Экран заданий, текущий"),
    "quests_probe": ("quests", "quests_menu", "Зонд экрана заданий"),
    "quests_probe_fast_540": ("quests", "quests_menu", "Зонд заданий (540x1218)"),
    "claims_final_menu": ("quests", "quests_menu", "Финальное меню наград"),
    # ---------------- alliance ----------------
    "alliance": ("alliance", "alliance_menu", "Экран альянса (онбординг «Присоединяйтесь»)"),
    # ---------------- popups ----------------
    "after_daily_reward_claim": ("popups", "popup_reward", "После получения ежедневной награды"),
    "after_event_double_tap": ("popups", "popup_generic", "Событие после двойного тапа"),
    "after_event_long_press": ("popups", "popup_generic", "Событие после долгого нажатия"),
    "after_event_x": ("popups", "popup_generic", "Событие после крестика"),
    "event_current2": ("popups", "popup_offer", "Событие «Новое Королевство»"),
    "reward_popup_current": ("popups", "popup_reward", "Попап награды"),
    "synth_receipt": ("popups", "popup_reward", "Синтетический чек награды (источник шаблона)"),
    "guide_quest_hammer": ("popups", "popup_offer", "Окно «Разблокировать золотой молот» — премиальное, не открывать"),
    "after_manual_exit": ("popups", "popup_generic", "Диалог «Выйти из игры?» после"),
    "before_manual_exit": ("popups", "popup_generic", "Диалог «Выйти из игры?» до"),
    "after_globe_tap": ("popups", "unknown", "Чат мира после тапа по глобусу"),
    "after_globe_toggle": ("popups", "unknown", "Переключение чата мира"),
    # ---------------- misc ----------------
    "recover_0_before": ("misc", "unknown", "Восстановление, кадр до 0"),
    "recover_1_before": ("misc", "unknown", "Восстановление, кадр до 1"),
    "recover_2_before": ("misc", "unknown", "Восстановление, кадр до 2"),
    "recover_3_before": ("misc", "unknown", "Восстановление, кадр до 3"),
    "recover_unknown_current": ("misc", "unknown", "Неизвестный экран во время восстановления"),
    "fast_capture": ("misc", "unknown", "Тест быстрого захвата (540x1218)"),
    "hs2_dome_base": ("misc", "unknown", "Heads-up уведомление, купол (540x1218)"),
    "hs2_dome_top": ("misc", "unknown", "Heads-up уведомление, верх купола (540x1218)"),
    "tracker_fast_0": ("misc", "unknown", "Трекинг шаблона в быстром потоке 0 (540x1218)"),
    "tracker_fast_1": ("misc", "unknown", "Трекинг шаблона 1 (540x1218)"),
    "tracker_fast_2": ("misc", "unknown", "Трекинг шаблона 2 (540x1218)"),
    "tracker_fast_3": ("misc", "unknown", "Трекинг шаблона 3 (540x1218)"),
    "tracker_fast_4": ("misc", "unknown", "Трекинг шаблона 4 (540x1218)"),
    "tracker_fast_5": ("misc", "unknown", "Трекинг шаблона 5 (540x1218)"),
    "post_commit_check_0": ("misc", "unknown", "Тёмный кадр проверки после коммита 0"),
    "post_commit_check_1": ("misc", "unknown", "Тёмный кадр проверки 1"),
    "post_commit_check_2": ("misc", "unknown", "Тёмный кадр проверки 2"),
    "tracker_after": ("misc", "city_view", "Кадр после трекинга"),
    # ---------------- assets/screens (эталоны, уже в репозитории) -------------
    "stage_1_victory": ("battle", "stage_victory", "Эталон: экран победы этапа 1"),
    "quests_screen": ("quests", "quests_menu", "Эталон: экран заданий"),
    "after_rewards_claimed": ("popups", "popup_reward", "Эталон: после получения наград"),
    "after_quest_hammer_tap": ("popups", "popup_offer", "Эталон: окно золотого молота"),
    "after_dismiss_hammer_modal": ("city", "city_view", "Эталон: город после закрытия окна молота"),
    "back_to_city_after_quests": ("city", "city_view", "Эталон: возврат в город после заданий"),
}

# Обрезки: всё, что не полный и не быстрый кадр. Описания точечные,
# остальные — общие «кандидат шаблона».
CROP_NOTES = {
    "wall_towers_area": "Область башен стены — источник building_wall_tower",
    "portal_template_crop": "Источник шаблона portal_campaign",
    "portal_template_candidate": "Кандидат шаблона портала",
    "upgrade_bottom_crop": "Нижняя полоса панели улучшения (кнопки)",
    "battle_pause_fast_crop": "Источник battle_pause_fast",
    "btn_world_map_template": "Кандидат шаблона мини-города",
    "title_campaign_candidate": "Кандидат заголовка кампании",
    "back_candidate": "Кандидат кнопки назад",
    "portal_region": "Область портала",
    "portal_arch": "Арка портала",
    "portal_arch2": "Арка портала 2",
    "portal_current_crop": "Текущий кроп портала",
    "city_check_portal_area": "Область портала на кадре города",
    "city_check_south": "Южная полоса города",
    "south_east_detail": "Юго-восток города",
    "marker_south_east": "Маркер юго-востока",
    "gold_blobs_tiles": "Золотые блобы на тайлах",
    "wall_banners": "Баннеры стены",
    "tower_candidates": "Кандидаты башен",
    "tower_relaunch_crop": "Кроп башни после перезапуска",
    "bottom_left_crop": "Левый нижний угол",
    "candidates_relaunch": "Кандидаты после перезапуска",
    "hand": "Рука-указатель обучения",
    "stage6_candidate": "Кандидат этапа 6",
    "stage7_candidate": "Кандидат этапа 7 (источник campaign_stage_7)",
    "stage7_new_a": "Кандидат этапа 7, вариант A",
    "stage7_new_b": "Кандидат этапа 7, вариант B",
    "stage7_new_c": "Кандидат этапа 7, вариант C",
    "stage7_new_d": "Кандидат этапа 7, вариант D",
    "battle_pause_a": "Кандидат кнопки паузы A",
    "battle_pause_b": "Кандидат кнопки паузы B",
    "battle_pause_c": "Кандидат кнопки паузы C",
    "battle_pause_fast_a": "Кандидат быстрой паузы A",
    "battle_pause_fast_b": "Кандидат быстрой паузы B",
    "battle_pause_fast_c": "Кандидат быстрой паузы C",
    "hero_test_a": "Кандидат «Испытание» героизма A",
    "hero_test_b": "Кандидат «Испытание» героизма B",
    "hero_test_c": "Кандидат «Испытание» героизма C",
    "20261021_201652_unknown": "Неизвестный экран (фиксация unknown)",
    "ds_quad_tl": "Верхне-левый квадрант города (сброс-камера)",
    "ds_quad_tr": "Верхне-правый квадрант города (сброс-камера)",
    "ds_quad_bl": "Нижне-левый квадрант города (сброс-камера)",
    "ds_quad_br": "Нижне-правый квадрант города (сброс-камера)",
    "ds_castle_zone": "Зона замка (сброс-камера)",
    "ds_sawmill_zone": "Зона пилорамы (сброс-камера)",
}


def iter_sources() -> list[Path]:
    files: list[Path] = []
    for folder in ("logs/inspect", "logs/unknown", "assets/screens"):
        base = ROOT / folder
        if base.exists():
            files.extend(sorted(base.glob("*.png")))
    return files


def classify(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    entry = FILE_MAP.get(stem)
    if entry is None:
        # Fallback для будущих файлов: грубые префиксы.
        lower = stem.lower()
        if lower.startswith(("city", "forge", "relaunch")):
            return "city", "city_view", "Город (авто-категория)"
        if "radial" in lower:
            return "radial", "city_view", "Радиальное меню (авто)"
        if "panel" in lower or "upgrade" in lower:
            return "panel", "building_upgrade", "Панель улучшения (авто)"
        if lower.startswith(("battle", "stage7_battle")):
            return "battle", "tutorial_battle", "Бой (авто)"
        if lower.startswith(("campaign", "stage7")):
            return "campaign", "campaign_map", "Кампания (авто)"
        if "quest" in lower:
            return "quests", "quests_menu", "Задания (авто)"
        if "royal" in lower:
            return "royal_road", "royal_road", "Королевская Дорога (авто)"
        return "misc", "unknown", "Не классифицирован (авто)"
    return entry


def load_labels() -> dict:
    if LABELS.exists():
        with open(LABELS, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Организация датасета скриншотов")
    parser.add_argument("--force", action="store_true", help="Перезаписывать существующие JPEG")
    args = parser.parse_args()

    labels = load_labels()
    manifest_images: list[dict] = []
    converted = 0

    for src in iter_sources():
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        category, state, description = classify(src)
        frame_kind = "crop"
        if (w, h) == FULL_SIZE:
            frame_kind = "full"
        elif (w, h) == FAST_SIZE:
            frame_kind = "fast"

        if frame_kind == "crop":
            category = "crops"
            description = CROP_NOTES.get(src.stem, "Обрезка-кандидат шаблона")
            rel_name = f"{src.stem}.png"
        else:
            rel_name = f"{src.stem}.jpg"

        is_asset_screen = "assets" in src.parts
        if is_asset_screen:
            # Уже в репозитории как эталон — не дублируем.
            dataset_rel = src.relative_to(ROOT).as_posix()
        else:
            dataset_rel = f"{category}/{rel_name}"
            dst = DATASET / category / rel_name
            if dst.exists() and not args.force:
                pass
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if frame_kind == "crop":
                    cv2.imwrite(str(dst), img)
                else:
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                converted += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entry: dict = {
            "path": dataset_rel,
            "source": src.relative_to(ROOT).as_posix(),
            "category": category,
            "state": state,
            "frame": frame_kind,
            "size": [w, h],
            "brightness": round(float(gray.mean()), 1),
            "description": description,
        }
        hand = labels.get(src.relative_to(ROOT).as_posix()) or labels.get(dataset_rel)
        if hand:
            entry.update(hand)
        manifest_images.append(entry)

    # Сортировка: категории в порядке CATEGORIES, внутри — по имени.
    cat_order = {name: i for i, name in enumerate(CATEGORIES)}
    manifest_images.sort(
        key=lambda e: (cat_order.get(e["category"], 99), e["path"])
    )

    DATASET.mkdir(parents=True, exist_ok=True)
    manifest = {
        "meta": {
            "device_resolution": list(FULL_SIZE),
            "fast_capture_resolution": list(FAST_SIZE),
            "updated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "categories": CATEGORIES,
            "note": (
                "Полные кадры и кадры FastCapture хранятся как JPEG q90 "
                "(эталоны для поиска шаблонов и регресс-тестов детекции). "
                "Обрезки-кандидаты шаблонов — PNG в crops/. "
                "Ручные/проверенные подписи зданий — labels.json, "
                "вливаются в манифест при каждой генерации. "
                "bbox — [x1, y1, x2, y2] в координатах исходного кадра."
            ),
        },
        "images": manifest_images,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Файлов в манифесте: {len(manifest_images)}")
    print(f"Сконвертировано/скопировано: {converted}")
    for name in CATEGORIES:
        count = sum(1 for e in manifest_images if e["category"] == name)
        print(f"  {name:12s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
