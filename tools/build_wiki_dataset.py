"""
Скачивание всех изображений фан-вики War and Order в отдельный датасет.

Источник: https://war-and-order-mobile.fandom.com/wiki/War_and_order_Wiki
MediaWiki API отдаёт список файлов; Fandom при скачивании возвращает WebP,
поэтому все изображения конвертируются в PNG (Pillow). Рядом с файлами
пишется manifest.json: исходное имя, URL, страницы, категория, размеры,
хеш и признак дубликата.

Категории:
    buildings   — изображения со страницы Buildings (изометрические кропы);
    units       — изображения со страницы Troops (рендеры-портреты);
    screenshots — крупные игровые скриншоты;
    site        — оформление вики (фоны, логотипы);
    other       — остальное.

Использование:
    python tools/build_wiki_dataset.py
    python tools/build_wiki_dataset.py --output assets/wiki --json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

API_URL = "https://war-and-order-mobile.fandom.com/api.php"
SOURCE_PAGE = "https://war-and-order-mobile.fandom.com/wiki/War_and_order_Wiki"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
IMAGE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://war-and-order-mobile.fandom.com/",
    "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
}
SITE_NAMES = {
    "site-background-dark",
    "site-background-light",
    "wiki-background",
    "site-logo.png",
    "wiki-wordmark.png",
    "camelgames.png",
}
# Нормализованные стемы (нижний регистр, подчёркивания → пробелы).
BUILDING_NAME_HINTS = {
    "barracks", "blacksmith", "castle", "castle wall", "city hall", "college",
    "depot", "drill grounds", "embassy", "farm", "guardiantemple",
    "guardian temple", "hospital", "iron mine", "ironmine", "medic tent",
    "medictent", "merchant ship", "merchantship", "merchantship-0", "sawmill",
    "stone mine", "stonemine", "training grounds", "wall", "war hall",
    "warhall", "warhalls", "watch tower", "watchtower", "workshop",
    "workshopsmall",
}
UNIT_NAME_HINTS = {
    "angel", "armored ogre", "cyclops", "elf archer", "elfarcher",
    "elf mage", "elfmage", "fire demon", "firedemon", "golem", "human archer",
    "humanarcher", "human calvary", "humancalvary", "human cavalry",
    "humancavalry", "human mage", "humanmage", "human soldier", "humansoldier",
    "minotaur", "night hound", "nighthound", "orc calvary", "orccalvary",
    "orc cavalry", "orccavalry", "orc shaman", "orcshaman", "orc soldier",
    "orcsoldier", "red dragon", "reddragon", "rock giant", "rockgiant",
    "skeleton",
}
BUILDING_PAGE_TITLES = {
    "Buildings", "Barracks", "Blacksmith", "Castle", "Castle Wall",
    "City Hall", "College", "Depot", "Drill Grounds", "Embassy", "Farm",
    "Guardian Temple", "Hospital", "Iron Mine", "Medic Tent", "Merchant Ship",
    "Sawmill", "Stone Mine", "Training Grounds", "War Hall", "Watch Tower",
    "Watchtower", "Workshop",
}
UNIT_PAGE_TITLES = {
    "Troops", "Angel", "Golem", "Infantry", "Mage", "Calvary", "Archer",
    "Monsters",
}
BUILDING_PAGE = "Buildings"
UNIT_PAGE = "Troops"


def _api_get(params: Dict[str, Any], *, retries: int = 3) -> Dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    url = f"{API_URL}?{query}"
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"MediaWiki API недоступен: {last_error}")


def _download(url: str, *, retries: int = 3) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=IMAGE_HEADERS)
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Не удалось скачать {url}: {last_error}")


def _normalize_title(name: str) -> str:
    return name.replace("_", " ").strip().lower()


def _sanitize_stem(stem: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", stem, flags=re.UNICODE).strip("._")
    return cleaned or "image"


def _collect_pages() -> tuple[Dict[str, list[str]], Dict[str, set[str]]]:
    """Возвращает карты «файл → страницы» и «страница → файлы»."""
    pages_data = _api_get({"action": "query", "list": "allpages", "aplimit": 500})
    titles = [page["title"] for page in pages_data.get("query", {}).get("allpages", [])]
    image_pages: Dict[str, list[str]] = {}
    page_images: Dict[str, set[str]] = {}
    for start in range(0, len(titles), 40):
        batch = titles[start : start + 40]
        data = _api_get(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "images",
                "imlimit": 500,
            }
        )
        for page in data.get("query", {}).get("pages", []):
            page_title = page.get("title", "")
            for image in page.get("images", []):
                file_title = image.get("title", "")
                if file_title.lower().startswith("file:"):
                    file_title = file_title[5:]
                key = _normalize_title(file_title)
                image_pages.setdefault(key, []).append(page_title)
                page_images.setdefault(page_title, set()).add(key)
        time.sleep(0.2)
    return image_pages, page_images


def _categorize(
    name: str,
    width: int,
    height: int,
    pages: list[str],
    building_images: set[str],
    unit_images: set[str],
) -> str:
    key = _normalize_title(name)
    stem = _normalize_title(Path(name).stem)
    if name.lower() in SITE_NAMES:
        return "site"
    if key in building_images or stem in BUILDING_NAME_HINTS:
        return "buildings"
    if key in unit_images or stem in UNIT_NAME_HINTS:
        return "units"
    page_titles = set(pages)
    if page_titles & BUILDING_PAGE_TITLES:
        return "buildings"
    if page_titles & UNIT_PAGE_TITLES:
        return "units"
    if width <= 200 and height <= 200:
        return "site"
    if width >= 700 or height >= 700:
        return "screenshots"
    return "other"


def _unique_name(used: set[str], stem: str, extension: str, hint: str = "") -> str:
    """Детерминированное имя без коллизий внутри набора файлов."""
    candidate = stem
    key = f"{candidate.lower()}{extension.lower()}"
    if key in used:
        candidate = f"{stem}_{hint or extension.lstrip('.')}"
        key = f"{candidate.lower()}{extension.lower()}"
    suffix = 2
    while key in used:
        candidate = f"{stem}_{suffix}"
        key = f"{candidate.lower()}{extension.lower()}"
        suffix += 1
    used.add(key)
    return f"{candidate}{extension}"


def build_dataset(output: Path) -> Dict[str, Any]:
    inventory = _api_get(
        {
            "action": "query",
            "list": "allimages",
            "ailimit": 500,
            "aiprop": "url|size|mime|sha1",
        }
    )
    files = inventory.get("query", {}).get("allimages", [])
    if not files:
        raise RuntimeError("Вики не вернула список изображений")
    format_rank = {"png": 0, "gif": 1, "jpg": 2, "jpeg": 3}
    files.sort(
        key=lambda item: (
            Path(item["name"]).stem.lower(),
            format_rank.get(Path(item["name"]).suffix.lstrip(".").lower(), 9),
            item["name"].lower(),
        )
    )

    image_pages, page_images = _collect_pages()
    building_images = page_images.get(BUILDING_PAGE, set())
    unit_images = page_images.get(UNIT_PAGE, set())
    print(
        f"Изображений в вики: {len(files)}; "
        f"зданий на странице {BUILDING_PAGE}: {len(building_images)}; "
        f"юнитов на странице {UNIT_PAGE}: {len(unit_images)}."
    )

    directories = {
        category: output / category
        for category in ("buildings", "units", "screenshots", "site", "other")
    }
    originals = output / "_original"
    for directory in [*directories.values(), originals]:
        directory.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    seen_hashes: Dict[str, str] = {}
    used_outputs: Dict[str, set[str]] = {category: set() for category in directories}
    used_originals: set[str] = set()
    for index, item in enumerate(files, start=1):
        name = item["name"]
        key = _normalize_title(name)
        pages = image_pages.get(key, [])
        category = _categorize(
            name,
            int(item.get("width") or 0),
            int(item.get("height") or 0),
            pages,
            building_images,
            unit_images,
        )
        stem = _sanitize_stem(Path(name).stem)
        entry: Dict[str, Any] = {
            "name": name,
            "category": category,
            "pages": pages,
            "source_url": item.get("url"),
            "width": item.get("width"),
            "height": item.get("height"),
            "mime": item.get("mime"),
            "sha1": item.get("sha1"),
            "file": None,
            "original_file": None,
            "duplicate_of": None,
        }
        try:
            blob = _download(item["url"])
            digest = hashlib.sha256(blob).hexdigest()
            if digest in seen_hashes:
                entry["duplicate_of"] = seen_hashes[digest]

            with Image.open(io.BytesIO(blob)) as image:
                actual_format = (image.format or "PNG").lower()
                hint = Path(name).suffix.lstrip(".").lower()
                original_name = _unique_name(
                    used_originals, stem, f".{actual_format}", hint=hint
                )
                original_path = originals / original_name
                original_path.write_bytes(blob)
                entry["original_file"] = str(original_path).replace("\\", "/")
                entry["downloaded_format"] = actual_format
                entry["bytes"] = len(blob)

                if image.mode in ("P", "PA", "LA"):
                    image = image.convert("RGBA")
                elif image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                output_name = _unique_name(
                    used_outputs[category], stem, ".png", hint=hint
                )
                out_path = directories[category] / output_name
                image.save(out_path, format="PNG")
                entry["file"] = str(out_path).replace("\\", "/")
                entry["output_width"], entry["output_height"] = image.size

            seen_hashes[digest] = entry["file"]
            entry["sha256"] = digest
            print(f"[{index}/{len(files)}] {category:11s} {name} -> {entry['file']}")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
            print(f"[{index}/{len(files)}] ОШИБКА {name}: {exc}")
        entries.append(entry)
        time.sleep(0.15)

    counts = Counter(entry["category"] for entry in entries if entry.get("file"))
    manifest = {
        "source": SOURCE_PAGE,
        "api": API_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(entries),
        "converted": sum(1 for entry in entries if entry.get("file")),
        "duplicates": sum(1 for entry in entries if entry.get("duplicate_of")),
        "counts": dict(counts),
        "images": entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Готово: {manifest['converted']}/{manifest['total']} файлов, "
        f"дубликатов {manifest['duplicates']}; manifest: {manifest_path}."
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Скачивание изображений вики War and Order")
    parser.add_argument("--output", default="assets/wiki")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        manifest = build_dataset(Path(args.output))
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in ("total", "converted", "duplicates", "counts")
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
