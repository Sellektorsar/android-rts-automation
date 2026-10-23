"""
Офлайн-проверка: насколько изображения вики подходят как OpenCV-шаблоны
для живых кадров игры.

Для каждого шаблона и кадра считается:
    * лучший multiscale-ответ TM_CCOEFF_NORMED (масштабы 0.5..1.5);
    * число inlier-совпадений ORB + RANSAC-гомография.

Результат: logs/inspect/wiki_match_report.json и по одному размеченному
кадру на лучший шаблон.

Использование:
    python tools/probe_wiki_match.py
    python tools/probe_wiki_match.py --category units --frames logs/inspect/battle.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FRAMES = [
    "logs/inspect/sweep_hit_forge.png",
    "logs/inspect/camera_shift_after_pan_settled.png",
]
DEFAULT_LIVE_POINTS = {
    # Подпись -> ожидаемая точка живого кадра (best-effort из прошлых прогонов).
    "sweep_hit_forge.png": {"forge": [470, 468]},
    "camera_shift_after_pan_settled.png": {"forge": [768, 528]},
}
SCALES = [round(0.5 + 0.1 * index, 2) for index in range(11)]
RATIO = 0.75
MIN_GOOD = 10


def _match_multiscale(frame_bgr: np.ndarray, template_bgr: np.ndarray) -> dict:
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    best = None
    for scale in SCALES:
        size = (
            max(8, int(round(template_gray.shape[1] * scale))),
            max(8, int(round(template_gray.shape[0] * scale))),
        )
        if size[0] >= frame_gray.shape[1] or size[1] >= frame_gray.shape[0]:
            continue
        scaled = cv2.resize(template_gray, size, interpolation=cv2.INTER_AREA)
        scaled = cv2.GaussianBlur(scaled, (5, 5), 0)
        response = cv2.matchTemplate(frame_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "scale": scale,
                "top_left": [int(location[0]), int(location[1])],
                "size": [int(size[0]), int(size[1])],
                "center": [
                    int(location[0] + size[0] / 2),
                    int(location[1] + size[1] / 2),
                ],
            }
    return best or {"score": 0.0, "scale": 0.0, "center": None}


def _match_orb(frame_bgr: np.ndarray, template_bgr: np.ndarray) -> dict:
    orb = cv2.ORB_create(nfeatures=4000)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame_gray, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(
        template_gray, None
    )
    result = {
        "keypoints_template": len(keypoints_template or []),
        "keypoints_frame": len(keypoints_frame or []),
        "good_matches": 0,
        "inliers": 0,
        "center": None,
    }
    if descriptors_template is None or descriptors_frame is None:
        return result
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        descriptors_template, descriptors_frame, k=2
    )
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance]
    result["good_matches"] = len(good)
    if len(good) < MIN_GOOD:
        return result
    source = np.float32([keypoints_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([keypoints_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return result
    inliers = int(mask.sum())
    result["inliers"] = inliers
    height, width = template_gray.shape[:2]
    center = np.float32([[[width / 2, height / 2]]])
    projected = cv2.perspectiveTransform(center, homography)[0][0]
    if np.isfinite(projected).all():
        result["center"] = [int(projected[0]), int(projected[1])]
    return result


def _annotate(frame_bgr: np.ndarray, label: str, match: dict) -> np.ndarray:
    annotated = frame_bgr.copy()
    if match.get("top_left") and match.get("size"):
        x, y = match["top_left"]
        w, h = match["size"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)
    if match.get("center"):
        cv2.circle(annotated, tuple(match["center"]), 10, (0, 0, 255), -1)
    text = f"{label} {match['score']:.3f} @{match['scale']:.2f}"
    cv2.putText(
        annotated, text, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 8
    )
    cv2.putText(
        annotated, text, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
    )
    return annotated


def probe(frames: list[Path], templates: list[Path], output: Path) -> dict:
    frame_images = {}
    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Не удалось открыть кадр: {path}")
        frame_images[path.name] = image

    report: dict = {"scales": SCALES, "orb_ratio": RATIO, "frames": {}, "templates": {}}
    started = time.time()
    for template_path in templates:
        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template is None:
            print(f"шаблон не открылся: {template_path}")
            continue
        if template.shape[2] == 4:
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        name = template_path.stem
        per_frame = {}
        for frame_name, frame in frame_images.items():
            multiscale = _match_multiscale(frame, template)
            orb = _match_orb(frame, template)
            per_frame[frame_name] = {"multiscale": multiscale, "orb": orb}
        report["templates"][name] = per_frame

        for frame_name, result in per_frame.items():
            if result["multiscale"].get("center"):
                annotated = _annotate(
                    frame_images[frame_name], name, result["multiscale"]
                )
                out_path = output / f"wiki_match_{Path(frame_name).stem}_{name}.png"
                cv2.imwrite(str(out_path), annotated)
            print(
                f"{name:20s} {frame_name:38s} "
                f"tm={result['multiscale']['score']:.3f}"
                f"@{result['multiscale']['scale']:.2f} "
                f"orb={result['orb']['inliers']}/{result['orb']['good_matches']}"
            )

    for frame_name, frame in frame_images.items():
        ranking = sorted(
            (
                (
                    stats["multiscale"]["score"],
                    name,
                    stats["multiscale"],
                    stats["orb"],
                )
                for name, stats in (
                    (name, per_frame[frame_name])
                    for name, per_frame in report["templates"].items()
                )
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        top = ranking[0]
        report["frames"][frame_name] = {
            "expected": DEFAULT_LIVE_POINTS.get(frame_name, {}),
            "top_multiscale": {
                "template": top[1],
                **top[2],
            },
            "top_by_orb": max(
                (
                    (stats["orb"]["inliers"], stats["orb"]["good_matches"], name, stats["orb"]["center"])
                    for name, stats in (
                        (name, per_frame[frame_name])
                        for name, per_frame in report["templates"].items()
                    )
                ),
                default=(0, 0, None, None),
            ),
        }
        if top[2].get("center"):
            annotated = _annotate(frame, f"top: {top[1]}", top[2])
            cv2.imwrite(
                str(output / f"wiki_match_top_{Path(frame_name).stem}.png"), annotated
            )

    report["seconds"] = round(time.time() - started, 1)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка шаблонов вики на живых кадрах")
    parser.add_argument(
        "--frames",
        nargs="+",
        default=DEFAULT_FRAMES,
        help="живые кадры для проверки",
    )
    parser.add_argument(
        "--category",
        default="buildings",
        help="категория assets/wiki (buildings, units, screenshots)",
    )
    parser.add_argument("--templates-dir", default=None)
    parser.add_argument("--output", default="logs/inspect")
    parser.add_argument("--report", default="logs/inspect/wiki_match_report.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    templates_dir = Path(args.templates_dir or f"assets/wiki/{args.category}")
    templates = sorted(templates_dir.glob("*.png"))
    if not templates:
        print(f"В {templates_dir} нет PNG-шаблонов")
        return 1
    frames = [Path(path) for path in args.frames]
    report = probe(frames, templates, output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nОтчёт: {report_path} ({report['seconds']} c)")
    for frame_name, info in report["frames"].items():
        top = info["top_multiscale"]
        print(
            f"{frame_name}: top={top['template']} tm={top['score']:.3f} "
            f"center={top['center']}"
        )
        print(f"  top_by_orb={info['top_by_orb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
