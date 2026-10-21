import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import yaml
import cv2
from src.adb_client import AdbClient
from rapidocr_onnxruntime import RapidOCR

config = yaml.safe_load(open('config.yaml', encoding='utf-8'))
dev_cfg = config['device']
c = AdbClient(adb_path=dev_cfg['adb_path'], ip=dev_cfg['ip'], port=dev_cfg['port'])
c.connect()

ocr = RapidOCR()

print("1. Tapping 'Заберите награды главы' banner at (284, 1958)...")
c.tap(284, 1958)
time.sleep(2.5)

screen1 = c.get_screenshot()
cv2.imwrite("screen_royal_road_claim_modal.png", screen1)
artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\royal_road_claim_modal.png"
cv2.imwrite(artifact_path, screen1)

res1, _ = ocr(screen1)
print("Royal Road modal OCR:")
for r in (res1 or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    if cy > 1800:
        print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")

print("2. Tapping 'Забрать награды главы' button (543, 1995)...")
c.tap(543, 1995)
time.sleep(2.0)

# Dismiss popups
print("3. Dismissing reward popups at (540, 1800)...")
for _ in range(4):
    c.tap(540, 1800)
    time.sleep(1.0)

screen2 = c.get_screenshot()
cv2.imwrite("screen_chapter4_live.png", screen2)
artifact_path2 = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_live.png"
cv2.imwrite(artifact_path2, screen2)

res2, _ = ocr(screen2)
print("=== AFTER CLAIMING CHAPTER 3 OCR ===")
for r in (res2 or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    if cy > 1800:
        print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
