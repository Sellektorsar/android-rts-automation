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

print("Tapping 'Забрать награды' at (542, 1994)...")
c.tap(542, 1994)
time.sleep(2.0)

for _ in range(5):
    c.tap(540, 1800)
    time.sleep(1.0)

screen = c.get_screenshot()
cv2.imwrite("screen_chapter4_transition.png", screen)
artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_transition.png"
cv2.imwrite(artifact_path, screen)

ocr = RapidOCR()
res, _ = ocr(screen)
print("=== CURRENT SCREEN OCR ===")
for r in (res or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    if cy > 1700 or cy < 600:
        print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
