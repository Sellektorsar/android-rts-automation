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

print("1. Opening Royal Road (75, 1960)...")
c.tap(75, 1960)
time.sleep(2.5)

print("2. Tapping 'Вперед' for Barracks Upgrade at (870, 1100)...")
c.tap(870, 1100)
time.sleep(2.5)

# Advance any Bella dialog if present
for _ in range(2):
    c.tap(540, 2200)
    time.sleep(1.0)

# Check screenshot
screen = c.get_screenshot()
cv2.imwrite("screen_barracks_centered_test.png", screen)
artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_centered_test.png"
cv2.imwrite(artifact_path, screen)

# Tapping centered building (540, 1200)
print("3. Tapping centered building (540, 1200)...")
c.tap(540, 1200)
time.sleep(1.5)

screen2 = c.get_screenshot()
cv2.imwrite("screen_barracks_radial_test.png", screen2)
artifact_path2 = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_radial_test.png"
cv2.imwrite(artifact_path2, screen2)

res2, _ = ocr(screen2)
print("=== RADIAL OCR ===")
for r in (res2 or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
