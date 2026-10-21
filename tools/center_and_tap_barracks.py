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

print("1. Tapping HUD quest banner at (375, 1957)...")
c.tap(375, 1957)
time.sleep(2.0)

# Take screenshot to see where camera centered
screen = c.get_screenshot()
cv2.imwrite("screen_barracks_centered_now.png", screen)
artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_centered_now.png"
cv2.imwrite(artifact_path, screen)

# Tapping centered building
print("2. Tapping centered building at (540, 1200)...")
c.tap(540, 1200)
time.sleep(1.5)

screen2 = c.get_screenshot()
cv2.imwrite("screen_radial_menu.png", screen2)
artifact_path2 = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\radial_menu.png"
cv2.imwrite(artifact_path2, screen2)

res2, _ = ocr(screen2)
print("=== RADIAL MENU OCR ===")
for r in (res2 or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    if 900 < cy < 1700:
        print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
