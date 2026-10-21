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

print("Tapping Castle upgrade queue banner at (272, 549)...")
c.tap(272, 549)
time.sleep(1.5)

# Also tap center castle (540, 1100)
c.tap(540, 1100)
time.sleep(1.5)

screen = c.get_screenshot()
cv2.imwrite("screen_after_free_check.png", screen)
artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\free_check.png"
cv2.imwrite(artifact_path, screen)

ocr = RapidOCR()
res, _ = ocr(screen)
print("=== AFTER TAP OCR ===")
for r in (res or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
