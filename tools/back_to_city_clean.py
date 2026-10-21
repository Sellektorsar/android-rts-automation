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

for i in range(3):
    screen = c.get_screenshot()
    res, _ = ocr(screen)
    text = " ".join([r[1].lower() for r in res]) if res else ""
    if "событие" in text or "рюкзак" in text or "почта" in text:
        print("Successfully back in City View!")
        break
    print(f"Attempt {i+1}: sending Back key...")
    c._run_cmd(["shell", "input", "keyevent", "4"])
    time.sleep(1.5)

screen = c.get_screenshot()
cv2.imwrite("screen_clean_city_live.png", screen)
res, _ = ocr(screen)
print("=== CITY VIEW LIVE OCR ===")
for r in (res or []):
    cy = int(sum(p[1] for p in r[0])/4)
    if 1800 < cy < 2100:
        print(f"[{float(r[2]):.2f}]: {r[1]}")
