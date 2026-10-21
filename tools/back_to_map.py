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
if not c.connect():
    print('Failed to connect')
    sys.exit(1)

ocr = RapidOCR()

# Check screen
screen = c.get_screenshot()
res, _ = ocr(screen)
text = " | ".join([r[1] for r in res]) if res else ""
print('Current screen:', text[:120])

# If on Hero Enhancement, press Back (75, 120)
if "укрепление" in text.lower() or "способности лорда" in text.lower() or "развивать" in text.lower():
    print('Closing Hero Enhancement with Back button (75, 120)...')
    c.tap(75, 120)
    time.sleep(2.0)
    screen = c.get_screenshot()
    res, _ = ocr(screen)
    text = " | ".join([r[1] for r in res]) if res else ""
    print('After Back:', text[:120])
