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

print('Tapping (822, 1284)...')
c.tap(822, 1284)
time.sleep(1.8)

screen = c.get_screenshot()
if screen is not None:
    cv2.imwrite('screen_after_822_1284.png', screen)
    artifact_path = r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\screen_after_822_1284.png'
    cv2.imwrite(artifact_path, screen)
    print('Saved screen_after_822_1284.png')
    
    ocr = RapidOCR()
    res, _ = ocr(screen)
    print('=== OCR Results after tap ===')
    for r in (res or []):
        box, text, score = r
        cx = int(sum(p[0] for p in box)/4)
        cy = int(sum(p[1] for p in box)/4)
        print(f'({cx:4d}, {cy:4d}) [{float(score):.2f}]: {text}')
else:
    print('Screenshot is None')
