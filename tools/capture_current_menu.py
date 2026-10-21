import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

screen = c.get_screenshot()
if screen is not None:
    cv2.imwrite('current_menu_screen.png', screen)
    artifact_path = r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\current_menu_screen.png'
    cv2.imwrite(artifact_path, screen)
    print('Screenshot saved!')
    
    ocr = RapidOCR()
    res, _ = ocr(screen)
    print('=== SCREEN OCR ===')
    for r in (res or []):
        box = r[0]
        cx = int(sum(p[0] for p in box)/4)
        cy = int(sum(p[1] for p in box)/4)
        print(f'({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}')
