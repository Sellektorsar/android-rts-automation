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
if c.connect():
    img = c.get_screenshot()
    if img is not None:
        cv2.imwrite('screen_lost_inspect.png', img)
        artifact_path = r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\screen_lost_inspect.png'
        cv2.imwrite(artifact_path, img)
        ocr = RapidOCR()
        res, _ = ocr(img)
        print('Current screen OCR:')
        for r in (res or []):
            cx = int(sum(p[0] for p in r[0])/4)
            cy = int(sum(p[1] for p in r[0])/4)
            print(f'({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}')
