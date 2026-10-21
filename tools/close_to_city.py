import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import yaml
import cv2
from src.adb_client import AdbClient

config = yaml.safe_load(open('config.yaml', encoding='utf-8'))
dev_cfg = config['device']
c = AdbClient(adb_path=dev_cfg['adb_path'], ip=dev_cfg['ip'], port=dev_cfg['port'])
c.connect()

# Close Royal Road modal with X at (968, 404) or Back
c.tap(968, 404)
time.sleep(2.0)

screen = c.get_screenshot()
cv2.imwrite("screen_chapter4_city_view.png", screen)
artifact_path = r"C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_city_view.png"
cv2.imwrite(artifact_path, screen)
print("Saved chapter4_city_view.png!")
