import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import cv2
from src.adb_client import AdbClient

config = yaml.safe_load(open('config.yaml', encoding='utf-8'))
dev_cfg = config['device']
c = AdbClient(adb_path=dev_cfg['adb_path'], ip=dev_cfg['ip'], port=dev_cfg['port'])
if c.connect():
    img = c.get_screenshot()
    if img is not None:
        cv2.imwrite('current_actual_screen.png', img)
        artifact_path = r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\current_actual_screen.png'
        cv2.imwrite(artifact_path, img)
        print('Saved screenshot to current_actual_screen.png and artifacts!')
    else:
        print('Screenshot was None!')
else:
    print('Failed to connect!')
