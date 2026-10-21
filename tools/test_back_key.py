import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import yaml
from src.adb_client import AdbClient
from rapidocr_onnxruntime import RapidOCR

config = yaml.safe_load(open('config.yaml', encoding='utf-8'))
dev_cfg = config['device']
c = AdbClient(adb_path=dev_cfg['adb_path'], ip=dev_cfg['ip'], port=dev_cfg['port'])
c.connect()

print('Sending KEYEVENT_BACK...')
c._run_cmd(["shell", "input", "keyevent", "4"])
time.sleep(2.0)

screen = c.get_screenshot()
ocr = RapidOCR()
res, _ = ocr(screen)
text = " | ".join([r[1] for r in res]) if res else ""
print('After Back Keyevent:', text[:120])
