import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.adb_client import AdbClient

config = yaml.safe_load(open('config.yaml', encoding='utf-8'))
dev_cfg = config['device']
c = AdbClient(adb_path=dev_cfg['adb_path'], ip=dev_cfg['ip'], port=dev_cfg['port'])
connected = c.connect()
print('Connected:', connected, 'Port:', c.port)
