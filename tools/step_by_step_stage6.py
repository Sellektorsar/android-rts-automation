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

# Step 1: Check screen now
s1 = c.get_screenshot()
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\step1_screen.png', s1)
res1, _ = ocr(s1)
text1 = " | ".join([r[1] for r in res1]) if res1 else "None"
print('Step 1 Screen OCR:', text1[:200])

# Step 2: If on map, tap (822, 1284) to open briefing
if "испытание" not in text1.lower() or "этап 6" not in text1.lower():
    print('Opening Stage 6 briefing by tapping (822, 1284)...')
    c.tap(822, 1284)
    time.sleep(2.0)
    s2 = c.get_screenshot()
    cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\step2_briefing.png', s2)
    res2, _ = ocr(s2)
    text2 = " | ".join([r[1] for r in res2]) if res2 else "None"
    print('Step 2 Briefing OCR:', text2[:200])
else:
    s2 = s1

# Step 3: Now look for the green button 'Испытание' in s2
# In s2, where is 'Испытание'?
for r in (res2 if 'res2' in locals() else res1):
    if "испыт" in r[1].lower():
        box = r[0]
        cx = int(sum(p[0] for p in box)/4)
        cy = int(sum(p[1] for p in box)/4)
        print(f"Found 'Испытание' text at ({cx}, {cy})")

# Let's tap 'Испытание' at (540, 1885)
print('Tapping (540, 1885)...')
c.tap(540, 1885)
time.sleep(2.0)

s3 = c.get_screenshot()
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\step3_after_tap_challenge.png', s3)
res3, _ = ocr(s3)
text3 = " | ".join([r[1] for r in res3]) if res3 else "None"
print('Step 3 After Tap Challenge OCR:', text3[:300])
