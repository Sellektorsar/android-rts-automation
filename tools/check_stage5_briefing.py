import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread('screen_stage5_briefing_opened.png')
if img is not None:
    res, _ = ocr(img)
    print('screen_stage5_briefing_opened OCR:')
    for r in (res or []):
        print(f'({int(r[0][0][0])}, {int(r[0][0][1])}): {r[1]}')
else:
    print('screen_stage5_briefing_opened is None')
