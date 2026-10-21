import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread('current_actual_screen.png')
res, _ = ocr(img)
print(f'Total detected OCR tokens: {len(res) if res else 0}')
for r in (res or []):
    box, text, score = r
    cx = int(sum(p[0] for p in box)/4)
    cy = int(sum(p[1] for p in box)/4)
    print(f'({cx:4d}, {cy:4d}) [{float(score):.2f}]: {text}')
