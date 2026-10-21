import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_city_view.png')
crop = img[500:1800, 100:980]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\modal_crop.png', crop)
res, _ = ocr(crop)
print('Modal OCR:')
for r in (res or []):
    print(r[1])
