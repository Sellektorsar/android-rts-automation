import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\dialogue_bubble_full.png')
res, _ = ocr(img)
print('Dialogue bubble OCR:')
for r in (res or []):
    print(r)
