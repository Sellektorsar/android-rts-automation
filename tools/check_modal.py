import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_transition.png')
res, _ = ocr(img)
print("=== FULL OCR of chapter4_transition.png ===")
for r in (res or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
