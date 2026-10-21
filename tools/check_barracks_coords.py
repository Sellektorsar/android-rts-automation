import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread('screen_barracks_target.png')
res, _ = ocr(img)
print("=== All text on screen_barracks_target.png ===")
for r in (res or []):
    cx = int(sum(p[0] for p in r[0])/4)
    cy = int(sum(p[1] for p in r[0])/4)
    print(f"({cx:4d}, {cy:4d}) [{float(r[2]):.2f}]: {r[1]}")
