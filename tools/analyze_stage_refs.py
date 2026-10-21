import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
s5 = cv2.imread('stage5_destination_center.png')
res5, _ = ocr(s5)
print('stage5 OCR:', res5)

s2 = cv2.imread('stage2_crop.png')
res2, _ = ocr(s2)
print('stage2 OCR:', res2)
