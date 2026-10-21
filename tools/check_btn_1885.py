import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\screen_after_822_1284.png')

# Crop button area
crop = img[1750:2000, 300:780]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\crop_btn_1885.png', crop)

res, _ = ocr(crop)
print('Crop OCR:', res)

# Check colors in the button area
print('Button mean BGR:', crop.mean(axis=(0,1)))
