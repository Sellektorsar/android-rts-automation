import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\screen_after_822_1284.png')
btn = img[1830:1940, 360:720]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\exact_btn.png', btn)

hsv = cv2.cvtColor(btn, cv2.COLOR_BGR2HSV)
print('Exact btn shape:', btn.shape)
print('Mean BGR:', btn.mean(axis=(0,1)))
print('Max BGR:', btn.max(axis=(0,1)))
print('Min BGR:', btn.min(axis=(0,1)))

# Let's inspect energy or requirements above the button!
# Is there an energy requirement (e.g. 5 energy, 10 energy, etc.)?
# In the top left of the screen, energy showed: '10'
# In Stage briefing:
# Let's crop y: 1600..1880, x: 200..880 to see if there is energy cost or locked condition!
req_crop = img[1600:1860, 200:880]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\req_crop.png', req_crop)

from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
res, _ = ocr(req_crop)
print('Requirements OCR:')
for r in (res or []):
    print(r)
