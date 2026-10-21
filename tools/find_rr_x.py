import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\chapter4_city_view.png')
crop_x = img[350:550, 850:1050]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\rr_close_btn.png', crop_x)

# Find X icon (usually red/gold or cross)
print('Crop X shape:', crop_x.shape)
# In Royal Road, the window has a wooden/gold frame.
# Let's inspect where the X button or click outside (e.g. 540, 300 or 950, 420) is.
