import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\target_1289.png')
print('target_1289 shape:', img.shape)

# Let's save a zoomed-in, scaled up version (4x) with contrast enhancement
zoom = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\target_1289_zoom.png', zoom)
print('Saved target_1289_zoom.png')
