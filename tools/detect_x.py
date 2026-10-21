import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\rr_close_btn.png')
# Find high contrast / red / white cross
# The crop is x: 850..1050, y: 350..550
# Let's find contours in the top-right quadrant of this crop
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 25 < w < 80 and 25 < h < 80:
        gx = 850 + x + w//2
        gy = 350 + y + h//2
        print(f'Possible X button at ({gx}, {gy}), size: {w}x{h}')
