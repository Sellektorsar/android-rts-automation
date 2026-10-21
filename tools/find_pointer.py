import cv2
import numpy as np

img = cv2.imread('screen_barracks_centered_now.png')
# Find tutorial hand / finger or pulsing circle
# Hand pointer has a white glove or yellow glowing ring
# Let's crop middle area (y: 600..1800, x: 200..900)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\bella_middle.png', img[600:1800, 200:900])

# Find yellow/orange glowing elements in this crop
hsv = cv2.cvtColor(img[600:1800, 200:900], cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (15, 120, 150), (35, 255, 255))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Found highlights in middle:')
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h > 150:
        gx = 200 + x + w//2
        gy = 600 + y + h//2
        print(f'Highlight at ({gx}, {gy}), size: {w}x{h}')
