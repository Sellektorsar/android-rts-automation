import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\current_menu_screen.png')

# Let's find yellow/golden pulsing highlights on current_menu_screen.png
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_yellow = cv2.inRange(hsv, (15, 100, 140), (35, 255, 255))
contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('Found yellow highlights:')
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h > 400:
        print(f'Highlight at ({x+w//2}, {y+h//2}), size: {w}x{h}')
