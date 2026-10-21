import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\target_1289.png')
# Find the exact center of the glowing circle/shield
# The crop is 160x160 centered at (822, 1289)
# Let's find the centroid of the brightest / most saturated pixels
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (15, 80, 150), (35, 255, 255))
M = cv2.moments(mask)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    global_x = 822 - 80 + cx
    global_y = 1289 - 80 + cy
    print(f'Exact glowing center is at: ({global_x}, {global_y}) [local: ({cx}, {cy})]')
else:
    print('No mask found')
