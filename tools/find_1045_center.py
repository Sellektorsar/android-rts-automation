import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\node_1045.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (15, 80, 150), (35, 255, 255))
M = cv2.moments(mask)
if M["m00"] > 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    global_x = 509 - 80 + cx
    global_y = 1045 - 80 + cy
    print(f'node_1045 center is at: ({global_x}, {global_y}) [local: ({cx}, {cy})]')
else:
    print('No mask found for 1045')
