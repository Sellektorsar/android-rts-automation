import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_radial_crop.png')
# Find circular icon buttons (typically radius 40..60)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=50, param2=30, minRadius=35, maxRadius=70)
print('Found buttons in radial crop:')
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        gx = 200 + x
        gy = 1000 + y
        print(f"Button at ({gx}, {gy}), radius={r}")
