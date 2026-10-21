import cv2
import numpy as np

for name in ["node_1", "node_2", "node_3", "node_4", "target_1289", "node_1106", "node_1045", "node_890"]:
    path = rf'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\{name}.png'
    img = cv2.imread(path)
    # Check center 60x60
    h, w = img.shape[:2]
    c = img[h//2-30:h//2+30, w//2-30:w//2+30]
    bgr = c.mean(axis=(0,1)).astype(int)
    # Check if there is red (e.g. current stage marker, glowing target)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    print(f'{name:12s}: center BGR={bgr}, red_px={np.count_nonzero(red_mask)}, yellow_px={np.count_nonzero(yellow_mask)}')
