import cv2
import numpy as np

for name, path in [('stage5', 'stage5_destination_center.png'), ('stage2', 'stage2_crop.png')]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    # Center pixel and color
    cx, cy = w // 2, h // 2
    center_roi = img[cy-30:cy+30, cx-30:cx+30]
    print(f'{name}: shape={img.shape}, center mean BGR={center_roi.mean(axis=(0,1)).astype(int)}')
    # Let's save a thumbnail or find template matching in current screen
    # Find match of stage2 or stage5 in current screen
    curr = cv2.imread('current_actual_screen.png')
    res = cv2.matchTemplate(curr, img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(f'{name} match in current screen: max_val={max_val:.3f} at loc={max_loc}')
