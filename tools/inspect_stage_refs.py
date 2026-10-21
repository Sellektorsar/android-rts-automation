import cv2
import numpy as np

img = cv2.imread('stage5_destination_center.png')
if img is not None:
    print('stage5_destination_center shape:', img.shape)
    cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\stage5_ref.png', img)

img2 = cv2.imread('stage2_crop.png')
if img2 is not None:
    print('stage2_crop shape:', img2.shape)
    cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\stage2_ref.png', img2)
