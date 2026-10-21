import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\hero_enhancement.png')
# Check top-left corner (0..200, 50..250)
corner = img[50:250, 0:200]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\corner_back.png', corner)

# Also let's check top-right corner (880..1080, 50..250) for 'X' button
corner_x = img[50:250, 880:1080]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\corner_x.png', corner_x)
print('Saved corner crops')
