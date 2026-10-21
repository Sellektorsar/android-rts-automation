import cv2

img = cv2.imread('screen_after_bella_step1.png')
# Save 4 quadrants around the center of the city:
# 1. (540, 1450) - Lower center
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\crop_1450.png', img[1250:1650, 350:750])
# 2. (540, 1050) - Upper center (Castle/College)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\crop_1050.png', img[850:1250, 350:750])
# 3. (300, 1250) - Left
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\crop_left.png', img[1050:1450, 100:500])
# 4. (780, 1250) - Right
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\crop_right.png', img[1050:1450, 600:1000])
print('Saved city crops!')
