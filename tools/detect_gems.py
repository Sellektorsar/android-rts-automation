import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\step3_after_tap_challenge.png')
# Center of screen is (540, 1200)
# Let's inspect the colors of crystals / gems in survivor games
# In WaO 'Спаситель порядка', what color are the EXP crystals?
# They are typically cyan/blue, green, or yellow glowing dots/gems on the ground.
# Let's check what colors exist on the battlefield ground around the hero.
roi = img[600:1800, 100:980]
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Blue/cyan gems:
cyan_mask = cv2.inRange(hsv, (80, 100, 100), (105, 255, 255))
# Green gems:
green_mask = cv2.inRange(hsv, (40, 100, 100), (75, 255, 255))
# Purple/pink gems:
purple_mask = cv2.inRange(hsv, (130, 100, 100), (160, 255, 255))

print('Cyan pixels:', np.count_nonzero(cyan_mask))
print('Green pixels:', np.count_nonzero(green_mask))
print('Purple pixels:', np.count_nonzero(purple_mask))
