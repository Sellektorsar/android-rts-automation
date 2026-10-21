import cv2
import numpy as np

img = cv2.imread('current_actual_screen.png')

# Let's inspect the entire map to find all node icons, numbers, or targets.
# In "Спаситель порядка", each stage is marked on a winding road/path.
# Let's crop vertical slices along the path and check what is drawn there.

# Let's save a composite debug visualization with annotated detections.
debug = img.copy()

# Find all high-contrast circular or rectangular elements in the central area (x: 100..980, y: 300..2100)
# Let's find contours of distinctive colors:
# Red swords, yellow rings, green checkmarks, level numbers.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Let's crop 5 key regions:
# Region 1: Around the dialogue bubble and boss
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\debug_region1_top.png', img[300:800, :])
# Region 2: Upper middle (800..1300)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\debug_region2_mid_upper.png', img[800:1300, :])
# Region 3: Center (1200..1700)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\debug_region3_mid_lower.png', img[1200:1700, :])
# Region 4: Lower (1600..2100)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\debug_region4_bottom.png', img[1600:2100, :])

print('Debug regions saved!')
