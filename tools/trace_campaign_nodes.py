import cv2
import numpy as np

img = cv2.imread('current_actual_screen.png')
# Stage 5 was at (830, 1605)
# Let's crop a wide area around (830, 1605) up towards y=900
# And let's find all circular nodes or landmarks along the road!

# Let's crop x: 200..1000, y: 800..1800
road_crop = img[800:1800, 200:1000]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\campaign_road_crop.png', road_crop)

# Let's inspect coordinates of interest:
# (830, 1605) was Stage 5
# What is directly above or connected to Stage 5?
# Let's check colors in 50x50 patches:
points = [
    (830, 1605, "Stage 5"),
    (830, 1400, "Above 5"),
    (730, 1300, "Mid 1300"),
    (822, 1289, "Yellow 1289"),
    (760, 1180, "Prev guess 1180"),
    (732, 1159, "Prev guess 1159"),
    (540, 1050, "Center 1050"),
    (350, 1100, "Left 1100"),
    (540, 1200, "Center 1200"),
]

for x, y, label in points:
    patch = img[y-20:y+20, x-20:x+20]
    bgr = patch.mean(axis=(0,1)).astype(int)
    print(f'{label:20s} at ({x}, {y}): mean BGR={bgr}')
