import cv2
import numpy as np

img = cv2.imread('current_actual_screen.png')

nodes = [
    ("node_1", 293, 2051),
    ("node_2", 279, 1710),
    ("node_3", 818, 1673),
    ("node_4", 463, 1563),
    ("target_1289", 822, 1289),
    ("node_1106", 641, 1106),
    ("node_1045", 509, 1045),
    ("node_890", 541, 890),
]

for name, x, y in nodes:
    crop = img[max(0, y-80):min(2436, y+80), max(0, x-80):min(1080, x+80)]
    cv2.imwrite(rf'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\{name}.png', crop)
    print(f'Saved {name} at ({x}, {y})')
