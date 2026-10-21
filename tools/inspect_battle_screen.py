import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\step3_after_tap_challenge.png')
print('Battle screen shape:', img.shape)

# Let's crop joystick area (y: 1700..2300, x: 250..850)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\joystick_area.png', img[1700:2300, 250:850])
# Let's crop center field (hero)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\field_center.png', img[800:1500, 300:800])
# Let's crop top HUD (health / timer / quest)
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\top_battle_hud.png', img[50:600, :])
print('Saved crops from battle screen!')
