import cv2

img = cv2.imread(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_radial_test.png')
# Crop radial menu (y: 1000..1600, x: 200..900)
radial = img[1000:1600, 200:900]
cv2.imwrite(r'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\barracks_radial_crop.png', radial)

# Check buttons in the radial menu
# In other buildings, radial menu buttons are placed radially:
# Left/Up: ~ (370, 1150) or (387, 1312)
# Center/Top: ~ (540, 1050)
# Right/Top: ~ (690, 1150) or (690, 1311)
# Center/Bottom: ~ (540, 1450)
print('Radial crop saved!')
