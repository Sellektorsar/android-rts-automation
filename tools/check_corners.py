import cv2

for name in ['corner_back.png', 'corner_x.png']:
    img = cv2.imread(rf'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\{name}')
    print(name, 'mean BGR:', img.mean(axis=(0,1)).astype(int))
