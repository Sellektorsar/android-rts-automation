import cv2
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
for name in ["node_1", "node_2", "node_3", "node_4", "target_1289", "node_1106", "node_1045", "node_890"]:
    path = rf'C:\Users\Иван\.gemini\antigravity\brain\dc85771f-4cb0-4f6f-af0a-fb79dcfd4d7a\{name}.png'
    img = cv2.imread(path)
    res, _ = ocr(img)
    print(f'=== {name} ===')
    if res:
        for r in res:
            print(f'  {r[1]} ({r[2]})')
    else:
        print('  No OCR')
