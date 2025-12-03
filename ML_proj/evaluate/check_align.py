import cv2
import matplotlib.pyplot as plt
import os
import numpy as np 

# 확인하고 싶은 ID 입력
TEST_ID = "1006427285" 

IMG_PATH = f"./dataset/images_aligned/{TEST_ID}.png"
MASK_PATH = f"./dataset/masks_v2/{TEST_ID}.png"

# 파일이 없으면 검색
if not os.path.exists(IMG_PATH):
    for f in os.listdir("./dataset/images_aligned"):
        if f.startswith(TEST_ID):
            IMG_PATH = os.path.join("./dataset/images_aligned", f)
            break

print(f"이미지 경로: {IMG_PATH}")
print(f"마스크 경로: {MASK_PATH}")

img = cv2.imread(IMG_PATH)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE) # 마스크는 흑백으로 로드

if img is None or mask is None:
    print("파일을 찾을 수 없습니다. 경로를 확인하세요.")
else:
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 마스크 크기를 이미지 크기에 맞춤
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = img.copy()
    
    
    overlay[mask > 0] = [0, 0, 255] 

    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.title("Aligned Image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Overlay (Red=Mask) : {TEST_ID}")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    plt.show()
    
    print("이미지 창을 확인하세요.")