import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RealAlignedDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(512, 1024), is_train=True):
        """
        img_dir: 전처리된 실제 이미지가 있는 곳 (dataset/images_aligned)
        mask_dir: CSV로 만든 마스크가 있는 곳 (dataset/masks_v2)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # 이미지 파일 리스트 (.png, .jpg)
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg')) and not f.startswith('._')]
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.RandomBrightnessContrast(p=0.5),
                
                # [수정] 파라미터 제거
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
                # Affine이나 ShiftScaleRotate 중 에러 안 나는 걸로 유지
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
                    scale=(0.95, 1.05),                                         
                    rotate=(-5, 5),                                             
                    p=0.5
                ),
                
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 1. 확장자부터 뗀다 (.png, .jpg 등 제거)
        filename_no_ext = os.path.splitext(img_name)[0]
        
        # 2. 그 다음 하이픈(-)이나 언더바(_) 기준으로 ID만 남김
        file_id = filename_no_ext.split('-')[0].split('_')[0]
        
        # 3. 마스크 경로 찾기
        mask_name = f"{file_id}.png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 4. 로드
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 예외처리 (로드 실패 시 다음 데이터로 넘김)
        if image is None:
            # print(f"[Skip] Image Load Failed: {img_path}")
            return self.__getitem__((idx + 1) % len(self))
            
        if mask is None:
            # print(f"[Skip] Mask Not Found: {mask_path}") 
            # 여기서 .png.png 에러가 나면 None이 반환되어 이쪽으로 옴
            return self.__getitem__((idx + 1) % len(self))

        # --- 마스크 크기 맞추기 ---
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        try:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].long()
            return image_tensor, mask_tensor
            
        except ValueError as e:
            print(f"[Augmentation Error] {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))