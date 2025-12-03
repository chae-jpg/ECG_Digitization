import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class SyntheticECGDataset(Dataset):
    def __init__(self, csv_dir, img_size=(1000, 1500), is_train=True, expand_factor=1):
        """
        [수정됨] expand_factor: 데이터 뻥튀기 배수 (기본값 1)
        """
        self.csv_files = []
        # 하위 폴더까지 뒤져서 CSV 찾기
        for root, dirs, files in os.walk(csv_dir):
            for f in files:
                if f.endswith('.csv') and not f.startswith('._') and f != "train.csv":
                    self.csv_files.append(os.path.join(root, f))
        
        self.img_size = img_size
        self.expand_factor = expand_factor # [핵심] 배수 저장
        
        # 확인용 로그
        if is_train:
            print(f"[Synthetic] 원본 CSV: {len(self.csv_files)}개 -> 학습 데이터: {len(self.csv_files) * expand_factor}개 (x{expand_factor}배)")

        # 리드 이름 -> 클래스 ID 매핑
        self.lead_to_id = {
            'I': 1, 'II': 2, 'III': 3, 'aVR': 4, 'aVL': 5, 'aVF': 6,
            'V1': 7, 'V2': 8, 'V3': 9, 'V4': 10, 'V5': 11, 'V6': 12
        }
        
        # 3x4 배치 레이아웃
        self.layout_mapping = {
            'I': (0, 0), 'aVR': (0, 1), 'V1': (0, 2), 'V4': (0, 3),
            'II': (1, 0), 'aVL': (1, 1), 'V2': (1, 2), 'V5': (1, 3),
            'III': (2, 0), 'aVF': (2, 1), 'V3': (2, 2), 'V6': (2, 3)
        }

        # Augmentation 정의
        if is_train:
            self.transform = A.Compose([
                # 1. 형태 왜곡
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
                A.Rotate(limit=2, p=0.5),
                A.Perspective(scale=(0.01, 0.05), p=0.3),
                
                A.GaussNoise(p=0.5), 
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
                A.CoarseDropout(
                    num_holes_range=(5, 15),       
                    hole_height_range=(5, 15),     
                    hole_width_range=(5, 15),      
                    fill=255,                      
                    p=0.5
                ),

                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])

    def __len__(self):
        
        return len(self.csv_files) * self.expand_factor

    def draw_ecg(self, df):
        # 캔버스 설정
        H, W = 1000, 2000
        img = np.ones((H, W), dtype=np.uint8) * 255
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # 레이아웃
        grid_h = int(H * 0.75)
        strip_h = H - grid_h
        cell_h = grid_h // 3
        cell_w = W // 4
        
        SCALE_Y = 150 
        
        for col in df.columns:
            raw_data = df[col].dropna()
            if raw_data.empty: continue
            
            class_id = self.lead_to_id.get(col, 0)
            if class_id == 0: continue
            
            # ---------------------------------------------------------
            # [JOB 1] 3x4 그리드 영역 (모든 리드)
            # ---------------------------------------------------------
            if col in self.layout_mapping:
                row_idx, col_idx = self.layout_mapping[col]
                
                # 데이터 자르기 (Lead II 같이 긴 데이터는 앞부분만)
                if len(raw_data) > 2000:
                    grid_data = raw_data.iloc[:1250].values # numpy array로 변환
                else:
                    grid_data = raw_data.values

                grid_data = grid_data - np.mean(grid_data)

                base_y = row_idx * cell_h + (cell_h // 2)
                base_x_start = col_idx * cell_w
                
                x = np.linspace(base_x_start + 10, base_x_start + cell_w - 10, len(grid_data))
                y = base_y - (grid_data * SCALE_Y)
                
                points = np.column_stack((x, y)).astype(np.int32)
                
                cv2.polylines(img, [points], isClosed=False, color=0, thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(mask, [points], isClosed=False, color=class_id, thickness=7, lineType=cv2.LINE_AA)

            # ---------------------------------------------------------
            # [JOB 2] 하단 리듬 스트립 (Lead II만)
            # ---------------------------------------------------------
            if col == 'II' and len(raw_data) > 2000:
                base_y = grid_h + (strip_h // 2)
                
                strip_data = raw_data.values
                
                strip_data = strip_data - np.mean(strip_data)
                
                x = np.linspace(20, W - 20, len(strip_data))
                y = base_y - (strip_data * SCALE_Y)
                
                points = np.column_stack((x, y)).astype(np.int32)
                
                cv2.polylines(img, [points], isClosed=False, color=0, thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(mask, [points], isClosed=False, color=class_id, thickness=7, lineType=cv2.LINE_AA)
                
        return img, mask

    def __getitem__(self, idx):
        real_idx = idx % len(self.csv_files)
        
        try:
            csv_path = self.csv_files[real_idx]
            df = pd.read_csv(csv_path)
            
            # 그리기
            clean_img, clean_mask = self.draw_ecg(df)
            
            # 변형 
            augmented = self.transform(image=clean_img, mask=clean_mask)
            
            final_img = augmented['image']
            final_mask = augmented['mask'].long()
            
            return final_img, final_mask
            
        except Exception as e:
            print(f"Error loading {self.csv_files[real_idx]}: {e}")
            return torch.zeros((1, self.img_size[0], self.img_size[1])), torch.zeros(self.img_size, dtype=torch.long)