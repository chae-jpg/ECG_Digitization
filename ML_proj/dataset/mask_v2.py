import pandas as pd
import numpy as np
import cv2
import os

def generate_ecg_mask(csv_path, img_shape=(1000, 1500)):
    """
    csv_path: 개별 ECG 데이터 파일 경로
    img_shape: (Height, Width) - 전처리 코드(RobustNormalizer)와 동일해야 함!
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None, None
        
    file_id = os.path.basename(csv_path).split('.')[0]
    
    # 2. 마스크 초기화 
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    PAD_X = 50
    PAD_Y = 50
    HEADER_H = 100
    
    # 실제 격자가 그려질 영역 
    effective_h = H - PAD_Y - HEADER_H
    effective_w = W - (PAD_X * 2)
    
    grid_h = int(effective_h * 0.75)
    strip_h = effective_h - grid_h
    
    cell_h = grid_h // 3
    cell_w = effective_w // 4
    
    SCALE_Y = 150 
    
    # 클래스 ID 매핑
    lead_to_id = {
        'I': 1, 'II': 2, 'III': 3, 
        'aVR': 4, 'aVL': 5, 'aVF': 6,
        'V1': 7, 'V2': 8, 'V3': 9, 
        'V4': 10, 'V5': 11, 'V6': 12
    }
    
    # 3x4 배치 좌표
    layout_mapping = {
        'I': (0, 0), 'aVR': (0, 1), 'V1': (0, 2), 'V4': (0, 3),
        'II': (1, 0), 'aVL': (1, 1), 'V2': (1, 2), 'V5': (1, 3),
        'III': (2, 0), 'aVF': (2, 1), 'V3': (2, 2), 'V6': (2, 3)
    }

    for col in df.columns:
        raw_data = df[col].dropna()
        if raw_data.empty: continue
        
        class_id = lead_to_id.get(col, 0)
        if class_id == 0: continue

        # ---------------------------------------------------------
        # [Job 1] 3x4 그리드 영역
        # ---------------------------------------------------------
        if col in layout_mapping:
            row_idx, col_idx = layout_mapping[col]
            
            if len(raw_data) > 2000:
                grid_data = raw_data.iloc[:1250].values
            else:
                grid_data = raw_data.values
            
            grid_data = grid_data - np.mean(grid_data)

            base_y = HEADER_H + (row_idx * cell_h) + (cell_h // 2)
            base_x_start = PAD_X + (col_idx * cell_w)
            
            x_coords = np.linspace(base_x_start + 10, base_x_start + cell_w - 10, len(grid_data))
            y_coords = base_y - (grid_data * SCALE_Y)
            
            points = np.column_stack((x_coords, y_coords)).astype(np.int32)
            cv2.polylines(mask, [points], isClosed=False, color=class_id, thickness=5)

        # ---------------------------------------------------------
        # [Job 2] 하단 리듬 스트립 (Lead II)
        # ---------------------------------------------------------
        
        if col == 'II' and len(raw_data) > 2000:
            base_y = HEADER_H + grid_h + (strip_h // 2)
            
            strip_data = raw_data.values
            strip_data = strip_data - np.mean(strip_data)
            
            x_coords = np.linspace(PAD_X + 20, W - PAD_X - 20, len(strip_data))
            y_coords = base_y - (strip_data * SCALE_Y)
            
            points = np.column_stack((x_coords, y_coords)).astype(np.int32)
            cv2.polylines(mask, [points], isClosed=False, color=class_id, thickness=5)
            
    return mask, file_id

# --- 실행 ---
if __name__ == "__main__":
    MASK_DIR = "/Volumes/Untitled/ML/physionet-ecg-image-digitization/train"
    OUTPUT_DIR = "./dataset/masks_v2"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    csv_files = []
    for root, dir, file in os.walk(MASK_DIR):
        for f in file:
            if f.lower().endswith(".csv") and not f.startswith("._") and f != "train.csv":
                csv_files.append(os.path.join(root, f))

    print(f"Found {len(csv_files)} CSV files. Generating masks...")

    for path in csv_files:
        try:
            sample_mask, id = generate_ecg_mask(path, img_shape=(1000, 2000))
            if sample_mask is not None:
                cv2.imwrite(f"{OUTPUT_DIR}/{id}.png", sample_mask)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            
    print("Mask generation completed.")