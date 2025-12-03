import torch
import cv2
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import resample, find_peaks
from torch.utils.data import Dataset
import sys


sys.path.append(os.path.abspath(".")) 

try:
    from dataset.dataset_v1 import UNet
except ImportError:
    
    import torch.nn as nn
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            def CBR(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.enc1 = CBR(1, 64)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = CBR(64, 128)
            self.pool2 = nn.MaxPool2d(2)
            self.enc3 = CBR(128, 256)
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = CBR(256, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = CBR(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = CBR(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = CBR(128, 64)
            self.final = nn.Conv2d(64, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool1(enc1))
            enc3 = self.enc3(self.pool2(enc2))
            bottleneck = self.bottleneck(self.pool3(enc3))
            dec3 = self.up3(bottleneck)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.dec3(dec3)
            dec2 = self.up2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.dec2(dec2)
            dec1 = self.up1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.dec1(dec1)
            return self.sigmoid(self.final(dec1))

# 전처리 파이프라인 
try:
    from preprocess_clustering.clustering_v3 import segment_ecg_pipeline
except ImportError:
    print("전처리 모듈(clustering_v3)을 찾을 수 없습니다. 기본 흑백 변환만 사용합니다.")
    def segment_ecg_pipeline(img, use_kmeans=True):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray, gray, gray

# ==========================================
# 1. 1D 신호 변환 (벡터화)
# ==========================================
def image_to_signal_vectorized(mask_image):
    h, w = mask_image.shape
    binary_mask = mask_image > 0.5
    col_counts = np.sum(binary_mask, axis=0)
    
    if np.sum(col_counts) == 0:
        return np.zeros(w)

    y_indices = np.arange(h).reshape(-1, 1)
    weighted_sum = np.sum(binary_mask * y_indices, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        center_y = np.divide(weighted_sum, col_counts)
    
    center_y[col_counts == 0] = np.nan
    signal = h - center_y # Y축 반전
    
    s = pd.Series(signal)
    signal = s.interpolate(method='linear', limit_direction='both').fillna(0).values
    return signal

# ==========================================
# 2. 이어붙이기(Stitching) 예측 함수
# ==========================================
def predict_full_signal(model, image, device, target_h=128, crop_w=256):
    # 전처리 (흑백 + 높이 리사이즈)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape
    image = cv2.resize(image, (w, target_h))
    
    full_mask = []
    model.eval()
    
    with torch.no_grad():
        for x in range(0, w, crop_w):
            chunk = image[:, x : x + crop_w]
            
            curr_h, curr_w = chunk.shape
            if curr_w < crop_w:
                pad_w = crop_w - curr_w
                chunk = np.pad(chunk, ((0,0), (0, pad_w)), mode='constant')
            else:
                pad_w = 0
            
            input_tensor = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0) / 255.0
            input_tensor = input_tensor.to(device)
            
            output = model(input_tensor)
            output = torch.sigmoid(output)
            pred_chunk = output.squeeze().cpu().numpy()
            
            if pad_w > 0:
                pred_chunk = pred_chunk[:, :-pad_w]
            
            full_mask.append(pred_chunk)
            
    full_mask = np.hstack(full_mask)
    signal = image_to_signal_vectorized(full_mask)
    
    return signal, full_mask

# ==========================================
# 3. 평가 데이터셋
# ==========================================
class FullImageDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
        self.lead_layout = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
        print(f"[{data_dir}] 평가 파일 수: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        
        filename = os.path.basename(img_path)
        file_id = filename.split("-")[0].strip()
        parent_dir = os.path.dirname(img_path)
        csv_path = os.path.join(parent_dir, f"{file_id}.csv")
        
        if not os.path.exists(csv_path): return None
        
        img = cv2.imread(img_path)
        if img is None: return None
        
        try:
            processed, _, _ = segment_ecg_pipeline(img, use_kmeans=True)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_inv = cv2.bitwise_not(gray)
            _, th = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            proj = np.sum(th, axis=1)
            peaks, _ = find_peaks(proj, distance=150, height=np.max(proj)*0.3)
            peaks = sorted(peaks)
            
            if len(peaks) < 3: return None
            
        except:
            return None

        # Lead II 
        row_idx, col_idx = 1, 0 
        lead_name = self.lead_layout[row_idx][col_idx] 
        
        boundaries = [0]
        for i in range(len(peaks)-1): boundaries.append((peaks[i] + peaks[i+1]) // 2)
        boundaries.append(img.shape[0])
        
        y_start, y_end = boundaries[row_idx], boundaries[row_idx+1]
        row_img = processed[y_start:y_end, :]
        
        h, w = row_img.shape
        seg_w = w // 4
        sub_img = row_img[:, col_idx*seg_w : (col_idx+1)*seg_w]
        
        return sub_img, csv_path, lead_name

# ==========================================
# 4. 실행 메인
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_PATH = "ecg_unet_onthefly_final.pth"
    DATA_PATH = "./test_tmp" 
    
    model = UNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"📂 모델 로드: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        exit()
        
    dataset = FullImageDataset(DATA_PATH)
    scores = []
    
    print("🚀 전체 파형 복원 및 평가 시작...")
    
    for i in range(len(dataset)):
        data = dataset[i]
        if data is None: continue
        
        sub_img, csv_path, lead_name = data
        
        # 1. 이어붙이기 예측
        pred_signal, pred_mask_full = predict_full_signal(model, sub_img, DEVICE)
        
        # 2. CSV 로드
        try:
            df = pd.read_csv(csv_path)
            if lead_name in df.columns:
                gt_signal = df[lead_name].values
            else:
                gt_signal = df.iloc[:, 1].values
        except:
            continue
            
        # 3. 길이 맞추기 & 평가
        if len(gt_signal) > 0 and len(pred_signal) > 0:
            pred_resampled = resample(pred_signal, len(gt_signal))
            
            if np.std(pred_resampled) == 0:
                corr = 0.0 # Flat line case
            else:
                corr, _ = pearsonr(gt_signal, pred_resampled)
                
            if not np.isnan(corr):
                scores.append(corr)
                print(f"[{i}] Lead: {lead_name} | Corr: {corr:.4f} | File: {os.path.basename(csv_path)}")

    if len(scores) > 0:
        print(f"\n====== 최종 평균 점수 ======")
        print(f"Average Correlation: {np.mean(scores):.4f}")
    else:
        print("\n평가 결과가 없습니다. 데이터나 모델을 확인하세요.")