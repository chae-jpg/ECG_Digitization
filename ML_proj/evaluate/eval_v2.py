import os
import cv2
import torch
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.dataset_v3 import RobustNormalizer
from evaluate.signal_extraction import SignalExtractor # root 디렉에서 실행한다는 전제 하에
from evaluate.eval_snr import compute_snr

# --- [설정] ---
CONFIG = {
    
    "MODEL_PATH": "./checkpoints/unet_ecg_epoch_15.pth", 
    
    "TEST_DIR": "/Users/koscom/Desktop/ML_proj/test_tmp", 
    # 실전 테스트용: "/Volumes/Untitled/ML/physionet-ecg-image-digitization/test"
    
    "OUTPUT_CSV": "submission.csv",
    
    "IMG_SIZE": (512, 1024),
    "TARGET_LEN": 5000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "CLASSES": 13
}

def main():
    # 1. 모델 로드
    print(f"Loading Model from {CONFIG['MODEL_PATH']}...")
    model = smp.Unet(encoder_name="efficientnet-b0", in_channels=1, classes=CONFIG["CLASSES"]).to(CONFIG['DEVICE'])
    try:
        model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
    except FileNotFoundError:
        print("모델 파일이 없습니다. 경로를 확인하세요.")
        return
    model.eval()
    
    # 2. 도구 초기화
    normalizer = RobustNormalizer(target_size=(1000, 1500)) 
    extractor = SignalExtractor(img_shape=CONFIG["IMG_SIZE"], total_len=CONFIG["TARGET_LEN"])
    
    transform = A.Compose([
        A.Resize(height=CONFIG['IMG_SIZE'][0], width=CONFIG['IMG_SIZE'][1]),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    
    # 3. 파일 리스트 확보
    img_files = []
    
    for root, dirs, files in os.walk(CONFIG["TEST_DIR"]):
        for f in files:
            if f.lower().endswith(('.png')) and not f.startswith("._"):
                img_files.append(os.path.join(root, f))
    
    print(f"Processing {len(img_files)} images...")
    
    # 4. CSV 파일 초기화 
    with open(CONFIG["OUTPUT_CSV"], 'w') as f:
        f.write("id,value\n")
    
    # SNR 점수 기록용
    total_snr_scores = []

    # 5. 메인 루프
    for path in tqdm(img_files):
        try:
            # (1) ID 추출
            filename_no_ext = os.path.splitext(os.path.basename(path))[0]
            record_id = filename_no_ext.split('-')[0].split('_')[0]
            
            # (2) 이미지 로드 & 전처리
            raw_img = cv2.imread(path)
            if raw_img is None: continue
            
            # 전처리 
            aligned_img, status = normalizer.process(raw_img)
            if aligned_img is None: continue # 실패 시 건너뜀

            # (3) 모델 추론
            if len(aligned_img.shape) == 3:
                gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = aligned_img
                
            input_tensor = transform(image=gray)['image'].unsqueeze(0).to(CONFIG['DEVICE']).float()
            
            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # (4) 신호 추출 (Pixel -> Voltage)
            signals = extractor.extract_signal(pred_mask)
            
            # (5) SNR 점수 계산 
            truth_csv_path = os.path.join(os.path.dirname(path), f"{record_id}.csv")

            if os.path.exists(truth_csv_path):
                df_true = pd.read_csv(truth_csv_path)
                leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                current_snr_list = []
                
                for lead in leads:
                    if lead in df_true.columns:
                        true_sig = df_true[lead].dropna().values
                        pred_sig = signals[lead]
                        
                        # 길이 맞춤 
                        min_len = min(len(true_sig), len(pred_sig))
                        score = compute_snr(true_sig[:min_len], pred_sig[:min_len])
                        current_snr_list.append(score)
                
                if current_snr_list:
                    avg_snr = np.mean(current_snr_list)
                    total_snr_scores.append(avg_snr)
                    
                    tqdm.write(f"[{record_id}] SNR: {avg_snr:.2f} dB")

            # (6) CSV 파일에 쓰기 
            with open(CONFIG["OUTPUT_CSV"], 'a') as f:
                leads_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                for lead in leads_order:
                    sig_data = signals[lead]
                    for idx, val in enumerate(sig_data):
                        
                        row_id = f"{record_id}_{idx}_{lead}"
                        f.write(f"{row_id},{val}\n")
                        
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # 최종 결과 출력
    if total_snr_scores:
        print(f"\n======== Final Result ========")
        print(f"Average SNR: {np.mean(total_snr_scores):.4f} dB")
        print(f"==============================")
    
    print(f"Submission file saved to {CONFIG['OUTPUT_CSV']}")

if __name__ == "__main__":
    main()