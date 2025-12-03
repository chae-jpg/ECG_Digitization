import os
import cv2
import torch
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.dataset_v3 import RobustNormalizer 
from signal_extraction import SignalExtractor 
from tqdm import tqdm
from eval_snr import compute_snr


# --- 설정 ---
CONFIG = {
    "MODEL_PATH": "./checkpoints/unet_ecg_epoch_15.pth", # 모델 경로
    "TEST_DIR": "/Volumes/Untitled/ML/physionet-ecg-image-digitization/test", # 대회 테스트셋 경로
    "OUTPUT_CSV": "submission.csv",
    "IMG_SIZE": (512, 1024),
    "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "CLASSES": 13,
    "TARGET_LEN": 5000 # 10초 데이터 기준 (500Hz)
}

def main():
    # 1. 모델 로드
    print("Loading Model...")
    model = smp.Unet(encoder_name="efficientnet-b0", in_channels=1, classes=CONFIG["CLASSES"]).to(CONFIG['DEVICE'])
    model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
    model.eval()
    
    # 2. 유틸리티 초기화
    
    normalizer = RobustNormalizer(target_size=(1000, 1500)) 
    extractor = SignalExtractor(img_shape=CONFIG["IMG_SIZE"], total_len=CONFIG["TARGET_LEN"])
    
    transform = A.Compose([
        A.Resize(height=CONFIG['IMG_SIZE'][0], width=CONFIG['IMG_SIZE'][1]),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    
    # 3. 테스트 파일 리스트
    img_files = []
    for root, dirs, files in os.walk(CONFIG["TEST_DIR"]):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith("._"):
                img_files.append(os.path.join(root, f))
    
    print(f"Processing {len(img_files)} images...")
    
    # 결과 담을 리스트
    submission_rows = []
    
    # 4. 루프 시작
    for path in tqdm(img_files):
        try:
            # ID 추출
            record_id = os.path.splitext(os.path.basename(path))[0]
            
            # 1. 모델 로드
            print("Loading Model...")
            model = smp.Unet(encoder_name="efficientnet-b0", in_channels=1, classes=CONFIG["CLASSES"]).to(CONFIG['DEVICE'])
            model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
            model.eval()
            
            normalizer = RobustNormalizer(target_size=(1000, 1500)) 
            extractor = SignalExtractor(img_shape=CONFIG["IMG_SIZE"], total_len=CONFIG["TARGET_LEN"])
            
            transform = A.Compose([
                A.Resize(height=CONFIG['IMG_SIZE'][0], width=CONFIG['IMG_SIZE'][1]),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ])
            
            # 3. 테스트 파일 리스트
            img_files = []
            for root, dirs, files in os.walk(CONFIG["TEST_DIR"]):
                for f in files:
                    if f.lower().endswith(('.png')) and not f.startswith("._"):
                        img_files.append(os.path.join(root, f))
            
            print(f"Processing {len(img_files)} images...")

            # (1) 이미지 로드
            raw_img = cv2.imread(path)
            if raw_img is None: continue
            
            # (2) 전처리 (Alignment - 펴기/자르기/회전)
            aligned_img, status = normalizer.process(raw_img)
            
            if aligned_img is None:
                # 전처리 실패 시 다음 이미지로 넘어감 (혹은 에러 로그)
                continue

            # (3) 모델 입력 변환 (Grayscale -> Tensor)
            
            if len(aligned_img.shape) == 3:
                gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = aligned_img
            
            # Albumentations 적용 & 텐서 변환
            # (1, 1, H, W) 형태로 만듦
            input_tensor = transform(image=gray)['image'].unsqueeze(0).to(CONFIG['DEVICE']).float()
            
            # (4) 모델 추론 (Inference)
            with torch.no_grad():
                output = model(input_tensor) 
                
                # Softmax로 확률 계산 -> Argmax로 가장 높은 확률의 클래스 선택
                # 결과: (H, W) 크기의 0~12 정수 배열
                pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # 결과 담을 리스트
            submission_rows = []
            
            # (4) 신호 추출 (Digitization) 완료
            signals = extractor.extract_signal(pred_mask)
            
            # =================================================================
            # SNR 점수 계산 코드 추가 (정답 파일이 있을 때만)
            # =================================================================
            
            truth_csv_path = f"/Volumes/Untitled/ML/physionet-ecg-image-digitization/train/{record_id}/{record_id}.csv" 
            # f"/path/to/train/{record_id}.csv"

            if os.path.exists(truth_csv_path):
                try:
                    df_true = pd.read_csv(truth_csv_path)
                    snr_list = []
                    valid_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                    
                    for lead in valid_leads:
                        # 정답 데이터 가져오기 (NaN 제거)
                        if lead not in df_true.columns: continue
                        true_sig = df_true[lead].dropna().values
                        
                        # 예측 데이터 가져오기
                        pred_sig = signals[lead]
                        
                        # 길이가 다르면 정답 길이에 맞춤 (SignalExtractor가 이미 5000개로 맞췄겠지만 안전장치)
                        if len(pred_sig) != len(true_sig):
                            # 보통 정답이 5000개면 예측도 5000개여야 함.
                            # 만약 정답이 짧다면(grid 부분만 있는 경우 등), 그 길이에 맞춰 잘라야 함
                            min_len = min(len(pred_sig), len(true_sig))
                            pred_sig = pred_sig[:min_len]
                            true_sig = true_sig[:min_len]

                        # --- [SNR 계산 함수 호출] ---
                        
                        current_snr = compute_snr(true_sig, pred_sig)
                        snr_list.append(current_snr)
                    
                    # 평균 SNR 출력
                    if snr_list:
                        avg_snr = np.mean(snr_list)
                        
                        tqdm.write(f"[{record_id}] SNR Score: {avg_snr:.2f} dB")
                        
                except Exception as snr_error:
                    tqdm.write(f"⚠️ SNR Calc Error for {record_id}: {snr_error}")

            # =================================================================
            
            # (5) Submission 포맷으로 변환 (기존 코드)
            leads_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            for lead in leads_order:
                sig_data = signals[lead]
                for idx, val in enumerate(sig_data):
                    row_id = f"{record_id}_{idx}_{lead}"
                    submission_rows.append({'id': row_id, 'value': val})
                    
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    # 5. CSV 저장
    print("Saving CSV...")
    df = pd.DataFrame(submission_rows)
    df.to_csv(CONFIG["OUTPUT_CSV"], index=False)
    print("Done!")

if __name__ == "__main__":
    main()