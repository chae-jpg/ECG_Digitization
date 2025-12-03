import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from dataset.dataset_v3 import RobustNormalizer
    from evaluate.signal_extraction import SignalExtractor
    from evaluate.eval_snr import compute_snr
except ImportError:
    print("❌ 오류: dataset_v3.py, signal_extractor.py, evaluate_snr.py 파일이 현재 폴더에 있어야 합니다.")
    exit()

CONFIG = {
    "MODEL_PATH": "./checkpoints/unet_ecg_epoch_21.pth", 
    "TEST_DIR": "/Users/koscom/Desktop/ML_proj/test_tmp", # 정답 CSV가 있는 디렉
    "SAVE_IMG": "sample_comparison.png", # 결과 저장할 이미지 파일명
    
    "IMG_SIZE": (512, 1024),
    "TARGET_LEN": 5000,
    "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "CLASSES": 13
}

def visualize_one_sample():
    print(f"Loading Model... ({CONFIG['DEVICE']})")
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        in_channels=1,
        classes=CONFIG["CLASSES"]
    ).to(CONFIG['DEVICE'])

    try:
        model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
    except:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        return
    model.eval()

    # 도구 준비
    normalizer = RobustNormalizer(target_size=(1000, 1500))
    extractor = SignalExtractor(img_shape=CONFIG["IMG_SIZE"], total_len=CONFIG["TARGET_LEN"])
    transform = A.Compose([
        A.Resize(height=CONFIG['IMG_SIZE'][0], width=CONFIG['IMG_SIZE'][1]),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    print("Searching for a valid sample (Image + CSV)...")

    # 이미지 탐색
    for root, dirs, files in os.walk(CONFIG["TEST_DIR"]):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith("._"):
                img_path = os.path.join(root, f)

                # ID 추출
                filename_no_ext = os.path.splitext(f)[0]
                record_id = filename_no_ext.split('-')[0].split('_')[0]

                # 정답 CSV
                csv_path = os.path.join(root, f"{record_id}.csv")
                if not os.path.exists(csv_path):
                    continue

                print(f"✅ Found Sample: {record_id}")
                print(f"   Image: {img_path}")
                print(f"   CSV: {csv_path}")

                # --- 1. 이미지 정규화 / 모델 inference ---
                raw_img = cv2.imread(img_path)
                aligned_img, status = normalizer.process(raw_img)
                if aligned_img is None:
                    continue

                if len(aligned_img.shape) == 3:
                    gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = aligned_img

                input_tensor = transform(image=gray)['image'].unsqueeze(0).to(CONFIG['DEVICE']).float()

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_mask = torch.argmax(
                        torch.softmax(output, dim=1), dim=1
                    ).squeeze().cpu().numpy().astype(np.uint8)

                # --- 2. segmentation → waveform extraction ---
                signals = extractor.extract_signal(pred_mask)

                # --- 3. Ground truth CSV load ---
                df_true = pd.read_csv(csv_path)
                lead_to_plot = 'I'

                if lead_to_plot not in df_true.columns:
                    print(f"Lead {lead_to_plot} not in CSV. Trying another sample...")
                    continue

                true_sig = df_true[lead_to_plot].dropna().values
                pred_sig = signals[lead_to_plot]

                # 길이 맞춤
                min_len = min(len(true_sig), len(pred_sig))
                true_sig = true_sig[:min_len]
                pred_sig = pred_sig[:min_len]

                # --- 4. SNR ---
                snr = compute_snr(true_sig, pred_sig, fs=500)

                # --- 5. 시각화 저장 ---
                plt.figure(figsize=(15, 6))
                plt.plot(true_sig, color='black', label='Ground Truth (CSV)', linewidth=1.5, alpha=0.7)
                plt.plot(pred_sig, color='red', label='Prediction (Model)', linewidth=1.0, alpha=0.8)
                plt.title(f"Sample ID: {record_id} (Lead {lead_to_plot}) | SNR: {snr:.2f} dB")
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(CONFIG["SAVE_IMG"])

                print(f"그래프 저장 완료: {CONFIG['SAVE_IMG']}")
                print("파형 확인...")
                print(f"SNR (2025 standard): {snr:.2f} dB")

                plt.figure(figsize=(8, 4))      
                plt.imshow(pred_mask, cmap="tab20")
                plt.colorbar()
                plt.title(f"Predicted Mask (Sample {record_id})")
                plt.tight_layout()
                plt.savefig("pred_mask_sample.png") 
                return

if __name__ == "__main__":
    visualize_one_sample()