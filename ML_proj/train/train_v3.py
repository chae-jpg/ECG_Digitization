import os
import torch
from torch.utils.data import DataLoader, ConcatDataset 
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# 두 데이터셋 클래스 임포트
from dataset.synthetic_data import SyntheticECGDataset
from dataset.real_dataset import RealAlignedDataset 

CONFIG = {
    # 경로 설정 
    "CSV_DIR": "/Volumes/Untitled/ML/physionet-ecg-image-digitization/train", # 합성용
    "REAL_IMG_DIR": "./dataset/images_aligned",  
    "REAL_MASK_DIR": "./dataset/masks_v2",       
    
    "IMG_SIZE": (512, 1024),
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 25,
    "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "CLASSES": 13
}

def train_model():
    os.makedirs("./checkpoints", exist_ok=True)
    print(f"Using device: {CONFIG['DEVICE']}")

    # --- [1] 데이터셋 준비 ---
    
    # 1. 합성 데이터셋 (부족한 패턴 보완용)
    
    ds_synthetic = SyntheticECGDataset(
        csv_dir=CONFIG["CSV_DIR"], 
        img_size=CONFIG["IMG_SIZE"], 
        is_train=True,
        expand_factor=5  
    )
    
    # 2. 리얼 데이터셋 (메인 학습용)
    ds_real = RealAlignedDataset(
        img_dir=CONFIG["REAL_IMG_DIR"],
        mask_dir=CONFIG["REAL_MASK_DIR"],
        img_size=CONFIG["IMG_SIZE"],
        is_train=True
    )
    
    # 3. 두 데이터셋 합치기 (Concat)
    full_dataset = ConcatDataset([ds_synthetic, ds_real])
    
    print(f"Synthetic Size: {len(ds_synthetic)}")
    print(f"Real Size: {len(ds_real)}")
    print(f"Total Training Data: {len(full_dataset)}")

    train_loader = DataLoader(
        full_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True, # 섞어서 학습 
        num_workers=4,
        pin_memory=False
    )

    # --- [2] 모델 정의 ---
    model = smp.Unet(
        encoder_name="efficientnet-b0", 
        encoder_weights="imagenet", 
        in_channels=1, 
        classes=CONFIG["CLASSES"],
    ).to(CONFIG['DEVICE'])

    resume_path = "./checkpoints/unet_ecg_epoch_15.pth" 
    
    if os.path.exists(resume_path):
        print(f"기존 학습된 모델을 발견했습니다! ({resume_path})")
        print(">> 가중치를 로드하고 이어서 학습합니다...")
        model.load_state_dict(torch.load(resume_path, map_location=CONFIG['DEVICE']))
    else:
        print("기존 모델이 없습니다. 처음부터 학습을 시작합니다.")

    # DiceLoss: 픽셀 하나하나가 아니라 '영역이 얼마나 겹치는지'를 봄 (선이 얇을 때 필수)
    dice_loss = DiceLoss(mode="multiclass", from_logits=True)
    
    # FocalLoss: 모델이 틀리는 '어려운 예제(선)'에 가중치를 더 줌
    focal_loss = FocalLoss(mode="multiclass")

    # 두 손실함수를 합쳐서 사용 
    def criterion(preds, targets):
        return dice_loss(preds, targets) + focal_loss(preds, targets)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'])

    # --- [3] 학습 루프 ---
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        
        for images, masks in loop:
            images = images.to(CONFIG['DEVICE'])
            masks = masks.to(CONFIG['DEVICE'])
            
            outputs = model(images)
            loss = criterion(outputs.cpu(), masks.cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(train_loader):.4f}")
        
        # 모델 저장
        torch.save(model.state_dict(), f"./checkpoints/unet_ecg_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_model()