import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset.dataset_v3 import RobustNormalizer

# --- [1] 사용자님이 쓰시던 전처리 클래스 그대로 가져옴 ---

# class RobustNormalizer:
#     def __init__(self, target_size=(1000, 1500)):
#         self.H, self.W = target_size

#     def order_points(self, pts):
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)] # TL
#         rect[2] = pts[np.argmax(s)] # BR
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)] # TR
#         rect[3] = pts[np.argmax(diff)] # BL
#         return rect

#     def four_point_transform(self, image, pts):
#         rect = self.order_points(pts)
#         dst = np.array([
#             [0, 0],
#             [self.W - 1, 0],
#             [self.W - 1, self.H - 1],
#             [0, self.H - 1]], dtype="float32")
#         M = cv2.getPerspectiveTransform(rect, dst)
#         return cv2.warpPerspective(image, M, (self.W, self.H))

#     def detect_paper_contour(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         h, w = gray.shape
#         total_area = w * h
        
#         # 전략 1: Adaptive Threshold
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edged = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                       cv2.THRESH_BINARY_INV, 11, 2)
        
#         cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
#         found = self._check_contours(cnts, total_area)
#         if found is not None: return found

#         # 전략 2: Canny Edge
#         edged = cv2.Canny(blurred, 50, 200)
#         edged = cv2.dilate(edged, None, iterations=1)
#         cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
#         return self._check_contours(cnts, total_area)

#     def _check_contours(self, contours, total_area):
#         for c in contours:
#             # 면적 기준 (20% 이상)
#             if cv2.contourArea(c) < (total_area * 0.2): 
#                 continue

#             # Convex Hull & Approx
#             hull = cv2.convexHull(c)
#             peri = cv2.arcLength(hull, True)
#             approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

#             if len(approx) == 4:
#                 return approx
#             if len(approx) > 4:
#                 rect = cv2.minAreaRect(c)
#                 box = cv2.boxPoints(rect)
#                 return np.int32(box)
#         return None

#     def process(self, image):
#         # 1. 가로/세로 비율 보정 (눕히기)
#         h, w = image.shape[:2]
#         if h > w:
#             image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

#         # 2. 종이 찾기
#         screenCnt = self.detect_paper_contour(image)

#         if screenCnt is not None:
#             # 찾았으면 펴기 (Warp)
#             warped = self.four_point_transform(image, screenCnt.reshape(4, 2))
#             return warped, "Aligned (Warp)"
#         else:
#             # 못 찾았으면 리사이즈 (Fallback)
#             resized = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
#             return resized, "Fallback (Resize)"

# --- [2] 테스트 설정 ---
CONFIG = {
    "MODEL_PATH": "./checkpoints/unet_ecg_epoch_15.pth", # 모델 경로 확인!
    "TEST_DIR": "./test_tmp",         # 테스트 이미지가 있는 폴더
    "SAVE_DIR": "./test_results_final", # 결과 저장할 폴더
    "IMG_SIZE": (512, 1024),          # 모델 입력 크기
    "CLASSES": 13,
    "DEVICE": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
}

def get_palette():
    # 1~12번 클래스 색상표
    palette = np.zeros((13, 3), dtype=np.uint8)
    colors = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255],
              [128,0,0], [0,128,0], [0,0,128], [128,128,0], [128,0,128], [0,128,128]]
    for i, c in enumerate(colors): palette[i+1] = c
    return palette

def run_test():
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
    
    # 1. 모델 로드
    print(f"Loading model... ({CONFIG['DEVICE']})")
    model = smp.Unet(encoder_name="efficientnet-b0", in_channels=1, classes=CONFIG["CLASSES"]).to(CONFIG['DEVICE'])
    try:
        model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
    except FileNotFoundError:
        print("❌ 모델 파일이 없습니다.")
        return
    model.eval()

    # 2. 전처리기 초기화
    normalizer = RobustNormalizer(target_size=(1000, 1500))
    palette = get_palette()
    
    # 3. 모델 입력 변환기
    transform = A.Compose([
        A.Resize(height=CONFIG['IMG_SIZE'][0], width=CONFIG['IMG_SIZE'][1]),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    img_files = []
    for root, dirs, files in os.walk(CONFIG["TEST_DIR"]):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith("._"):
                # root와 f를 합쳐서 '진짜 전체 경로'를 저장함
                full_path = os.path.join(root, f)
                img_files.append(full_path)
    
    print(f"총 {len(img_files)}장의 이미지 테스트 시작...")

    # --- [수정 2] 반복문 수정 ---
    for path in tqdm(img_files):
        # path는 이미 '전체 경로'이므로 os.path.join을 또 하면 안 됨!
        
        raw_img = cv2.imread(path)
        if raw_img is None: 
            # tqdm.write(f"로드 실패: {path}") # 디버깅용
            continue
        
        # [Step 1] 전처리
        aligned_img, status = normalizer.process(raw_img)
        
        # [Step 2] 추론
        gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        tens = transform(image=gray)['image'].unsqueeze(0).to(CONFIG['DEVICE']).float()
        
        with torch.no_grad():
            output = model(tens)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy().astype(np.uint8)

        # [Step 3] 시각화
        h, w = aligned_img.shape[:2]
        pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in range(1, 13): color_mask[pred_resized == cls] = palette[cls]
        
        overlay = cv2.addWeighted(aligned_img, 0.7, color_mask, 0.3, 0)
        
        disp_h = 500
        disp_w = int(w * (disp_h / h))
        res = np.hstack([
            cv2.resize(aligned_img, (disp_w, disp_h)), 
            cv2.resize(overlay, (disp_w, disp_h))
        ])
        
        # [수정 3] 저장 파일명 겹침 방지 (폴더명_파일명)
        # 하위 폴더에 같은 이름(0001.png)이 있을 수 있으므로 폴더명도 붙여줌
        folder_name = os.path.basename(os.path.dirname(path))
        file_name = os.path.basename(path)
        save_name = f"res_{folder_name}_{file_name}"
        
        cv2.imwrite(os.path.join(CONFIG["SAVE_DIR"], save_name), res)

    print(f"✅ 완료! {CONFIG['SAVE_DIR']} 폴더 확인하세요.")

if __name__ == "__main__":
    run_test()