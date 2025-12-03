import cv2
import numpy as np
import os
from tqdm import tqdm

class RobustNormalizer:
    def __init__(self, target_size=(1000, 1500)):
        self.H, self.W = target_size

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        dst = np.array([
            [0, 0],
            [self.W - 1, 0],
            [self.W - 1, self.H - 1],
            [0, self.H - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (self.W, self.H))

    def detect_paper_contour(self, image):
        """
        종이 테두리를 찾는 강력한 함수 (2단계 전략)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_area = w * h
        
        # 전략 1: Adaptive Threshold 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        found_cnt = self._check_contours(cnts, total_area)
        if found_cnt is not None: return found_cnt

        edged = cv2.Canny(blurred, 50, 200)
        # Dilation
        edged = cv2.dilate(edged, None, iterations=1)
        
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        found_cnt = self._check_contours(cnts, total_area)
        return found_cnt

    def _check_contours(self, contours, total_area):
        for c in contours:
            area = cv2.contourArea(c)
            
            if area < (total_area * 0.2): 
                continue

            # Convex Hull 적용
            # 구겨지거나 접힌 종이의 외곽선을 매끄럽게 감쌈 -> 사각형 찾기 쉬워짐
            hull = cv2.convexHull(c)
            
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            # 꼭짓점이 4개면 
            if len(approx) == 4:
                return approx
                
            if len(approx) > 4:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                return box
                
        return None

    def process(self, img_path):
        if isinstance(img_path, str):
            image = cv2.imread(img_path)
        elif isinstance(img_path, np.ndarray):
            image = img_path
        else:
            return None, "Invalid Input Type"

        if image is None: return None, "Load Error"
        
        # 1. 가로/세로 비율 보정
        h, w = image.shape[:2]
        if h > w:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # 2. 종이 찾기
        screenCnt = self.detect_paper_contour(image)

        if screenCnt is not None:
            
            warped = self.four_point_transform(image, screenCnt.reshape(4, 2))
            return warped, "Warped"
        else:
            
            resized = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            return resized, "Fallback_Resize"

# --- 실행 코드 ---
if __name__ == "__main__":
    RAW_IMG_DIR = "/Volumes/Untitled/ML/physionet-ecg-image-digitization/kmeans_output_v3"
    ALIGNED_DIR = "./dataset/images_aligned"
    os.makedirs(ALIGNED_DIR, exist_ok=True)
    
    normalizer = RobustNormalizer(target_size=(1000, 1500))
    
    img_files = []
    for root, dirs, files in os.walk(RAW_IMG_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg')) and not f.startswith("._"):
                img_files.append(os.path.join(root, f))
    
    print(f"총 {len(img_files)}장 정규화 시작 (Robust Mode)...")
    
    cnt_warp = 0
    cnt_fallback = 0
    
    for path in tqdm(img_files):
        try:
            result, status = normalizer.process(path)
            
            if result is not None:
                filename = os.path.basename(path)
                cv2.imwrite(os.path.join(ALIGNED_DIR, filename), result)
                
                if "Warped" in status: cnt_warp += 1
                else: cnt_fallback += 1
        except Exception as e:
            print(f"Error {path}: {e}")
            
    print(f"\n[완료] Warped(사진 펴짐): {cnt_warp} / Fallback(원본 유지): {cnt_fallback}")