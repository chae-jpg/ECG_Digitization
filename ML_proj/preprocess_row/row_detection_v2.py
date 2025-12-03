import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from scipy.signal import find_peaks

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# 1. 이미지 로드
# image_path = "/Volumes/Untitled/ML/physionet-ecg-image-digitization/train/225208096/225208096-0010.png"
# # /Volumes/Untitled/ML/physionet-ecg-image-digitization/train/249637450/249637450-0009.png
# # /Volumes/Untitled/ML/physionet-ecg-image-digitization/train/225208096/225208096-0010.png


# 1. 자동 회전 함수
def auto_rotate(image):
    """
    이미지의 높이가 너비보다 크면(세로 사진) 시계 방향으로 90도 회전
    """
    h, w = image.shape[:2]
    if h > w:
        print("[Info] 세로 이미지 감지. 90도 회전합니다.")
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

# 2. 격자 제거 필터 
def apply_diagonal_filter(gray_img):
    """
    수평/수직 격자를 제거하고 심전도 파형의 사선 성분만 남김
    """
    gray_inv = cv2.bitwise_not(gray_img)
    th = cv2.adaptiveThreshold(gray_inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, 5)

    # 대각선 커널 생성
    kernel_size = 5
    diag_kernel1 = np.eye(kernel_size, dtype=np.uint8) # \ 
    diag_kernel2 = np.fliplr(diag_kernel1)             # / 

    # Morphology Open으로 사선만 추출
    diag1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, diag_kernel1)
    diag2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, diag_kernel2)
    
    
    combined_diag = cv2.add(diag1, diag2)
    return combined_diag

def detect_ecg_rows_robust_final(image_path):
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("이미지 로드 실패")
        return

    # 자동 회전 적용
    image = auto_rotate(original_image)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 격자 제거
    filtered_img = apply_diagonal_filter(gray)

    x_start = int(w * 0.35)
    x_end = int(w * 0.65)
    center_strip = filtered_img[:, x_start:x_end]

    # 가로 스미어링 
    smear_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    smeared_strip = cv2.morphologyEx(center_strip, cv2.MORPH_CLOSE, smear_kernel)

    # 수평 투영 계산
    projection = np.sum(smeared_strip, axis=1)
    mask_top = int(h * 0.05)
    mask_bottom = int(h * 0.02)
    projection[:mask_top] = 0
    projection[h-mask_bottom:] = 0
    projection_blur = cv2.GaussianBlur(projection.reshape(-1,1), (101, 1), 0).reshape(-1)

    # 피크 검출
    # distance: 행 간 최소 간격을 전체 높이의 15%
    min_dist = int(h * 0.15)
    # prominence: 주변보다 툭 튀어나온 정도가 최대값의 10% 이상인 것만
    peaks, properties = find_peaks(projection_blur, distance=min_dist, 
                                   prominence=np.max(projection_blur)*0.1)

    # 상위 4개 선택 
    if len(peaks) > 4:
        sorted_indices = np.argsort(properties['prominences'])[::-1]
        peaks = peaks[sorted_indices[:4]]
    
    peaks = np.sort(peaks) # Y좌표 순으로 정렬

    print(f"최종 검출된 행 위치 (Peaks): {peaks}")

    # =========================================================
    # 시각화
    # =========================================================
    plt.figure(figsize=(12, 10))

    # 1. 중앙 스트립 스미어링 결과 (투영 대상)
    plt.subplot(2, 2, 1)
    # 원본 크기의 빈 이미지에 중앙 스트립만 붙여서 보여줌
    vis_strip = np.zeros_like(filtered_img)
    vis_strip[:, x_start:x_end] = smeared_strip
    plt.imshow(vis_strip, cmap='gray')
    plt.title("Center Strip Smeared (For Projection)")
    plt.axis('off')

    # 2. 투영 그래프 및 피크
    plt.subplot(2, 2, 2)
    plt.plot(projection_blur)
    plt.plot(peaks, projection_blur[peaks], "xr", markersize=10, markeredgewidth=2)
    plt.title(f"Projection Profile (Peaks: {len(peaks)})")
    plt.xlabel("Row Index (Y)")
    plt.ylabel("Sum of Intensity")
    plt.grid(True)

    # 3. 최종 결과 
    vis_img = image.copy()
    for p in peaks:
        # 이미지 전체 가로를 가로지르는 선 그리기
        cv2.line(vis_img, (0, int(p)), (w, int(p)), (0, 0, 255), 3)
    
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Final Detection Result (on Auto-Rotated Image)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 실행 
detect_ecg_rows_robust_final("/Volumes/Untitled/ML/physionet-ecg-image-digitization/train/225208096/225208096-0010.png")