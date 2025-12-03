import cv2
import numpy as np
import cv2
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm 
import joblib 

DATA_DIR = "/Volumes/Untitled/ML/physionet-ecg-image-digitization"
OUTPUT_DIR = "./kmeans_output_v3"
CACHE_FILE = "./ecg_images_cache.pkl"  # ⚡ 처리된 데이터를 저장할 파일명

def get_images_cached(folder):
    # 1. 캐시 파일 확인 및 로드
    if os.path.exists(CACHE_FILE):
        print(f"⚡ 캐시 파일 발견! ({CACHE_FILE})")
        try:
            # joblib으로 로드 (메모리 맵핑을 사용하여 훨씬 효율적)
            data = joblib.load(CACHE_FILE)
            return data['images'], data['paths']
        except Exception as e:
            print(f"캐시 로드 실패: {e}")

    # 2. 원본 로드 (Load 함수는 그대로 사용)
    print("🐢 원본 이미지 로드 시작...")
    images, paths = load_images_fast(folder) 
    
    # 3. joblib으로 저장 (압축 옵션 3 정도 주면 용량도 줄고 메모리도 덜 씀)
    print(f"💾 데이터를 {CACHE_FILE}에 저장 중...")
    try:
        joblib.dump({'images': images, 'paths': paths}, CACHE_FILE, compress=3)
        print("✅ 저장 완료!")
    except Exception as e:
        print(f"⚠️ 저장 중 메모리 부족 발생 가능성: {e}")
        print("팁: process_one_image 함수에서 이미지 크기(resize)를 줄여보세요.")
    
    return images, paths

def get_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.startswith("._"): continue
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, f))
    return image_paths

# 2. 개별 이미지를 읽는 함수 (병렬 처리를 위해 분리)
def process_one_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    # 리사이즈 옵션도 고려했으나, 리사이즈 시 격자가 같이 줄어들면서 파형과의 분리가 어려워짐
    
    return img

# 3. 메인 실행 함수
def load_images_fast(folder):
    print("파일 목록 스캔 중...")
    paths = get_image_paths(folder)
    print(f"총 {len(paths)}개의 이미지를 찾았습니다. 병렬 로딩 시작...")

    imgs = []
    valid_paths = []

    # 쓰레드 8개를 사용하여 병렬로 읽기 - 그냥 이미지 빨리 읽어오려고
    with ThreadPoolExecutor(max_workers=8) as executor:
        
        results = list(tqdm(executor.map(process_one_image, paths), total=len(paths)))

    # 결과 필터링 (None 제외)
    for i, result in enumerate(results):
        if result is not None:
            imgs.append(result)
            valid_paths.append(paths[i])

    print(f"======= 이미지 추합 완료: 총 {len(imgs)}장 =======")
    return imgs, valid_paths

# =========================================================
# 1) Grid Removal - Morphology Black-Hat 
# =========================================================
def remove_grid(gray):
    """
    기존 방식(Grid 추출 후 Subtract)이 흰 배경에서 0을 만드는 문제를 해결하기 위해
    Black-Hat 연산으로 변경했습니다.
    이 함수는 이제 '격자가 제거되고 파형이 강조된 이미지(검은 배경)'를 반환합니다.
    """
    # 커널 크기: 파형 두께보다 크고, 굵은 그림자보다는 작게 (15~25 사이 추천)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    
    # Black-Hat: 밝은 배경 날리고 어두운 객체(파형, 글자)만 추출
    # 결과는 검은 배경에 흰색 파형이 됨
    waveform_extracted = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    return waveform_extracted


# --------------------------------------
# 2) EDGE FILTERING (Canny + Sobel)
# --------------------------------------
def extract_edge_mask(gray):
    """
    강한 edge (파형)만 남기기 위한 에지 필터링
    """

    # Sobel (세밀한 파형 edge 강화)
    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    sobel = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))

    # Canny
    canny = cv2.Canny(gray, 40, 120)

    # 두 개 OR 해서 edge map 강화
    edge_mask = cv2.bitwise_or(sobel, canny)

    # 이진화
    _, edge_mask = cv2.threshold(edge_mask, 30, 255, cv2.THRESH_BINARY)

    return edge_mask


# --------------------------------------
# 3) ADAPTIVE THRESHOLDING (K-means 대체 옵션)
# --------------------------------------
def adaptive_thresholding(gray):
    """
    grid 제거 후 adaptive threshold 적용
    """

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 2
    )
    return th

# =========================================================
# 3-2) K-means (grid 제거 후 한 번 더 필터링)
# =========================================================
def kmeans_waveform(enhanced, k=2):
    """
    grid 제거 + 대비 강화된 grayscale을 K-means로 파형만 분리 - 사실상 이진 분류긴 함
    """
    h, w = enhanced.shape
    pixels = enhanced.reshape(-1, 1)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # 가장 어두운 cluster = ECG 파형
    cluster_means = [enhanced.reshape(-1)[labels == i].mean() for i in range(k)]
    waveform_cluster = np.argmin(cluster_means)

    mask = (labels == waveform_cluster).reshape(h, w).astype(np.uint8) * 255
    return mask


# --------------------------------------
# 4) EDGE GUIDED K-MEANS
# --------------------------------------
def kmeans_edge_guided(original_img, gray, edge_mask, k=2):

    # edge가 있는 곳만 K-means 대상으로
    edge_pixels = np.where(edge_mask > 0)

    if len(edge_pixels[0]) == 0:
        print("edge가 너무 적어서 K-means 생략")
        return np.zeros_like(gray)

    # K-means input (L,a,b 중 L만 써도 됨)
    pixels = gray[edge_pixels].reshape(-1, 1)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)

    # 가장 어두운 cluster = 파형
    cluster_means = []
    for i in range(k):
        cluster_means.append(np.mean(pixels[labels == i]))

    waveform_cluster = np.argmin(cluster_means)

    # 전체 마스크로 확장
    mask = np.zeros_like(gray)
    mask[edge_pixels] = (labels == waveform_cluster).astype(np.uint8) * 255

    return mask


# =========================================================
# Combined Pipeline 
# =========================================================
def segment_ecg_pipeline(image_bgr, use_kmeans=False):
    """
    기존 파이프라인의 반환값 개수(mask, no_grid, enhanced)를 유지합니다.
    """
    # 1. Grayscale
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    # 2. Grid 제거 및 파형 추출 (Black-Hat 사용)
    # 여기서 no_grid는 사실상 '파형만 남은 신호(검은 배경)'가 됩니다.
    no_grid = remove_grid(gray)

    # 3. Contrast 강화
    # Black-Hat 결과는 이미 대비가 극명하므로 Normalize만 해도 충분합니다.
    enhanced = cv2.normalize(no_grid, None, 0, 255, cv2.NORM_MINMAX)

    # 4. Waveform segmentation (Thresholding)
    # 이미 배경이 검고 파형이 밝으므로 복잡한 K-means 없이 Threshold만으로도 잘 따집니다.
    # K-means 옵션을 켰을 때도 작동하도록 분기 처리 유지
    if use_kmeans:
        # K-means를 쓴다면 0(배경)과 0이 아닌 값(파형)을 구분
        mask = kmeans_waveform(enhanced, k=2)
    else:
        # 단순 Threshold (값이 30 이상이면 파형으로 간주)
        _, mask = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)

    # 5. Morphology로 끊어진 선 연결 및 잡음 제거
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 6. 작은 노이즈(글자 등) 제거 (선택 사항: 필요 없으면 주석 처리)
    # 파형보다 작은 점들을 지웁니다.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # 기존 코드와의 호환성을 위해 3개 변수 리턴
    return mask, no_grid, enhanced

def main():
    images, paths = get_images_cached(DATA_DIR)
    print(f"Loaded {len(images)} images.")

    if images is None:
        return
    
    # 저장할 폴더가 없으면 만듦 
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for img, path in tqdm(zip(images, paths), total=len(images), desc="ECG Processing"):
        
        # 1) 파이프라인 실행
        mask_k, _, _ = segment_ecg_pipeline(img, use_kmeans=True)
        
        # 2) 원본 파일명에서 ID 추출
        filename_full = os.path.basename(path)      # "1234.jpg"
        file_id = os.path.splitext(filename_full)[0] # "1234"
        
        # 3) 저장 
        save_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")
        cv2.imwrite(save_path, mask_k)

    print("전체 저장 완료!")

if __name__ == "__main__":
    main()
