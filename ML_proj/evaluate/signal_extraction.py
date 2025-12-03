import numpy as np
import cv2
from scipy.signal import resample

class SignalExtractor:
    def __init__(self, img_shape=(512,1024), total_len=5000):
        self.H, self.W = img_shape
        self.target_len = total_len

        # ECG 컴포넌트 class 범위 (wave pixel)
        # segmentation mask에서 wave pixel은 (class_id == lead_idx)
        # background는 0
        self.wave_value = lambda m, lead: (m == lead).astype(np.uint8)

        # synthetic dataset의 스케일
        ORIGIN_H = 1000
        scale_factor = img_shape[0] / ORIGIN_H
        ORIGIN_SCALE_Y = 150
        self.pixel_per_mv = ORIGIN_SCALE_Y * scale_factor

        # row 별 baseline
        grid_h = int(img_shape[0] * 0.75)
        cell_h = grid_h // 3

        self.baselines = {
            0: int(cell_h * 0.9),
            1: int(cell_h * 1.9),
            2: int(cell_h * 2.9)
        }

        self.id_to_lead = {
            1:'I', 2:'II', 3:'III', 4:'aVR',5:'aVL',6:'aVF',
            7:'V1',8:'V2',9:'V3', 10:'V4',11:'V5',12:'V6'
        }
            
        self.lead_to_row = {
            'I':0,'aVR':0,'V1':0,'V4':0,
            'II':1,'aVL':1,'V2':1,'V5':1,
            'III':2,'aVF':2,'V3':2,'V6':2
        }

    def extract_signal(self, mask):
        signals={}

        for class_id in range(1,13):
            lead = self.id_to_lead[class_id]
            row = self.lead_to_row[lead]
            baseline = self.baselines[row]

            # wave pixel mask
            wave = (mask == class_id).astype(np.uint8)

            # x-axis shape
            xs = np.arange(self.W)
            ys = np.zeros(self.W)

            # =====================
            # 핵심 개선 1: "top-most contour pixel" 찾기
            # =====================
            for x in range(self.W):
                col = wave[:,x]
                ys_in_col = np.where(col > 0)[0]

                if len(ys_in_col) == 0:
                    ys[x] = baseline
                else:
                    # contour = 가장 위(파형) 픽셀
                    ys[x] = np.min(ys_in_col)

            # =====================
            # 핵심 개선 2: 전압 변환
            # =====================
            volt = (baseline - ys) / self.pixel_per_mv

            # =====================
            # 핵심 개선 3: 보간 및 smoothing
            # =====================
            # missing 값(=baseline)은 양옆 interpolation
            filled = np.interp(xs, xs[volt!=0], volt[volt!=0], left=0, right=0)

            # smoothing (optional)
            filled = cv2.GaussianBlur(filled.reshape(-1,1),(15,1),1).flatten()

            # =====================
            # 핵심 개선 4: 5000 길이로 resample
            # =====================
            final = resample(filled, self.target_len)

            # baseline shift 제거
            final = final - np.mean(final)

            signals[lead] = final

        return signals
