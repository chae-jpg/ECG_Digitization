import numpy as np
from scipy.signal import correlate, correlation_lags, resample

def compute_snr(true_signal, pred_signal, fs=500, max_shift_ms=100):
    """
    2025 PhysioNet Challenge 호환 SNR 계산
    포함 기능:
    1) Length match
    2) Time shift alignment (±100 ms)
    3) Baseline shift compensation (mean alignment)
    4) Amplitude scaling correction (L2 normalization)
    5) Final SNR computation
    """

    # -------------------------------
    # 1. 길이 맞추기
    # -------------------------------
    N = len(true_signal)
    if len(pred_signal) != N:
        pred_signal = resample(pred_signal, N)

    # -------------------------------
    # 2. Time shift alignment: cross-correlation
    # -------------------------------
    max_shift_samples = int((max_shift_ms / 1000) * fs)

    corr = correlate(true_signal, pred_signal, mode='full')
    lags = correlation_lags(N, N, mode='full')

    # 전체 lag 중 ±max_shift 범위로 제한
    valid = np.where(np.abs(lags) <= max_shift_samples)[0]
    lag_opt = lags[ valid ][ np.argmax(corr[ valid ]) ]

    # shift 적용
    if lag_opt > 0:
        pred_aligned = np.pad(pred_signal, (lag_opt, 0), 'constant')[:N]
    else:
        pred_aligned = np.pad(pred_signal, (0, -lag_opt), 'constant')[-N:]

    # -------------------------------
    # 3. Baseline shift removal (vertical mean alignment)
    # -------------------------------
    true_c = true_signal - np.mean(true_signal)
    pred_c = pred_aligned - np.mean(pred_aligned)

    # -------------------------------
    # 4. Scaling correction 
    # peak 스케일 또는 L2 노말라이즈
    # -------------------------------
    
    if np.linalg.norm(pred_c) == 0:
        return -50  # 노의미

    scale = np.linalg.norm(true_c) / np.linalg.norm(pred_c)
    pred_scaled = pred_c * scale

    # -------------------------------
    # 5. Final SNR
    # -------------------------------
    noise = true_c - pred_scaled

    signal_power = np.sum(true_c ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        return 50.0  # perfect reconstruction

    snr = 10 * np.log10(signal_power / noise_power)
    return snr
