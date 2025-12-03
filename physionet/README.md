# ecg-image-digitizer
End-to-end ECG image digitization system: homography normalization, grid rectification, pixel-wise waveform extraction, and 12-lead reconstruction with evaluation metrics (SNR/NRMSE/Correlation).


---

## Dataset

Please download the original PhysioNet ECG Image Digitization dataset directly from Kaggle:

https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data

## Code

My code is contained in the `physionet/archive/` directory of this repository.  
Once you set the correct paths in the notebook, you can run the code and the folders and result files shown here will be created automatically.

---

Modern ECG software requires **digital time-series**, while hospitals often store ECGs only as **printed sheets or scanned images**.   

To address this issue, this project provide a complete workflow for converting **ECG images** (scans, mobile photographs, printed ECG sheets) into **digital 12-lead time-series signals (mV)**.   

Follows the core goals of the PhysioNet ECG Digitization Challenge, but is implemented independently with customized preprocessing, reconstruction, and evaluation modules.   

The codebase includes:

* **Image preprocessing pipeline**
  (orientation correction, geometry normalization, grid alignment, artifact handling)
* **Waveform extraction system**
  (pixel-wise waveform tracing, multi-strip reconstruction, 12-lead assembly)
* **Ground-truth alignment utilities**
  (segment mapping for I/II/III, augmented leads, and V1–V6)
* **Quality evaluation metrics**
  (SNR, NRMSE, correlation coefficient)
* **Automatic figure generation**
  (ECG overlays, grid visualizations, waveform comparison plots)
* **Dataset generation utilities**
  (rectified images, pixel-level labels for model compression or knowledge distillation)
* **Batch evaluation and error-case debugging tools**

---

## **Repository Components**

### **1. ECG Image → Digital Time-Series Conversion**

End-to-end pipeline that converts a single ECG image into:

* **12 reconstructed leads** (I, II, III, aVR, aVL, aVF, V1–V6)
* Output format: NumPy arrays representing millivolt (mV) waveforms
* Lead II long strip (10 seconds), and 2.5-second segments for all other leads

Supports all PhysioNet image types (0001, 0003, 0004, …, 0012).

<img width="1802" height="889" alt="0003-0012" src="https://github.com/user-attachments/assets/85582968-3ea7-46f3-aad8-ae1f37fbe421" />

---

### **2. Ground-Truth Mapping & Alignment**

Because PhysioNet CSV files do **not store all 12 leads consecutively**, this repo includes a precise segment-mapping system:

| Lead                 | GT Position in CSV | Duration |
| -------------------- | ------------------ | -------- |
| I, II, III           | First 625 samples  | 2.5 s    |
| aVR, aVL, aVF        | Samples 625–1250   | 2.5 s    |
| V1, V2, V3           | Samples 1250–1875  | 2.5 s    |
| V4, V5, V6           | Samples 1875–2500  | 2.5 s    |
| Lead II (long strip) | Full 0–2500        | 10 s     |

The system ensures accurate, lead-specific comparison between prediction and ground truth.

---

### **3. Evaluation Metrics (SNR / NRMSE / r)**

For each lead, three widely used waveform-quality metrics are computed:

* **Signal-to-Noise Ratio (SNR, dB)**
  Measures reconstruction fidelity relative to noise power
* **Normalized RMSE (NRMSE, %)**
  Relative shape error normalized by signal amplitude range
* **Pearson Correlation Coefficient (r)**
  Global shape agreement between predicted and true waveform

These metrics are combined in a unified figure title such as:

```
Lead V2  (SNR = 12.8 dB,  NRMSE = 2.87%,  r = 0.974)
```

---

### **4. Automatic Figure Generation**

For any image ID–type pair (e.g., 18736–0009), the following files are produced:

```
figures/{id}-{type}/
    fig_stage0_marker_overlay.png
    fig_stage0_normalised.png
    fig_stage1_gh.png
    fig_stage1_gv.png
    fig_stage1_mapping.png
    fig_stage1_rectified.png
    fig_stage2_pixel_overlay.png
    fig_stage2_series.png
```

These include:

* Input image with lead markers
* Normalized image
* Grid heatmaps
* Grid reconstruction visualization
* Rectified ECG image
* Pixel-level waveform overlay
* Final **12-lead GT vs Pred plot** with SNR/NRMSE/correlation

<img width="1877" height="637" alt="flow0" src="https://github.com/user-attachments/assets/db25ec2b-00ac-48dd-a02e-33919f03c1e6" />
<img width="1903" height="677" alt="flow1" src="https://github.com/user-attachments/assets/5ff5c10f-a2fc-4977-a5b0-ace48f00ab6f" />
<img width="1851" height="677" alt="flow2" src="https://github.com/user-attachments/assets/36d54489-1b33-45ca-9eba-e9ca11711c98" />

<img width="3600" height="3000" alt="fig_stage2_series" src="https://github.com/user-attachments/assets/15cb7167-a663-4f61-ae7a-df5f710bc51e" />

---

### **5. Batch Evaluation Tools**

Includes utilities to:

* Evaluate **hundreds of ECG images** automatically
* Export **Lead II SNR distribution** across the dataset
* Identify **worst-case samples** (e.g., SNR < 10 dB)
* Automatically run full visualization for those difficult cases

This enables systematic performance analysis across different image conditions (scanned, printed, photos, degraded, mold-damaged, etc.).

---

### **6. Dataset Generation for Model Compression / KD**

Tools for constructing datasets used for:

* **knowledge distillation** (teacher Stage-2 → student lightweight U-Net)
* rectified image storage
* pixel-level supervision

Outputs:

```
rectified/{id}/{id}-{type}.rect.png      # geometrically corrected image
rectified_labels/{id}/{pixel maps}.npy  # optional soft-label maps
```

---

## **Credits & Data Source**

This project uses components aligned with the PhysioNet 2024/2025 ECG Digitization Challenge:

* ECG-Image-Kit (synthetic image generation)
* ECG-Image-Database (real-world imaging artifacts)
* Ground-truth time-series CSVs from PhysioNet competition data

---
