# Digitization of ECG Images

## Description

This is a repository for ECG image digitization, which is conducted for the team project of **25-2 Machine Learning (02) course**.  
We used the dataset from [PhysioNet - Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization), and constructed three different pipelines to preprocess images and convert images into digital signals including:

  - Classical CV Preprocessing + DNN/CNN Regression
  - Mask-based U-Net Segmentation
  - Marker-based Rectification + Pixel Regression

A detailed explanation of the methodology, model architectures, and experimental results can be found in our Project Poster below.
![](https://github.com/chae-jpg/ECG_Digitization/blob/main/poster.png?raw=true)

## Members
|Eunseo Ko|Hyelee Lee|[Chaewon Lee](https://github.com/chae-jpg)|
|-|-|-|

## How To Run
### Pipeline 1
1. Open `1st_pipeline.ipynb`.
2. Execute the notebook.

## References
- https://moody-challenge.physionet.org/2024/
- https://www.kaggle.com/code/ambrosm/ecg-original-explained-baseline
- https://github.com/felixkrones/ECG-Digitiser
- https://github.com/viggi1000/Unet-ECG-Segmentation-Wavelet
