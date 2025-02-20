# 3D BiFormer-Net: A Hybrid Transformer-CNN Framework for Robust Medical Image Landmark Detection

## 1. Introduction
### 1.1 What for?

The project of **3D BiFormer-Net** aims to accurately localize anatomical landmarks in 3D medical images, which is a critical task for applications such as surgical navigation, disease diagnosis, and treatment planning.


### 1.2 HighLights
* **3D BiFormer-Net** is designed to meet the demands of clinical environments, providing robust and accurate landmark detection even in challenging scenarios such as low-resolution images, occlusions, or missing landmarks.
* We propose a hybrid framework that combines the strengths of **CNNs** and **Transformers**. CNNs are used for efficient local feature extraction, while the Transformer module, equipped with a **bi-level routing attention mechanism**, enhances global context modeling. This combination addresses the limitations of standalone architectures.
* Extensive experiments on the **[Head Landmark_24nc]** dataset demonstrate that **3D BiFormer-Net** significantly outperforms existing baselines across multiple metrics, particularly in scenarios involving complex anatomical structures and missing landmarks.


## 2. Preparation
### 2.1 Requirements
- python >=3.7
- pytorch >=1.10.0
- Cuda 10 or higher
- numpy
- pandas
- scipy
- nrrd
- time

### 2.2 Data Preparation
<!The dataset will be available soon!>
The dataset is available at https://drive.google.com/file/d/1NGsBbqXZLDlkiSJtDQdyMlXzgnkFoVON/view?usp=sharing>
* Data division
```
    - mmld_dataset/train     # 458 samples for training
    - mmld_dataset/val       # 100 samples for validation
    - mmld_dataset/test      # 100 samples for testing
```
* Data format
```
    - *_volume.nrrd     # 3D volumes
    - *_label.npy       # landmarks
    - *_spacing.npy     # CT spacings, used for calculating MRE
```

## 3. Train and Test
### 3.1 Training baseline heatmap regression model

```
python main_baseline.py --model_name BiFormer_Unet   # network training for baseline heatmap regression model using backbone BiFormer_Unet
```

### 3.2 Training with PBiFormer_Unet

```
python main_yolol.py --model_name PBiFormer_Unet              # network training using backbone PBiFormer_Unet
```

### 3.3 Fine-tuning in a pretrained checkpoint

```
python main_baseline.py --resume ../SavePath/baseline/model.ckpt

python main_yolol.py --resume ../SavePath/yolol/model.ckpt
```

### 3.4 Metric counting
```
python main_baseline.py --test_flag 0 --resume ../SavePath/baseline/model.ckpt  # calculate MRE and SDR in validation set
python main_baseline.py --test_flag 1 --resume ../SavePath/baseline/model.ckpt  # calculate MRE and SDR in test set

python main_yolol.py --test_flag 0 --resume ../SavePath/yolol/model.ckpt  # calculate MRE and SDR in validation set
python main_yolol.py --test_flag 1 --resume ../SavePath/yolol/model.ckpt  # calculate MRE and SDR in test set
```
