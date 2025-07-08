
#  Breast Cancer Classification from Mammograms using Deep Learning

This repository contains code and methodology for a medical imaging project focused on the **binary classification** of mammogram images into **Benign** and **Malignant** categories. The dataset used is the **CBIS-DDSM** dataset, which includes cropped mammogram images annotated for pathology. This project explores both a **Simple CNN** and **MobileNetV2 (Transfer Learning)** approach for the classification task.

---

##  Introduction

Breast cancer is one of the leading causes of death among women globally. Early detection and accurate classification between benign and malignant masses are crucial for effective treatment. This project leverages deep learning to automate and improve the diagnostic process using mammogram images.

---

##  Methodology

### 1. **Data Preprocessing**
- Loaded and verified image paths from the CBIS-DDSM dataset.
- Resized images to 224x224 and normalized pixel values.
- Labels were encoded as 0 (Benign) and 1 (Malignant).
- Applied data augmentation (rotation, zoom, flipping) for model generalization.

### 2. **Modeling**

####  Simple CNN
- Constructed a lightweight CNN with 2 Conv2D layers, max pooling, dropout, and dense layers.
- Served as a baseline for evaluating classification performance.

####  MobileNetV2
- Used a pre-trained MobileNetV2 model from ImageNet.
- Replaced the head with Global Average Pooling, Dropout, and Dense layers.
- Enabled class weighting and augmentation for improved learning.

---

##  Results & Discussion

| Model         | Test Accuracy | ROC AUC |
|---------------|---------------|---------|
| Simple CNN    | ~60%          | ~0.66   |
| MobileNetV2   | ~72%          | ~0.74   |

- MobileNetV2 outperformed the custom CNN in terms of both accuracy and generalization.
- ROC AUC showed improved separability between benign and malignant classes using transfer learning.
- Despite class imbalance, class weighting and augmentation helped improve recall for the minority class.

---

## Key Components

- `01_preprocessing.`: Data cleaning, resizing, label encoding
- `02_simple_cnn.`: Baseline CNN model training and evaluation
- `03_mobilenetv2.`: Transfer learning using MobileNetV2
- `04_evaluation.`: Classification report, ROC-AUC curve

---

##  How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/adeedamukhtar/Breast-Cancer-Mammogram/blob/main/cancer.ipynb

```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the CBIS-DDSM dataset from:
[Kaggle Link](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data)


---

##  Requirements

- Python 3.7+
- TensorFlow / Keras
- NumPy, pandas, matplotlib, scikit-learn
- OpenCV (for image loading)

Install all dependencies via:
```bash
pip install tensorflow opencv-python pandas numpy scikit-learn matplotlib tqdm
```

---

## References

- [CBIS-DDSM Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/cbis-ddsm-breast-cancer-image-dataset)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- TensorFlow Documentation

---

##  License

This project is open-source and available under the MIT License. You may reuse or modify it for educational purposes.
