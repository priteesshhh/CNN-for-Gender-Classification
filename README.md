# CNN-for-Gender-Classification

## Overview
This assignment focuses on building and training a Convolutional Neural Network (CNN) using the ResNet18 architecture for gender classification. The dataset contains facial images of males and females, structured into separate folders. The core objective was to implement a complete deep learning pipeline using PyTorch to classify images into two gender categories:
- **Male** → `0`
- **Female** → `1`

## Dataset
- **URL**: [Kaggle - Adience Gender Dataset](https://www.kaggle.com/datasets/alfredhhw/adiencegender)
- Contains face images divided into `'m'` and `'f'` directories.

## Major Steps

### 1. Environment Setup & Libraries Installation
Installed the following Python libraries:
- `torch`
- `torchvision`
- `matplotlib`
- `opencv-python`
- `Pillow`

### 2. Dataset Preparation
- Mounted Google Drive to Google Colab for file access.
- Extracted the dataset archive to a working directory.
- Verified existence of gender-labeled folders: `'m'` and `'f'`.

### 3. Custom Dataset Class
- Created `GenderDataset` class extending `torch.utils.data.Dataset`.
- Mapped:
  - `'m'` → label `0`
  - `'f'` → label `1`
- Applied image transformations:
  - Resizing
  - Normalization
  - Augmentation using `torchvision.transforms`

### 4. Data Loading
- Split dataset:
  - 80% Training
  - 10% Validation
  - 10% Testing
- Used `torch.utils.data.DataLoader` for efficient batching and shuffling.

### 5. Model Definition
- Loaded a pre-trained **ResNet18** model from `torchvision.models`.
- Replaced the final fully connected layer to output **2 classes** for gender classification.

### 6. Training Loop
- Trained for **10 epochs**.
- Used:
  - **Loss Function**: `CrossEntropyLoss`
  - **Optimizer**: `Adam`
- Tracked:
  - Training loss
  - Training accuracy
  - Validation loss
  - Validation accuracy

### 7. Evaluation
- Evaluated the model on the **test dataset**.
- Achieved a **test accuracy of approximately 82.35%**.

### 8. Results & Observations
- Observed consistent accuracy improvements across epochs.
- Noted improved generalization due to **data augmentation**.
- Validation accuracy remained stable, suggesting **no significant overfitting**.

---

## Final Accuracy
**Test Accuracy:** _~82.35%_

## Tools Used
- Google Colab
- PyTorch
- torchvision
- OpenCV
- matplotlib
- Pillow

## Author
*Priteesh Madhav Reddy*  
*Date: April 2025*
