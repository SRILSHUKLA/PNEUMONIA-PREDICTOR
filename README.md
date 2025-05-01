# Hybrid CNN-Transformer for Multi-Class Classification of Pneumonia, COVID-19, and Normal Chest X-rays

![custom-cnn-backbone](https://github.com/user-attachments/assets/237a4ddf-9835-45fa-9745-55fcac63f9f8)


## 📌 Overview
This repository contains the implementation of a hybrid CNN-Transformer architecture for classifying chest X-rays into three categories: **Normal**, **Pneumonia**, and **COVID-19**. The model combines convolutional neural networks' local feature extraction capabilities with transformers' global context modeling, achieving **96.33% F1-score** on the test set.



## 📂 Dataset
We use the [COVID-19-Pneumonia-Normal Chest X-ray Dataset](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset) by Asraf (2020), containing 6,871 expert-annotated chest X-rays

## 🛠️ Model Architecture
### Hybrid CNN-Transformer Design
1. **CNN Backbone**: Extracts local spatial features (Supports ResNet50, ConvNeXt, EfficientNetB5, or custom CNN)
2. **Feature Projection**: Maps CNN outputs to 512-D space
3. **Transformer Head**:
   - 2 Transformer encoder layers (8 attention heads)
   - 2048-dim feedforward network
   - GELU activation with dropout (p=0.1)
4. **Classifier**: 2-layer MLP with ReLU → 3-class output

## 🏆 Key Results
| Model Backbone | Validation F1 | Validation Accuracy | Training Time |
|----------------|--------------|---------------------|---------------|
| ResNet50       | 96.19%       | 96.15%              | 61 min        |
| ConvNeXt-base  | **96.33%**   | **96.29%**          | 541 min       |
| EfficientNetB5 | 96.05%       | 96.00%              | 94.8 min      |
