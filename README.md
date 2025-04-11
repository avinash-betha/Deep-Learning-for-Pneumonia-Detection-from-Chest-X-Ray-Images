# Deep Learning for Pneumonia Detection from Chest X-Ray Images

## Overview
This project presents a deep learning-based pipeline for automated pneumonia detection using chest X-ray images. Leveraging transfer learning with a fine-tuned **ResNet50** architecture, the model classifies images as either Normal or Pneumonia. The pipeline integrates robust data augmentation, class imbalance handling, hyperparameter optimization using **Keras Tuner**, and interpretable results via **Grad-CAM** visualizations.

## Problem Statement
Pneumonia is a life-threatening lung infection, especially dangerous for children and the elderly. Traditional diagnosis via X-ray interpretation can be time-consuming and prone to subjectivity. This project builds a scalable and explainable deep learning model that supports radiologists by automating the classification of chest X-rays.

## Dataset
- **Source**: [Chest X-Ray Pneumonia Dataset – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**:
  - `train/`: 5,216 images
  - `val/`: 16 images
  - `test/`: 624 images
- **Classes**: Binary classification (Normal, Pneumonia)
- **Challenge**: Severe class imbalance handled using weighted loss

## Tools & Technologies
- Python, TensorFlow, Keras
- Keras Tuner (Random Search)
- ResNet50 (transfer learning)
- Grad-CAM for model explainability
- Matplotlib, Scikit-learn

## Methodology

### 1. Data Preprocessing
- Rescaling, normalization, and image augmentation (rotation, zoom, shifts, flips)
- Addressed class imbalance using `class_weight` during training

### 2. Model Architecture
- Pre-trained **ResNet50** base (ImageNet weights, frozen layers)
- Custom top layers:
  - GlobalAveragePooling
  - Dense layer (ReLU + L2 regularization)
  - Dropout
  - Final Sigmoid output for binary classification

### 3. Hyperparameter Tuning
- Used **Keras Tuner** (Random Search) to optimize:
  - Dense layer size
  - Dropout rate
  - Learning rate
- Selected model based on best validation accuracy

### 4. Model Training & Evaluation
- Trained for 10 epochs using class-weighted binary cross-entropy
- Evaluated using:
  - Accuracy: **68%**
  - ROC-AUC: **0.83**
  - F1 Score, Precision, Recall
  - Confusion Matrix

### 5. Explainability
- Applied **Grad-CAM** on selected test images to visualize decision-making regions
- Helped interpret how the model detected pneumonia areas in X-rays

## Results

| Metric         | Value  |
|----------------|--------|
| Accuracy       | 68%    |
| ROC-AUC        | 0.83   |
| Precision (Pneumonia) | 0.93 |
| Recall (Pneumonia)    | 0.53 |
| F1 Score       | 0.67   |

- High recall for Normal cases (0.94)
- Grad-CAM heatmaps provided valuable insights for clinical interpretability

## Key Takeaways
- Transfer learning drastically improved convergence on a small dataset
- Hyperparameter tuning and weighted loss improved model performance
- Grad-CAM visualizations enhanced transparency of predictions

## Future Enhancements
- Explore ensemble models (e.g., InceptionV3, EfficientNet)
- Improve recall for Pneumonia with more data and class balancing
- Deploy on-device or as a web API for real-world clinical integration

## Author
**Avinash Betha**  
MS in Data Science – DePaul University  
Email: abetha@depaul.edu  
GitHub: [avinash-betha](https://github.com/avinash-betha)  
LinkedIn: [betha-avinash](https://linkedin.com/in/betha-avinash)
