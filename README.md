# MRI Sequence Classification Using Custom CNN with 5-Fold Cross Validation

## 1. Introduction and Motivation

Magnetic Resonance Imaging (MRI) provides multiple sequence types (e.g., T1, T1c+, T2), each highlighting distinct anatomical and pathological features. In neuroimaging, selecting the optimal MRI sequence for downstream tasks such as tumor segmentation or tissue delineation is critical. This project addresses the challenge of **automated MRI sequence classification**, aiming to support intelligent sequence selection.

### Research Objectives

* Develop a robust classifier to distinguish between T1, T1c+ (contrast-enhanced), and T2 MRI sequences.
* Evaluate the generalization of the model using 5-fold cross-validation.
* Leverage a lightweight, custom convolutional neural network (CNN) suitable for deployment.

---

## 2. Dataset and Preprocessing

### Dataset Overview

* Source: [Kaggle - Brain Tumor MRI Images](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)
* Modalities: T1, T1c+, T2
* Total Samples: 4478 images

  * T1: 15 folders
  * T1c+: 14 folders
  * T2: 15 folders

### Preprocessing Steps

* Image resizing to **(200 × 200)**
* Grayscale conversion (1 channel)
* Normalization to \[0, 1]
* Label mapping:

  * `0`: T1
  * `1`: T1c+
  * `2`: T2
* Dataset shuffled and split using **KFold (5 splits)**
* `.webp` image removed to ensure format consistency

### Input Pipeline

Implemented using `tf.data.Dataset` for efficient prefetching and batching. Each sample is:

* Read using `tf.io.read_file()` and decoded as JPEG
* Resized and cast to float32
* Batched with `BATCH = 128`

---

## 3. Model Architecture

### Custom CNN Design

A compact, efficient CNN was designed with modular `ConvBlock` layers:

```python
Input(200x200x1)
→ ConvBlock(32) → ConvBlock(64) → ConvBlock(128) → ConvBlock(256)
→ GlobalAveragePooling2D → Dense(128) → Dense(32) → Softmax(3 classes)
```

### ConvBlock Configuration

Each block includes:

* Conv2D with ReLU activation
* BatchNormalization (optional)
* MaxPooling2D

### Model Summary

* Parameters: 211,683
* Activation: ReLU
* Output: 3-class softmax
* Loss: Sparse Categorical Crossentropy
* Optimizer: Adam

### Visualization

Model architecture rendered using `visualkeras.layered_view()` for interpretability.

---

## 4. Experimental Setup

* **Cross-Validation**: 5-fold (Shuffle=True, Seed=5)
* **Epochs**: 40
* **Batch Size**: 128
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score (all weighted)

For each fold:

* Data is split into training and testing sets
* Model is trained and evaluated on each fold
* Predictions collected to compute metrics

---

## 5. Results

### Training Curves (Per Fold)

![__results___15_1](https://github.com/user-attachments/assets/1b150f54-5796-4402-b81e-3372b38d8bd9)

### Test Metrics

| Fold   | Accuracy (%) | Precision | Recall | F1 Score | Loss   |
| ------ | ------------ | --------- | ------ | -------- | ------ |
| Fold 1 | 74.22        | 0.8265    | 0.7422 | 0.7406   | 1.2140 |
| Fold 2 | 92.86        | 0.9367    | 0.9286 | 0.9273   | 0.2449 |
| Fold 3 | 99.22        | 0.9923    | 0.9922 | 0.9922   | 0.0159 |
| Fold 4 | 99.11        | 0.9912    | 0.9911 | 0.9910   | 0.0294 |
| Fold 5 | 99.22        | 0.9922    | 0.9922 | 0.9922   | 0.0235 |

### Observations

* Model generalizes extremely well on most folds (Folds 3–5)
* Fold 1 underperforms, possibly due to image quality/class imbalance
* Extremely low loss in high-performing folds indicates excellent convergence

---

## 6. Conclusion

This project demonstrates the efficacy of a custom-designed CNN for classifying MRI sequence types. Despite the simplicity of the architecture, it performs exceptionally well across most folds, with peak F1 scores above 99%.

### Key Contributions

* Efficient preprocessing pipeline using `tf.data`
* Lightweight and interpretable CNN model
* Strong generalization validated through cross-validation

### Future Directions

* Expand to 3D MRI classification (e.g., volume-based sequence identification)
* Augmentation with synthetic data to balance folds
* Deploy as a part of a preprocessing module in medical image analysis pipelines

---
