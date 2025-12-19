# 🌸 Flower Classification using VGG16 (Transfer Learning & Fine-Tuning)

A deep learning–based image classification project that classifies flower images into five categories using **VGG16 pretrained on ImageNet**. The project demonstrates **feature extraction**, **fine-tuning**, and **data augmentation** to achieve high accuracy on a real-world dataset.

---

## 📌 Project Overview

Image classification is a core problem in computer vision. Instead of training a CNN from scratch, this project leverages **transfer learning** using the VGG16 architecture to efficiently learn from a relatively small dataset.

The model is trained in two phases:

1. **Feature Extraction** – Freeze pretrained VGG16 layers and train a custom classifier
2. **Fine-Tuning** – Unfreeze top VGG16 layers and retrain with a low learning rate

---

## 📂 Dataset

* **Dataset**: `tf_flowers` (TensorFlow Datasets)
* **Total Images**: 3,670
* **Classes (5)**:

  * Daisy
  * Dandelion
  * Roses
  * Sunflowers
  * Tulips

The dataset is converted into a directory structure compatible with `ImageDataGenerator`.

---

## 🧠 Model Architecture

### 🔹 Base Model

* VGG16 (pretrained on ImageNet)
* `include_top = False`
* Input shape: `(224, 224, 3)`

### 🔹 Custom Classifier

* Flatten layer
* Dense (512 units, ReLU)
* Dropout (0.5)
* Dense (5 units, Softmax)

---

## 🏋️ Training Strategy

### 1️⃣ Feature Extraction

* All VGG16 layers frozen
* Optimizer: Adam
* Loss: Categorical Crossentropy

**Results:**

* Training Accuracy: **77.36%**
* Validation Accuracy: **80.03%**

---

### 2️⃣ Fine-Tuning

* Last 4 layers of VGG16 unfrozen
* Lower learning rate (1e-5)

**Results:**

* Training Accuracy: **90.83%**
* Validation Accuracy: **85.77%**

➡️ Fine-tuning significantly improved performance by adapting pretrained features to the flower dataset.

---

## 📊 Performance Comparison

| Phase              | Train Accuracy | Validation Accuracy |
| ------------------ | -------------- | ------------------- |
| Feature Extraction | 77.36%         | 80.03%              |
| Fine-Tuning        | 90.83%         | 85.77%              |

---

## 🔍 Data Augmentation

Applied using `ImageDataGenerator`:

* Rotation
* Width & height shift
* Zoom
* Horizontal flip
* Rescaling (1/255)

This helps reduce overfitting and improves generalization.

---

## 🧪 Inference Example

The trained model correctly predicts the flower class from unseen images after proper preprocessing (RGB conversion and normalization).

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* TensorFlow Datasets
* Google Colab (GPU)

## 🙋‍♀️Sonia Firdous
soniafirdous1985@gmail.com
