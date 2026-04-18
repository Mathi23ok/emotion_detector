# Real-Time Emotion Recognition System

A computer vision system that detects human emotions from facial expressions in real time using a CNN model and webcam input.

---

## Overview

This project implements an end-to-end pipeline for emotion recognition:

* Train a CNN model on facial expression data
* Detect faces in real-time using OpenCV
* Predict emotions live from webcam feed
* Stabilize predictions using temporal smoothing

---

## System Architecture

```text
Image (Webcam)
   ↓
Face Detection (OpenCV Haar Cascade)
   ↓
Preprocessing (resize + normalization)
   ↓
CNN Model (TensorFlow)
   ↓
Prediction Probabilities
   ↓
Temporal Smoothing (deque)
   ↓
Final Emotion Output
```

---

## Features

* Real-time emotion detection using webcam
* CNN-based classification model
* Temporal smoothing to reduce prediction noise
* Confidence-based filtering for stable output
* End-to-end training + deployment pipeline

---

## Model Architecture

* Conv2D (32 filters) → MaxPooling
* Conv2D (64 filters) → MaxPooling
* Dense (128) + Dropout
* Output layer (Softmax)

---

## Training

* Dataset loaded using `image_dataset_from_directory`
* Images normalized to [0, 1]
* Early stopping to prevent overfitting
* Model checkpointing for best weights

---

## Tech Stack

* TensorFlow / Keras
* OpenCV
* NumPy
* Python

---

## Limitations

* Uses Haar Cascade (not state-of-the-art face detection)
* Model performance depends on dataset quality
* Limited robustness to lighting and occlusion
* No transfer learning or advanced architectures

---

## Future Improvements

* Replace Haar cascade with MediaPipe or MTCNN
* Add data augmentation
* Use transfer learning (ResNet / MobileNet)
* Add evaluation metrics (confusion matrix, F1-score)

---

## How to Run

### Train model

```bash
python train.py
```

### Run webcam app

```bash
python webcam_app.py
```

---

## Summary

This project demonstrates how to build and deploy a real-time computer vision system, combining deep learning with live video processing and prediction stabilization techniques.
