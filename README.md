# Autonomous Vehicles: Traffic Sign Recognition CNN Model

A perception module built with PyTorch to recognize and classify real-world traffic signs. This project implements a Deep Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, achieving robust generalization against noise, glare, and viewing angles.

## Project Overview
* **Target Categories**: 42 distinct classes of traffic signs (e.g., Speed Limits, Stop, Yield, Turn Signs).
* **Framework**: PyTorch & Torchvision.
* **Dataset Management**: `kagglehub` API for hands-off pipeline downloads.
* **Image Preprocessing**: Color space translation (BGR to RGB), uniform resizing ($30 \times 30$), and pixel matrix intensity scaling.

## Model Architecture & Optimization
The perception engine implements a balanced structural stack designed to mitigate overfitting gaps:
* **Feature Extraction**: Dual 2D Convolution layers coupled with ReLU activations and MaxPool downsamping.
* **Regularization**: Spatial and structural Dropout layers (`0.25` and `0.50`) to break over-memorization dependencies.
* **Data Augmentation**: Real-time random canvas rotation ($15^\circ$) and color jittering (brightness/contrast adjustments) to mimic dynamic weather conditions.
* **Early Stopping**: Monitored validation loss checkpoints with a patience rating of 3 epochs to preserve ideal generalizes weights.

## Performance Metrics
* **Final Evaluation Strategy**: Stratified 80/20 train/validation splits with completely isolated holdout `Test.csv` verification.
* **Final Test Accuracy**: **97.28%** on entirely unseen out-of-sample visual arrays.
