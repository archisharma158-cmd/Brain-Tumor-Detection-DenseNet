# Brain Tumor Detection using DenseNet121

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-DenseNet121-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B?logo=streamlit&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transfer%20Learning-0A66C2)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-MRI%20Classification-00A8CC)

A complete academic deep-learning project for classifying brain MRI images into **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** using **DenseNet121 transfer learning**, with evaluation tooling, a Streamlit interface, and **Grad-CAM** explainability.

> **Medical disclaimer:** This system is developed for educational and research purposes only and is not intended to replace professional medical diagnosis.

## Project Overview

The project demonstrates a full machine-learning lifecycle rather than only a prediction page. It inspects an actual folder-based MRI dataset, preprocesses and augments images, trains a pretrained DenseNet121 in two stages, evaluates predictions on a testing split, saves real result artifacts, and exposes the trained model through a presentation-ready Streamlit application.

No model accuracy, confidence value, dataset statistic, graph, model file, or Grad-CAM image is fabricated. Those artifacts appear only after the relevant code has run on real data.

## Problem Statement

Brain MRI image classification is a computer-vision task in which a model learns visual patterns that distinguish MRI categories. This project studies whether transfer learning with DenseNet121 can classify four MRI categories in a reproducible academic workflow.

It is **not** a clinical diagnostic system, a tumor-localization system, or a substitute for a radiologist or doctor.

## Objectives

- Build a four-class MRI image classifier with DenseNet121.
- Use ImageNet-pretrained transfer learning instead of training a large CNN from scratch.
- Apply conservative augmentation only to training images.
- Fine-tune only selected upper DenseNet layers with a lower learning rate.
- Calculate real test accuracy, precision, recall, F1-score, confusion matrix, and ROC/AUC where valid.
- Produce Grad-CAM explanations from the actual network gradients.
- Provide a clean Streamlit interface that remains usable even before training.

## Features

- Folder-based dataset validation
- Exploratory data analysis calculated from actual images
- RGB conversion and `224 × 224` resizing
- DenseNet-specific `preprocess_input`
- Conservative training-only augmentation
- DenseNet121 + lightweight custom classifier head
- Two-stage feature extraction and fine-tuning
- Early stopping, checkpointing, and learning-rate reduction
- Test-set evaluation and saved plots
- Reusable single-image prediction pipeline
- Model confidence and all four class probabilities
- Grad-CAM heatmap and overlay
- Session-only prediction history
- Downloadable educational text report
- Graceful missing-dataset and missing-model behavior
- Basic unit tests without requiring a large trained model

## Classes

The class order is fixed across training, evaluation, and inference:

1. `glioma` → Glioma
2. `meningioma` → Meningioma
3. `notumor` → No Tumor
4. `pituitary` → Pituitary Tumor

Keeping one class order prevents label-index mismatches between the training pipeline and the web application.

## Project Workflow

```text
Brain MRI Dataset
        ↓
Exploratory Data Analysis
        ↓
Image Preprocessing
        ↓
Training-only Data Augmentation
        ↓
DenseNet121 (ImageNet weights)
        ↓
Feature Extraction
        ↓
Selective Fine-Tuning
        ↓
4-Class Softmax Classification
        ↓
Model Evaluation
        ↓
Saved Best Model
        ↓
MRI Upload + Prediction
        ↓
Probability Distribution
        ↓
Grad-CAM Explanation
```

## Model Architecture

```text
Input: 224 × 224 × 3
        ↓
DenseNet121
ImageNet pretrained, include_top=False
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, ReLU, L2 regularization)
        ↓
Dropout(0.35)
        ↓
Dense(4, Softmax)
```

The custom head is intentionally compact so that the pretrained convolutional backbone remains the main feature extractor.

## Why DenseNet121?

DenseNet connects each layer to subsequent layers, allowing efficient feature reuse and improved gradient flow. For an academic image-classification project, DenseNet121 provides a strong pretrained feature extractor without requiring an unnecessarily large custom network.

### What does “121” mean?

DenseNet121 refers to a DenseNet variant with 121 layers in its standard architecture. The project loads the convolutional network without its original ImageNet classification head and adds a new four-class head.

## Dataset

The code expects a brain MRI dataset with four classes and separate `Training` and `Testing` folders. A compatible example is the **Brain Tumor MRI Dataset** on Kaggle by Masoud Nickparvar:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download/extract the dataset yourself; the large dataset is intentionally **not** committed to GitHub.

### Dataset Structure

Place the files under the repository root exactly like this:

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The training loader creates a reproducible validation subset from `Training/`. The supplied `Testing/` folder is kept for final evaluation.

### Exploratory Data Analysis

Run:

```bash
python -m src.eda
```

The EDA code calculates from the files on your machine:

- total number of training images
- images per class
- class distribution
- observed image dimensions
- corrupted/unreadable images
- simple max-to-min class imbalance ratio
- a sample image from each available class

Generated EDA files are stored in `results/` and `results/plots/`.

## Preprocessing

Each MRI image is processed consistently:

```text
Read image
→ validate image
→ convert grayscale/RGBA/etc. to RGB
→ resize to 224 × 224
→ convert to float32 array
→ tensorflow.keras.applications.densenet.preprocess_input
→ DenseNet121
```

Supported uploaded image extensions are `.jpg`, `.jpeg`, and `.png`.

## Data Augmentation

Only the training dataset receives conservative augmentation:

- small rotation
- small zoom
- small translation
- minor contrast change

Validation, testing, and uploaded inference images receive **no random augmentation**. Extreme distortions and vertical flips are deliberately avoided.

## Transfer Learning

### Stage 1 — Feature Extraction

The DenseNet121 convolutional backbone is frozen. Only the custom classification head is trained.

### Stage 2 — Fine-Tuning

The project reloads the best Stage-1 checkpoint, unfreezes only selected upper DenseNet layers, keeps Batch Normalization layers frozen, lowers the learning rate, and fine-tunes the network.

The final `models/best_densenet121.keras` is chosen by comparing validation loss across the saved Stage-1 and Stage-2 checkpoints, so a worse fine-tuning stage does not silently replace a better feature-extraction checkpoint.

## Training Strategy

Default configuration is centralized in `src/config.py`:

- image size: `224 × 224`
- batch size: `16`
- initial epochs: `12`
- fine-tuning epochs: `8`
- initial learning rate: `1e-3`
- fine-tuning learning rate: `1e-5`
- validation split: `20%` of the Training folder
- random seed: `42`

Callbacks:

- `EarlyStopping`
- `ModelCheckpoint`
- `ReduceLROnPlateau`

Python, NumPy, and TensorFlow seeds are set where appropriate. Small numerical differences can still occur across hardware, TensorFlow builds, and GPU implementations.

## Evaluation Metrics

`python -m src.evaluate` calculates values from the actual testing images and model predictions:

- Test Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Classification Report
- Confusion Matrix
- One-vs-Rest ROC curves and AUC for classes where ROC is mathematically valid
- Class-wise precision, recall, and F1 visualization

Generated artifacts include:

```text
results/
├── training_history.json
├── evaluation_metrics.json
├── classification_report.json
├── dataset_statistics.json
└── plots/
    ├── class_distribution.png
    ├── sample_mri_grid.png
    ├── training_accuracy.png
    ├── training_loss.png
    ├── confusion_matrix.png
    ├── class_wise_metrics.png
    └── roc_curves.png          # when valid for the test labels
```

## Grad-CAM Explainability

Grad-CAM uses gradients from the predicted class back to the final convolutional feature maps of the DenseNet121 backbone. The implementation dynamically finds the last `Conv2D` layer rather than depending on a hard-coded layer name.

The app shows:

- original MRI
- Grad-CAM heatmap
- heatmap overlay

> Grad-CAM indicates areas that influenced the neural network's prediction. It does not identify or medically localize a tumor with clinical certainty.

## Project Structure

```text
Brain-Tumor-Detection-DenseNet/
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── predictor.py
│   ├── gradcam.py
│   └── utils.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   └── Brain_Tumor_DenseNet.ipynb
├── models/
│   └── .gitkeep
├── results/
│   ├── plots/
│   │   └── .gitkeep
│   └── .gitkeep
├── sample_images/
│   └── .gitkeep
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── .gitignore
├── pytest.ini
├── requirements.txt
├── README.md
├── LICENSE
└── run_app.py
```

## Installation

Python **3.11 or 3.12** is recommended.

```bash
git clone https://github.com/archisharma158-cmd/Brain-Tumor-Detection-DenseNet.git
cd Brain-Tumor-Detection-DenseNet
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first DenseNet121 construction may download ImageNet pretrained weights if they are not already cached locally.

## Running the Training Pipeline

1. Add the dataset in the required folder structure.
2. Train the model:

```bash
python -m src.train
```

Optional epoch overrides:

```bash
python -m src.train --initial-epochs 15 --fine-tune-epochs 10
```

The final model is saved to:

```text
models/best_densenet121.keras
```

## Running Evaluation

After training:

```bash
python -m src.evaluate
```

The command writes real evaluation metrics and figures into `results/`.

## Running a Single CLI Prediction

```bash
python -m src.predict path/to/mri_image.jpg
```

The output contains the predicted MRI category, model confidence, and probabilities for all four classes.

## Running the Application

```bash
streamlit run app/app.py
```

or:

```bash
python run_app.py
```

If no trained model exists, the application starts normally and explains how to create the model instead of crashing.

## Sample Prediction Format

The app/CLI presents **real model output** in this form after a trained model is available:

```text
Predicted MRI category: <model-selected class>
Model confidence: <model probability>%

Class probabilities:
- Glioma: <probability>%
- Meningioma: <probability>%
- No Tumor: <probability>%
- Pituitary Tumor: <probability>%
```

No example percentage is hard-coded because that could be mistaken for a measured result.

## Tests

Run:

```bash
pytest -q
```

The tests cover image resizing, grayscale-to-RGB conversion, corrupted input handling, unsupported file extensions, DenseNet input-batch shape, and the prediction output contract using a lightweight fake model.

## Technologies Used

- Python
- TensorFlow / Keras
- DenseNet121
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Plotly
- Pillow
- OpenCV (headless)
- Streamlit
- pytest

## Viva-Friendly Concepts

**What is MRI?**  
Magnetic Resonance Imaging is an imaging technique that uses magnetic fields and radio-frequency signals to create detailed images of internal body structures. This project works with MRI image files supplied by the dataset.

**What is a brain tumor?**  
A brain tumor is an abnormal growth of cells in or around the brain. Tumors have different medical types; this project only performs image-category classification for academic study.

**What is image classification?**  
It assigns an input image to one of a predefined set of categories.

**What is a CNN?**  
A Convolutional Neural Network learns spatial visual features using convolutional filters. DenseNet121 is a CNN architecture.

**What is transfer learning?**  
It starts from a model already trained on a large dataset and adapts its learned visual features to a new task.

**Why ImageNet weights?**  
ImageNet pretraining provides broadly useful visual feature filters, reducing the need to learn every low-level image feature from scratch.

**Why resize to 224 × 224?**  
The project uses a consistent DenseNet-compatible input size and keeps training and inference preprocessing identical.

**Why augmentation?**  
Conservative augmentation creates slightly varied training examples and can reduce overfitting without changing the validation/test images.

**Why train/validation/test?**  
Training data updates model weights, validation data guides training decisions, and test data is reserved for final evaluation.

**What is overfitting?**  
Overfitting happens when a model learns the training data too specifically and performs poorly on unseen data.

**What is fine-tuning?**  
Fine-tuning unfreezes selected pretrained layers and updates them with a small learning rate for the new task.

**Accuracy, precision, recall, F1?**  
Accuracy is the fraction of all correct predictions. Precision describes how often predictions for a class are correct. Recall describes how many true examples of a class are found. F1 balances precision and recall.

**What is a confusion matrix?**  
It shows true classes against predicted classes so class-specific mistakes can be inspected.

**What is Grad-CAM?**  
Grad-CAM uses gradients to highlight feature-map regions that influenced a CNN prediction. It is an explanation aid, not a clinical tumor segmentation.

**How does an upload become a prediction?**  
The image is validated, converted to RGB, resized, DenseNet-preprocessed, passed through DenseNet121 and the classification head, converted to class probabilities by softmax, then displayed with a Grad-CAM explanation.

## Future Improvements

- Validate on additional independently sourced MRI datasets.
- Add stronger dataset provenance and leakage checks.
- Compare DenseNet121 against other transfer-learning backbones under the same split.
- Add calibrated uncertainty analysis rather than relying only on softmax confidence.
- Add segmentation only if a properly annotated mask dataset and a segmentation model are introduced.
- Add deployment after model hosting is chosen for the potentially large `.keras` file.

## Model File Size / Deployment

Trained `.keras` and `.h5` model files are ignored by Git. Keep the final local model at:

```text
models/best_densenet121.keras
```

If the model is too large for normal GitHub storage, publish it separately using a GitHub Release, model hosting service, or cloud object storage and document the download step. Do not commit credentials or private tokens.

## Medical Disclaimer

This project is for **education and research only**. It does not provide medical advice or a clinical diagnosis, does not replace MRI interpretation by qualified professionals, and must not be used to make patient-care decisions.

## Author

**Archi Sharma**  
GitHub: https://github.com/archisharma158-cmd

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
