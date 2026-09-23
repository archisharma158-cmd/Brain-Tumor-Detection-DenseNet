<p align="center">
  <img src="app/assets/logo.png" alt="Brain Tumor Detection Logo" width="180" />
</p>

# Brain Tumor Detection using DenseNet121

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-DenseNet121-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Transfer%20Learning-0A66C2)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-MRI%20Classification-00A8CC)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A complete academic deep-learning pipeline for classifying brain MRI scans into **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** using **DenseNet121 transfer learning**, featuring robust dataset discovery, an academic Jupyter notebook, evaluation tooling, a Streamlit web application, and **Grad-CAM** visual explainability.

> [!CAUTION]
> **Medical Disclaimer:** This system is developed strictly for educational and research purposes. It does not provide medical advice, does not perform tumor localization with clinical certainty, and must not be used to replace professional diagnosis by certified radiologists or medical oncologists.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Download](#dataset--download)
3. [Accepted Dataset Folder Structures](#accepted-dataset-folder-structures)
4. [Windows Quick-Start Guide](#windows-quick-start-guide)
5. [Jupyter Notebook Workflow (29 Steps)](#jupyter-notebook-workflow)
6. [Dataset Verification](#dataset-verification)
7. [Command-Line Execution](#command-line-execution)
   - [Training Pipeline](#1-training-pipeline)
   - [Model Evaluation](#2-model-evaluation)
   - [Single Image Prediction](#3-single-image-prediction)
   - [Streamlit Web Interface](#4-streamlit-web-interface)
8. [DenseNet121 Architecture](#densenet121-architecture)
9. [Grad-CAM Explainability](#grad-cam-explainability)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Unit Tests](#unit-tests)
12. [Project Structure](#project-structure)
13. [Viva Voce Q&A Preparation](#viva-voce-qa-preparation)

---

## Project Overview

This project implements a reproducible, college-level deep learning lifecycle for brain MRI classification. Rather than just wrapping pre-existing models or fabricating performance metrics, every result—from exploratory data analysis (EDA) to training curves, confusion matrices, ROC curves, and Grad-CAM overlays—is computed strictly from real data on disk.

No model accuracy, confidence percentage, confusion matrix, or Grad-CAM heatmap is hard-coded or fabricated.

---

## Dataset & Download

The model is designed for the **Brain Tumor MRI Dataset** by **Masoud Nickparvar** on Kaggle:
- **Kaggle Link:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Modality:** T1-weighted contrast-enhanced brain MRI images.
- **Classes:** 
  1. `glioma` (Glioma)
  2. `meningioma` (Meningioma)
  3. `notumor` (No Tumor)
  4. `pituitary` (Pituitary Tumor)

The large image dataset is intentionally **not** committed to Git (it is ignored via `.gitignore`). Please download and extract it locally.

---

## Accepted Dataset Folder Structures

The built-in dataset discovery engine (`src/dataset_utils.py`) automatically discovers your dataset in either of the following structures:

### Structure 1 — Standard Layout (Recommended)
```text
Brain-Tumor-Detection-DenseNet/
└── dataset/
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

### Structure 2 — Nested Kaggle Archive Layout
```text
Brain-Tumor-Detection-DenseNet/
└── dataset/
    └── Brain Tumor MRI Dataset/
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

### Structure 3 — Custom Location via Environment Variable
Set `BRAIN_TUMOR_DATASET_DIR`:
```powershell
$env:BRAIN_TUMOR_DATASET_DIR = "D:\Datasets\Brain Tumor MRI Dataset"
```

The system automatically recognizes case variations (e.g. `Glioma`, `Meningioma`, `No Tumor`, `no_tumor`, `pituitary_tumor`) while strictly maintaining the canonical internal class order:
`("glioma", "meningioma", "notumor", "pituitary")`.

---

## Windows Quick-Start Guide

Follow these step-by-step commands in **Windows PowerShell**:

### 1. Clone Repository & Navigate
```powershell
git clone https://github.com/archisharma158-cmd/Brain-Tumor-Detection-DenseNet.git
cd Brain-Tumor-Detection-DenseNet
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register Jupyter Kernel
Register your virtual environment kernel so it appears in VS Code and Jupyter:
```powershell
python -m ipykernel install --user --name brain-tumor-densenet --display-name "Brain Tumor DenseNet"
```

### 5. Select Kernel in VS Code or Jupyter
- In **VS Code**: Open `notebooks/Brain_Tumor_DenseNet.ipynb`, click the kernel selector in the top-right corner, click **Python Environments...**, and select **Brain Tumor DenseNet** (or `.venv`).
- In **Jupyter Notebook / Lab**: Select **Kernel → Change Kernel → Brain Tumor DenseNet**.

---

## Jupyter Notebook Workflow

The academic notebook (`notebooks/Brain_Tumor_DenseNet.ipynb`) is organized into **29 clear, sequential sections**:

1. **Problem Statement & Medical Disclaimer**
2. **Environment Diagnostics & Library Imports** (Python, TensorFlow, NumPy, GPU detection)
3. **Project Root & Path Discovery** (Dynamic parent walking)
4. **Project and Dataset Configuration** (`DATASET_PATH` setting)
5. **Dataset Setup and Validation Diagnostic Cell** (Counts, extensions, missing checks)
6. **Exploratory Data Analysis (EDA)** (File-based statistics, corruption check)
7. **Class Distribution Visualization** (Bar chart of real training counts)
8. **Sample MRI Visualization** (Representative MRI slice per class)
9. **Image Preprocessing Pipeline** (Original vs. RGB $224 \times 224$ vs. $[-1, 1]$ float32 tensor)
10. **Data Augmentation** (Demonstration of conservative transformations)
11. **Prepare Training, Validation, and Test Datasets** (`tf.data` pipeline with prefetch)
12. **Build DenseNet121 Architecture** (Model summary, parameter counts)
13. **Explain DenseNet121 Architecture** (Feature reuse, dense connections, mathematical intuition)
14. **Stage 1 — Transfer Learning Setup & Execution Controls** (`RUN_TRAINING = False`, `QUICK_TEST = False`)
15. **Train Classification Head (Stage 1 Feature Extraction)** (Frozen backbone, callbacks)
16. **Stage 1 Training Curves** (Accuracy and loss plots)
17. **Stage 2 — Selective Fine-Tuning Setup** (Unfreezing upper 40 layers, low learning rate)
18. **Fine-Tuning Execution & Merged Curves** (Training with transition marker)
19. **Final Model Selection and Checkpoint Saving** (`models/best_densenet121.keras`)
20. **Evaluate Test Dataset** (Evaluation on held-out `Testing/`)
21. **Accuracy, Precision, Recall, and F1-Score** (Macro-averaged summary table)
22. **Classification Report** (Per-class precision, recall, F1, and support)
23. **Confusion Matrix** ($4 \times 4$ heatmap with annotated counts)
24. **One-vs-Rest ROC / AUC Analysis** (ROC curves with per-class AUC scores)
25. **Single MRI Prediction Demo** (Ground truth, prediction, confidence, probabilities)
26. **Probability Distribution Bar Chart** (Full softmax distribution)
27. **Grad-CAM Explainability** (Original, heatmap, overlay + warning)
28. **Final Results and Academic Conclusions**
29. **Viva Voce Q&A Preparation Guide** (Common examiner questions and answers)

---

## Dataset Verification

To verify that your dataset is correctly positioned and recognized without running training or launching a notebook, run:

```bash
python -c "import src.dataset_utils as du; print(du.inspect_dataset_structure())"
```

Or run the standalone EDA script:
```bash
python -m src.eda
```

---

## Command-Line Execution

### 1. Training Pipeline
Run two-stage training (Stage 1 feature extraction + Stage 2 selective fine-tuning):
```bash
python -m src.train
```

Optional arguments:
```bash
python -m src.train --dataset-dir "dataset" --initial-epochs 12 --fine-tune-epochs 8 --batch-size 16
```

Output checkpoints saved to:
- `models/stage1_best.keras`
- `models/stage2_best.keras`
- `models/best_densenet121.keras` (Best overall model by validation loss)

### 2. Model Evaluation
Evaluate the trained model on held-out test data:
```bash
python -m src.evaluate
```

This updates metrics and plots in `results/`:
- `results/evaluation_metrics.json`
- `results/classification_report.json`
- `results/plots/confusion_matrix.png`
- `results/plots/class_wise_metrics.png`
- `results/plots/roc_curves.png`

### 3. Single Image Prediction
Predict a single MRI slice from the command line:
```bash
python -m src.predict sample_images/glioma_sample.jpg
```

Output format:
```text
Predicted MRI category: Glioma
Model confidence: 94.21%
Class probabilities:
  Glioma: 94.21%
  Meningioma: 3.12%
  No Tumor: 1.45%
  Pituitary Tumor: 1.22%
```

### 4. Streamlit Web Interface
Launch the interactive web application:
```bash
streamlit run app/app.py
```
*(or run `python run_app.py`)*

If `models/best_densenet121.keras` has not been trained yet, the application loads gracefully and explains how to run training rather than crashing.

---

## DenseNet121 Architecture

```text
Input Tensor: (224, 224, 3)
      ↓
Pretrained DenseNet121 (ImageNet weights, include_top=False)
      ↓
GlobalAveragePooling2D
      ↓
BatchNormalization
      ↓
Dense (256 units, ReLU activation, L2 regularization 1e-4)
      ↓
Dropout (0.35 rate)
      ↓
Dense (4 units, Softmax activation) -> Class Probabilities
```

---

## Grad-CAM Explainability

**Gradient-weighted Class Activation Mapping** computes the gradients of the predicted class score with respect to the feature activation maps of the final convolutional layer of DenseNet121 (`conv5_block16_2_conv`).

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

The resulting activation map is upsampled to $224 \times 224$, color-mapped using the Jet colormap, and blended over the original grayscale MRI slice.

> **Note:** Grad-CAM illustrates regions influencing network decisions; it is an interpretive visual aid, not a clinical tumor segmentation tool.

---

## Troubleshooting Guide

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **DatasetValidationError: Missing split directories** | Dataset folder is missing or incorrectly named. | Ensure `dataset/Training` and `dataset/Testing` exist, or place dataset inside `dataset/Brain Tumor MRI Dataset/`. |
| **Missing class folder or 0 images found** | One of the 4 class folders is missing or empty. | Ensure all 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`) exist in both `Training` and `Testing` with valid JPG/PNG files. |
| **Wrong Jupyter Kernel** | Notebook running on global Python without dependencies. | In VS Code or Jupyter, select the kernel named `Brain Tumor DenseNet` (pointing to `.venv`). |
| **No module named 'src'** | Notebook launched from unexpected working directory. | Cell 1 dynamically resolves `PROJECT_ROOT` and inserts it into `sys.path`. Verify you run Cell 1 first. |
| **ImageNet weights failed to download** | Network connection blocked or offline during first run. | DenseNet121 downloads weights (~29MB) from Keras storage once. Ensure an active internet connection on first run. |
| **ResourceExhaustedError / Out of Memory (OOM)** | GPU/CPU RAM exceeded with default batch size. | Reduce batch size from `16` to `8` in `src/config.py` or use `--batch-size 8`. |
| **No trained model found** | Attempting evaluation or prediction before training. | Train the model first via `python -m src.train` or set `RUN_TRAINING = True` in the notebook. |
| **Corrupted MRI image detected** | An image file is truncated or unreadable. | The EDA script and data loader automatically report corrupt files. Delete or replace the reported files. |

---

## Unit Tests

The test suite validates dataset discovery, class normalization, preprocessing, and prediction contracts without requiring the large dataset or a full trained model:

```bash
pytest -v
```

All 19 tests pass in ~10 seconds.

---

## Project Structure

```text
Brain-Tumor-Detection-DenseNet/
├── app/
│   ├── __init__.py
│   ├── app.py                     # Interactive Streamlit dashboard
│   ├── predictor.py               # Streamlit-facing prediction adapter
│   ├── gradcam.py                 # Grad-CAM heatmap and overlay generator
│   └── utils.py                   # Report generator and JSON reader
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project paths and hyperparameters
│   ├── dataset_utils.py           # Robust dataset resolver & validator
│   ├── data_loader.py             # tf.data train/val/test pipeline
│   ├── preprocessing.py           # RGB conversion, resize, DenseNet scaling
│   ├── eda.py                     # Real exploratory data analysis
│   ├── model.py                   # DenseNet121 construction & fine-tuning
│   ├── train.py                   # Two-stage training pipeline (CLI)
│   ├── evaluate.py                # Test set evaluation & metrics (CLI)
│   └── predict.py                 # Single MRI inference utility (CLI)
├── notebooks/
│   └── Brain_Tumor_DenseNet.ipynb # Complete 29-step academic notebook
├── dataset/
│   ├── README.md                  # Dataset source and setup documentation
│   ├── Training/                  # (Ignored by Git)
│   └── Testing/                   # (Ignored by Git)
├── models/                        # Saved .keras models (Ignored by Git)
├── results/                       # Generated evaluation JSONs & plots
├── sample_images/                 # Real sample images for quick testing
├── tests/
│   ├── test_dataset_utils.py      # Tests for discovery, aliases, validation
│   ├── test_prediction.py         # Tests for prediction contract
│   ├── test_preprocessing.py      # Tests for preprocessing pipeline
│   └── test_project_structure.py  # Tests for repository contract
├── .gitignore
├── pytest.ini
├── requirements.txt
├── README.md
├── LICENSE
└── run_app.py
```

---

## Viva Voce Q&A Preparation

- **Why DenseNet121 instead of VGG or ResNet?**  
  DenseNet connects each layer to every other layer ($x_l = H_l([x_0, \dots, x_{l-1}])$). This enables maximum feature reuse, mitigates vanishing gradients, and achieves higher parameter efficiency (~7M vs ~25M for ResNet50 and ~138M for VGG16).
- **Why use two training stages?**  
  Randomly initialized dense classification head weights produce large gradient updates initially. Freezing the backbone in Stage 1 protects pretrained ImageNet weights. In Stage 2, upper layers are unfrozen with a 100× lower learning rate.
- **Why keep BatchNormalization layers frozen during fine-tuning?**  
  Updating batch statistics on small medical batches degrades pretrained representation quality.
- **What does Grad-CAM compute?**  
  It pools gradients of the target class score with respect to feature maps of the final convolutional layer to visualize model attention.
- **How is data leakage prevented?**  
  The held-out `Testing/` split is never used for training or validation. The `Training/` directory is partitioned into 80% train and 20% validation, and augmentation is applied exclusively to training data.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
