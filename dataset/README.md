# Brain Tumor MRI Dataset Guide

## 1. Dataset Source

This project is built around the **Brain Tumor MRI Dataset** by **Masoud Nickparvar**, hosted on Kaggle:
- **Kaggle URL**: [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **License**: Public domain / Open database
- **Modality**: T1-weighted contrast-enhanced brain MRI slices (axial, coronal, sagittal)

The dataset contains four distinct diagnostic categories:
1. **Glioma** (`glioma`)
2. **Meningioma** (`meningioma`)
3. **No Tumor** (`notumor`)
4. **Pituitary Tumor** (`pituitary`)

---

## 2. Supported Folder Layouts

The repository features an automatic dataset discovery engine (`src/dataset_utils.py`) that detects either of the two standard extraction layouts:

### Layout A — Standard Structure (Recommended)
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

### Layout B — Nested Kaggle Download Structure
When extracted directly from the downloaded Kaggle zip archive:
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

### Layout C — Custom Directory via Environment Variable
You can store the dataset anywhere on your system and point the project to it using an environment variable:

**Windows PowerShell:**
```powershell
$env:BRAIN_TUMOR_DATASET_DIR = "D:\Datasets\Brain Tumor MRI Dataset"
```

**Linux / macOS:**
```bash
export BRAIN_TUMOR_DATASET_DIR="/data/Brain Tumor MRI Dataset"
```

---

## 3. Class Name Normalization

The system automatically recognizes case and naming variations, including:
- `Glioma` / `glioma`
- `Meningioma` / `meningioma`
- `No Tumor` / `no_tumor` / `notumor` / `no tumor`
- `Pituitary` / `pituitary` / `pituitary_tumor`

The canonical internal order is strictly maintained across all stages:
`("glioma", "meningioma", "notumor", "pituitary")`

---

## 4. How to Verify Dataset Setup

To verify that your dataset is correctly placed and detected without opening a notebook or starting training, run:

```bash
python -c "import src.dataset_utils as du; print(du.inspect_dataset_structure())"
```

Or run exploratory data analysis:
```bash
python -m src.eda
```

If the dataset is incomplete, a diagnostic report will indicate which folders or classes are missing.
