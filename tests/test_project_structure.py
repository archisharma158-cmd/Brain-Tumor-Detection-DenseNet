"""Dependency-light checks for the repository contract."""
from pathlib import Path

from src.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, ROOT_DIR


def test_class_contract_is_stable() -> None:
    assert CLASS_NAMES == ("glioma", "meningioma", "notumor", "pituitary")
    assert NUM_CLASSES == 4
    assert IMAGE_SIZE == (224, 224)


def test_required_project_files_exist() -> None:
    required = [
        "app/app.py",
        "app/predictor.py",
        "app/gradcam.py",
        "src/train.py",
        "src/evaluate.py",
        "src/predict.py",
        "notebooks/Brain_Tumor_DenseNet.ipynb",
        "requirements.txt",
        "README.md",
    ]
    assert all((ROOT_DIR / relative).is_file() for relative in required)
