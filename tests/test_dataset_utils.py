"""Unit tests for dataset discovery, path resolution, and validation."""
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import pytest

from src.dataset_utils import (
    CANONICAL_CLASSES,
    DatasetValidationError,
    find_dataset_root,
    find_project_root,
    inspect_dataset_structure,
    normalize_class_name,
    resolve_class_directory,
    validate_dataset_structure,
)


def _create_dummy_image(path: Path) -> None:
    """Create a minimal valid JPEG image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(path, format="JPEG")


def test_find_project_root_from_cwd(tmp_path: Path) -> None:
    """Root detection should locate directory containing src and README.md."""
    repo = tmp_path / "mock_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("# Mock", encoding="utf-8")
    subfolder = repo / "notebooks" / "nested"
    subfolder.mkdir(parents=True)

    detected = find_project_root(subfolder)
    assert detected == repo


def test_normalize_class_name_aliases() -> None:
    """Check case-insensitivity and alias normalization."""
    assert normalize_class_name("glioma") == "glioma"
    assert normalize_class_name("Glioma") == "glioma"
    assert normalize_class_name("glioma_tumor") == "glioma"
    assert normalize_class_name("meningioma") == "meningioma"
    assert normalize_class_name("Meningioma") == "meningioma"
    assert normalize_class_name("notumor") == "notumor"
    assert normalize_class_name("No Tumor") == "notumor"
    assert normalize_class_name("no_tumor") == "notumor"
    assert normalize_class_name("pituitary") == "pituitary"
    assert normalize_class_name("Pituitary") == "pituitary"
    assert normalize_class_name("pituitary_tumor") == "pituitary"
    assert normalize_class_name("unknown_class") is None


def test_standard_dataset_discovery(tmp_path: Path) -> None:
    """Standard layout: dataset/Training and dataset/Testing."""
    ds_root = tmp_path / "dataset"
    for split in ("Training", "Testing"):
        for cls in CANONICAL_CLASSES:
            _create_dummy_image(ds_root / split / cls / "sample.jpg")

    root, train_dir, test_dir = find_dataset_root(dataset_dir=ds_root)
    assert root == ds_root
    assert train_dir == ds_root / "Training"
    assert test_dir == ds_root / "Testing"


def test_nested_kaggle_dataset_discovery(tmp_path: Path) -> None:
    """Nested layout: dataset/Brain Tumor MRI Dataset/Training & Testing."""
    ds_root = tmp_path / "dataset"
    nested = ds_root / "Brain Tumor MRI Dataset"
    for split in ("Training", "Testing"):
        for cls in CANONICAL_CLASSES:
            _create_dummy_image(nested / split / cls / "sample.jpg")

    root, train_dir, test_dir = find_dataset_root(dataset_dir=ds_root)
    assert root == nested
    assert train_dir == nested / "Training"
    assert test_dir == nested / "Testing"


def test_environment_variable_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BRAIN_TUMOR_DATASET_DIR environment variable should be prioritized."""
    custom_root = tmp_path / "custom_location"
    for split in ("Training", "Testing"):
        for cls in CANONICAL_CLASSES:
            _create_dummy_image(custom_root / split / cls / "sample.jpg")

    monkeypatch.setenv("BRAIN_TUMOR_DATASET_DIR", str(custom_root))
    root, train_dir, test_dir = find_dataset_root()
    assert root == custom_root
    assert train_dir == custom_root / "Training"
    assert test_dir == custom_root / "Testing"


def test_missing_dataset_raises_informative_error(tmp_path: Path) -> None:
    """Missing dataset directory should produce a rich diagnostic message."""
    empty_root = tmp_path / "non_existent_dataset"
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset_structure(dataset_dir=empty_root)
    message = str(excinfo.value)
    assert "Expected directory layout:" in message
    assert "Training" in message
    assert "Testing" in message
    assert "How to fix:" in message


def test_missing_class_directory_reported(tmp_path: Path) -> None:
    """Missing a specific class directory should be reported specifically."""
    ds_root = tmp_path / "dataset"
    for split in ("Training", "Testing"):
        for cls in ("glioma", "meningioma", "notumor"):  # 'pituitary' missing
            _create_dummy_image(ds_root / split / cls / "sample.jpg")

    info = inspect_dataset_structure(dataset_dir=ds_root)
    assert not info["is_valid"]
    assert "pituitary" in info["missing_training"]
    assert "pituitary" in info["missing_testing"]


def test_empty_class_directory_detected(tmp_path: Path) -> None:
    """Class directory exists but contains no valid images."""
    ds_root = tmp_path / "dataset"
    for split in ("Training", "Testing"):
        for cls in CANONICAL_CLASSES:
            folder = ds_root / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            if cls != "glioma":
                _create_dummy_image(folder / "sample.jpg")
            # glioma folder left empty

    info = inspect_dataset_structure(dataset_dir=ds_root)
    assert not info["is_valid"]
    assert any("glioma" in item and "0 images" in item for item in info["missing_training"])


def test_case_variant_class_folders_resolved(tmp_path: Path) -> None:
    """Folder names like 'Glioma' or 'No Tumor' should resolve correctly."""
    split_dir = tmp_path / "Training"
    variant_names = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "notumor": "No Tumor",
        "pituitary": "pituitary_tumor",
    }
    for canonical, folder_name in variant_names.items():
        folder = split_dir / folder_name
        _create_dummy_image(folder / "sample.jpg")
        resolved = resolve_class_directory(split_dir, canonical)
        assert resolved is not None
        assert resolved.name == folder_name


def test_tiny_dataset_loader_integration(tmp_path: Path) -> None:
    """Verify that load_datasets works on a minimal valid synthetic dataset."""
    ds_root = tmp_path / "tiny_dataset"
    # Create 5 images per class in Training and 2 in Testing
    for cls in CANONICAL_CLASSES:
        for i in range(5):
            _create_dummy_image(ds_root / "Training" / cls / f"tr_{i}.jpg")
        for i in range(2):
            _create_dummy_image(ds_root / "Testing" / cls / f"te_{i}.jpg")

    from src.data_loader import load_datasets
    train_ds, val_ds, test_ds = load_datasets(
        dataset_dir=ds_root,
        batch_size=2,
        validation_split=0.20,
    )
    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None
