"""Folder-based dataset loading for training, validation, and testing."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    VALIDATION_SPLIT,
    get_dataset_paths,
)
from src.dataset_utils import (
    resolve_class_directory,
    validate_dataset_structure as _validate_structure,
)
from src.preprocessing import create_data_augmentation


def validate_dataset_structure(
    training_dir: Path | None = None,
    testing_dir: Path | None = None,
    dataset_dir: Path | str | None = None,
) -> None:
    """Validate expected Training/Testing class folders and report actionable diagnostics."""
    if training_dir is None or testing_dir is None:
        _, train_path, test_path = get_dataset_paths(dataset_dir)
        training_dir = training_dir or train_path
        testing_dir = testing_dir or test_path
    _validate_structure(training_dir=training_dir, testing_dir=testing_dir, dataset_dir=dataset_dir)


def _resolve_split_subfolders(split_dir: Path) -> list[str]:
    """Find actual subfolder names matching canonical CLASS_NAMES in order.
    
    Ensures that class ordering (0: glioma, 1: meningioma, 2: notumor, 3: pituitary)
    is preserved even if folders on disk are titled 'Glioma', 'No Tumor', etc.
    """
    subfolder_names: list[str] = []
    for canonical in CLASS_NAMES:
        resolved = resolve_class_directory(split_dir, canonical)
        if resolved is not None:
            subfolder_names.append(resolved.name)
        else:
            subfolder_names.append(canonical)
    return subfolder_names


def _load_directory(
    directory: Path,
    *,
    batch_size: int,
    shuffle: bool,
    validation_split: float | None = None,
    subset: str | None = None,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    class_names = _resolve_split_subfolders(directory)
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )


def load_datasets(
    dataset_dir: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, validation, and test datasets with augmentation on training data only."""
    _, training_dir, testing_dir = get_dataset_paths(dataset_dir)
    validate_dataset_structure(training_dir=training_dir, testing_dir=testing_dir)

    train_ds = _load_directory(
        training_dir,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="training",
        seed=random_seed,
    )
    val_ds = _load_directory(
        training_dir,
        batch_size=batch_size,
        shuffle=False,
        validation_split=validation_split,
        subset="validation",
        seed=random_seed,
    )
    test_ds = _load_directory(
        testing_dir,
        batch_size=batch_size,
        shuffle=False,
        seed=random_seed,
    )

    augmentation = create_data_augmentation(random_seed)
    autotune = tf.data.AUTOTUNE

    def augment_and_preprocess(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = augmentation(images, training=True)
        images = preprocess_input(tf.cast(images, tf.float32))
        return images, labels

    def only_preprocess(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return preprocess_input(tf.cast(images, tf.float32)), labels

    train_ds = train_ds.map(augment_and_preprocess, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(only_preprocess, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(only_preprocess, num_parallel_calls=autotune).prefetch(autotune)
    return train_ds, val_ds, test_ds
