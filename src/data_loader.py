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
    TESTING_DIR,
    TRAINING_DIR,
    VALIDATION_SPLIT,
    SUPPORTED_EXTENSIONS,
)
from src.preprocessing import create_data_augmentation


def validate_dataset_structure(training_dir: Path = TRAINING_DIR, testing_dir: Path = TESTING_DIR) -> None:
    """Validate expected Training/Testing class folders."""
    missing: list[str] = []
    for split_dir in (training_dir, testing_dir):
        if not split_dir.is_dir():
            missing.append(str(split_dir))
            continue
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                missing.append(str(class_dir))
                continue
            image_count = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if image_count == 0:
                missing.append(f"{class_dir} (no supported images found)")
    if missing:
        formatted = "\n - ".join(missing)
        raise FileNotFoundError(
            "Dataset structure is incomplete. Missing paths:\n - " + formatted
        )


def _load_directory(
    directory: Path,
    *,
    batch_size: int,
    shuffle: bool,
    validation_split: float | None = None,
    subset: str | None = None,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=RANDOM_SEED,
        validation_split=validation_split,
        subset=subset,
    )


def load_datasets(batch_size: int = BATCH_SIZE) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load datasets with augmentation only on training data."""
    validate_dataset_structure()

    train_ds = _load_directory(
        TRAINING_DIR,
        batch_size=batch_size,
        shuffle=True,
        validation_split=VALIDATION_SPLIT,
        subset="training",
    )
    val_ds = _load_directory(
        TRAINING_DIR,
        batch_size=batch_size,
        shuffle=False,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
    )
    test_ds = _load_directory(TESTING_DIR, batch_size=batch_size, shuffle=False)

    augmentation = create_data_augmentation(RANDOM_SEED)
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
