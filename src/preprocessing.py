"""MRI image validation, loading, preprocessing, and augmentation utilities."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

from src.config import IMAGE_SIZE, SUPPORTED_EXTENSIONS

ImageSource = Union[str, Path, bytes, bytearray, BinaryIO, Image.Image]


def load_rgb_image(source: ImageSource, image_size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """Load, validate, convert to RGB, and resize an image safely."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image extension '{path.suffix}'. Use JPG, JPEG, or PNG."
                )
            if not path.is_file():
                raise ValueError(f"Image file not found: {path}")
            image = Image.open(path)
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(source))
        elif hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            image = Image.open(source)
        else:
            raise ValueError("Unsupported image input type.")

        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Image has invalid dimensions.")
        return image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("The uploaded file is not a valid, readable MRI image.") from exc


def image_to_model_batch(source: ImageSource) -> np.ndarray:
    """Convert an image into a DenseNet121-compatible batch."""
    image = load_rgb_image(source)
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def create_data_augmentation(seed: int = 42) -> tf.keras.Sequential:
    """Return conservative augmentation for MRI training images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04, seed=seed),
            tf.keras.layers.RandomZoom(0.08, seed=seed + 1),
            tf.keras.layers.RandomTranslation(0.04, 0.04, seed=seed + 2),
            tf.keras.layers.RandomContrast(0.08, seed=seed + 3),
        ],
        name="mri_augmentation",
    )


def preprocess_dataset_image(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply DenseNet preprocessing to a validation/test image tensor."""
    image = tf.cast(image, tf.float32)
    return preprocess_input(image), label
