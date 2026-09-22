"""Unit tests for MRI preprocessing."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("tensorflow")
from PIL import Image

from src.config import IMAGE_SIZE
from src.preprocessing import image_to_model_batch, load_rgb_image


def _image_bytes(mode: str = "L", size: tuple[int, int] = (80, 60)) -> bytes:
    image = Image.new(mode, size, color=100)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_grayscale_is_converted_to_rgb_and_resized() -> None:
    image = load_rgb_image(_image_bytes("L"))
    assert image.mode == "RGB"
    assert image.size == IMAGE_SIZE


def test_rgb_is_resized() -> None:
    image = load_rgb_image(_image_bytes("RGB", (320, 180)))
    assert image.mode == "RGB"
    assert image.size == IMAGE_SIZE


def test_model_batch_shape_and_values() -> None:
    batch = image_to_model_batch(_image_bytes("RGB"))
    assert batch.shape == (1, IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all()


def test_corrupted_bytes_are_rejected() -> None:
    with pytest.raises(ValueError, match="valid, readable MRI image"):
        load_rgb_image(b"this is not an image")


def test_unsupported_path_extension_is_rejected(tmp_path) -> None:
    bad_path = tmp_path / "scan.bmp"
    bad_path.write_bytes(_image_bytes("RGB"))
    with pytest.raises(ValueError, match="Unsupported image extension"):
        load_rgb_image(bad_path)
