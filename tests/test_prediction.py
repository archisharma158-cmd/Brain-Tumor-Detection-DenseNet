"""Unit tests for the prediction output contract without a large trained model."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

pytest.importorskip("tensorflow")
from PIL import Image

from src.predict import predict_with_model


class FakeModel:
    def __init__(self, output: list[float]):
        self.output = np.asarray([output], dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert batch.shape[0] == 1
        return self.output


def _valid_image_bytes() -> bytes:
    image = Image.new("RGB", (100, 100), color=(80, 80, 80))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_prediction_output_structure() -> None:
    result = predict_with_model(_valid_image_bytes(), FakeModel([0.05, 0.10, 0.15, 0.70]))
    assert result["predicted_class"] == "Pituitary Tumor"
    assert result["class_index"] == 3
    assert result["confidence"] == pytest.approx(0.70)
    assert set(result["probabilities"]) == {
        "Glioma",
        "Meningioma",
        "No Tumor",
        "Pituitary Tumor",
    }


def test_unexpected_model_output_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected model output shape"):
        predict_with_model(_valid_image_bytes(), FakeModel([0.2, 0.8]))
