"""Reusable prediction pipeline and command-line inference utility."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES, DISPLAY_NAMES, MODEL_PATH
from src.preprocessing import ImageSource, image_to_model_batch


def load_trained_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    """Load the trained model or provide an actionable error."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained DenseNet model was found at {model_path}. "
            "Train it using `python -m src.train` before using MRI prediction."
        )
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        raise RuntimeError(f"The trained model could not be loaded: {exc}") from exc


def predict_with_model(source: ImageSource, model: Any) -> dict:
    """Predict one MRI image and return class, confidence, and all probabilities."""
    batch = image_to_model_batch(source)
    raw = np.asarray(model.predict(batch, verbose=0))
    if raw.shape != (1, len(CLASS_NAMES)):
        raise ValueError(
            f"Unexpected model output shape {raw.shape}; expected (1, {len(CLASS_NAMES)})."
        )
    probabilities = raw[0].astype(float)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite probability values.")

    class_index = int(np.argmax(probabilities))
    class_name = CLASS_NAMES[class_index]
    return {
        "class_index": class_index,
        "class_key": class_name,
        "predicted_class": DISPLAY_NAMES[class_name],
        "confidence": float(probabilities[class_index]),
        "probabilities": {
            DISPLAY_NAMES[name]: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def predict_image(source: ImageSource, model_path: Path = MODEL_PATH) -> dict:
    """Load the trained model and predict one MRI image."""
    model = load_trained_model(model_path)
    return predict_with_model(source, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one brain MRI image.")
    parser.add_argument("image", type=Path, help="Path to a JPG/JPEG/PNG MRI image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        result = predict_image(args.image)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Prediction stopped: {exc}") from exc

    print(f"Predicted MRI category: {result['predicted_class']}")
    print(f"Model confidence: {result['confidence'] * 100:.2f}%")
    print("Class probabilities:")
    for name, probability in result["probabilities"].items():
        print(f"  {name}: {probability * 100:.2f}%")
