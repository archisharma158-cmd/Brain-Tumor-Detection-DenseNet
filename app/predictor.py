"""Application-facing prediction helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import MODEL_PATH
from src.predict import load_trained_model, predict_with_model
from src.preprocessing import ImageSource


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the project model with user-friendly errors handled upstream."""
    return load_trained_model(model_path)


def analyze_mri(source: ImageSource, model: Any) -> dict:
    """Analyze one MRI with an already-loaded model."""
    return predict_with_model(source, model)
