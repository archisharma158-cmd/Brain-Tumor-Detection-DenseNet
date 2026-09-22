"""Small utilities shared by Streamlit sections."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import EDUCATIONAL_DISCLAIMER


def read_json(path: Path) -> dict | None:
    """Read a JSON object if it exists and is valid."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def build_prediction_report(filename: str, result: dict, timestamp: datetime) -> bytes:
    """Build a simple text report without patient-identifying information."""
    lines = [
        "BRAIN TUMOR DETECTION USING DENSENET121",
        "Educational Prediction Report",
        "=" * 48,
        f"File name: {filename}",
        f"Timestamp: {timestamp.isoformat(timespec='seconds')}",
        "Model name: DenseNet121",
        f"Predicted MRI category: {result['predicted_class']}",
        f"Model confidence: {result['confidence'] * 100:.2f}%",
        "",
        "Probability distribution:",
    ]
    lines.extend(
        f"- {name}: {probability * 100:.2f}%"
        for name, probability in result["probabilities"].items()
    )
    lines.extend(["", f"Educational use only. This report is not a medical diagnosis.", EDUCATIONAL_DISCLAIMER])
    return "\n".join(lines).encode("utf-8")
