"""Two-stage DenseNet121 training entry point."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import (
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAYERS,
    INITIAL_EPOCHS,
    MODEL_PATH,
    PLOTS_DIR,
    RANDOM_SEED,
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
    TRAINING_HISTORY_PATH,
    ensure_directories,
)
from src.data_loader import load_datasets
from src.eda import run_eda
from src.model import build_densenet121_model, prepare_for_fine_tuning


def set_reproducible_seeds(seed: int = RANDOM_SEED) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_callbacks(checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Create robust callbacks for a training stage."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def _history_to_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def merge_histories(stage1: dict[str, list[float]], stage2: dict[str, list[float]]) -> dict:
    """Merge stage histories while retaining the stage boundary."""
    keys = sorted(set(stage1) | set(stage2))
    merged = {key: stage1.get(key, []) + stage2.get(key, []) for key in keys}
    merged["stage1_epochs_completed"] = len(stage1.get("loss", []))
    merged["stage2_epochs_completed"] = len(stage2.get("loss", []))
    return merged


def save_training_plots(history: dict) -> None:
    """Save training/validation accuracy and loss plots from actual history."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_boundary = int(history.get("stage1_epochs_completed", 0))

    for metric, filename, title in (
        ("accuracy", "training_accuracy.png", "Training and Validation Accuracy"),
        ("loss", "training_loss.png", "Training and Validation Loss"),
    ):
        train_values = history.get(metric, [])
        val_values = history.get(f"val_{metric}", [])
        if not train_values:
            continue
        epochs = range(1, len(train_values) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_values, label=f"Training {metric.title()}")
        if val_values:
            plt.plot(epochs, val_values, label=f"Validation {metric.title()}")
        if 0 < stage_boundary < len(train_values):
            plt.axvline(stage_boundary + 0.5, linestyle="--", label="Fine-tuning begins")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=160)
        plt.close()


def _select_best_stage_model(val_ds: tf.data.Dataset) -> tuple[tf.keras.Model, str, float]:
    """Choose the lower validation-loss checkpoint across both training stages."""
    candidates: list[tuple[tf.keras.Model, str, float]] = []
    for name, path in (("feature_extraction", STAGE1_MODEL_PATH), ("fine_tuning", STAGE2_MODEL_PATH)):
        if path.exists():
            model = tf.keras.models.load_model(path)
            values = model.evaluate(val_ds, verbose=0, return_dict=True)
            candidates.append((model, name, float(values["loss"])))
    if not candidates:
        raise RuntimeError("No stage checkpoint was created during training.")
    return min(candidates, key=lambda item: item[2])


def train(initial_epochs: int = INITIAL_EPOCHS, fine_tune_epochs: int = FINE_TUNE_EPOCHS) -> Path:
    """Run EDA, feature extraction, selective fine-tuning, and save the best model."""
    if initial_epochs <= 0 or fine_tune_epochs <= 0:
        raise ValueError("Both training stage epoch counts must be greater than zero.")

    ensure_directories()
    set_reproducible_seeds()

    print("Running dataset analysis...")
    stats = run_eda()
    print(f"Training images detected: {stats['total_images']}")
    if stats.get("corrupted_files"):
        examples = ", ".join(stats["corrupted_files"][:5])
        suffix = " ..." if len(stats["corrupted_files"]) > 5 else ""
        raise ValueError(
            f"Detected {len(stats['corrupted_files'])} corrupted/unreadable image(s). "
            f"Remove or replace them before training. Examples: {examples}{suffix}"
        )

    train_ds, val_ds, _ = load_datasets()

    print("\nStage 1/2: feature extraction with frozen DenseNet121 backbone")
    model, _ = build_densenet121_model()
    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=make_callbacks(STAGE1_MODEL_PATH),
    )

    print("\nStage 2/2: selective fine-tuning of upper DenseNet121 layers")
    stage1_best = tf.keras.models.load_model(STAGE1_MODEL_PATH)
    fine_tune_model = prepare_for_fine_tuning(stage1_best, FINE_TUNE_LAYERS)
    stage2_history = fine_tune_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=make_callbacks(STAGE2_MODEL_PATH),
    )

    merged = merge_histories(_history_to_dict(stage1_history), _history_to_dict(stage2_history))
    TRAINING_HISTORY_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    save_training_plots(merged)

    best_model, best_stage, best_val_loss = _select_best_stage_model(val_ds)
    best_model.save(MODEL_PATH)
    print(f"\nSaved final model: {MODEL_PATH}")
    print(f"Selected checkpoint stage: {best_stage} (validation loss={best_val_loss:.6f})")
    print("Run `python -m src.evaluate` to calculate test metrics.")
    return MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DenseNet121 brain MRI classifier.")
    parser.add_argument("--initial-epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args.initial_epochs, args.fine_tune_epochs)
    except Exception as exc:
        raise SystemExit(f"Training stopped: {exc}") from exc
