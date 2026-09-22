"""Comprehensive evaluation for a trained DenseNet121 model."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CLASS_NAMES,
    DISPLAY_NAMES,
    EVALUATION_METRICS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    ensure_directories,
)
from src.data_loader import load_datasets


def _save_confusion_matrix(matrix: np.ndarray) -> None:
    labels = [DISPLAY_NAMES[name] for name in CLASS_NAMES]
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _save_class_metrics(report: dict) -> None:
    metrics = ("precision", "recall", "f1-score")
    labels = [DISPLAY_NAMES[name] for name in CLASS_NAMES]
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5))
    for index, metric in enumerate(metrics):
        values = [report[name][metric] for name in CLASS_NAMES]
        axis.bar(x + (index - 1) * width, values, width, label=metric.title())
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Class-wise Evaluation Metrics")
    axis.set_xticks(x, labels, rotation=20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_wise_metrics.png", dpi=160)
    plt.close(fig)


def _save_multiclass_roc(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    binary_true = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    auc_values: dict[str, float] = {}
    fig, axis = plt.subplots(figsize=(8, 6))
    plotted = False
    for class_index, class_name in enumerate(CLASS_NAMES):
        positives = binary_true[:, class_index]
        if len(np.unique(positives)) < 2:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            positives, probabilities[:, class_index]
        )
        class_auc = float(auc(false_positive_rate, true_positive_rate))
        auc_values[class_name] = class_auc
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{DISPLAY_NAMES[class_name]} (AUC={class_auc:.3f})",
        )
        plotted = True
    if plotted:
        axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title("One-vs-Rest ROC Curves")
        axis.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)
    return auc_values


def evaluate() -> dict:
    """Evaluate the saved model using the actual testing split."""
    ensure_directories()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.train` first."
        )

    _, _, test_ds = load_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(test_ds, verbose=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    y_pred = np.argmax(probabilities, axis=1)

    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "number_of_test_images": int(len(y_true)),
    }
    auc_values = _save_multiclass_roc(y_true, probabilities)
    if auc_values:
        metrics["one_vs_rest_auc"] = auc_values

    EVALUATION_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CLASSIFICATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_confusion_matrix(matrix)
    _save_class_metrics(report)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as exc:
        raise SystemExit(f"Evaluation stopped: {exc}") from exc
