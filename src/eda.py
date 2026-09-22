"""Exploratory dataset analysis computed only from real files on disk."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from src.config import (
    CLASS_NAMES,
    DATASET_STATS_PATH,
    DISPLAY_NAMES,
    PLOTS_DIR,
    SUPPORTED_EXTENSIONS,
    TRAINING_DIR,
    ensure_directories,
)


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def analyze_dataset(training_dir: Path = TRAINING_DIR) -> dict:
    """Compute class counts, dimensions, corrupt files, and imbalance from disk."""
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"Training dataset not found at {training_dir}. Add the dataset before running EDA."
        )

    counts: dict[str, int] = {}
    dimensions: Counter[str] = Counter()
    corrupt_files: list[str] = []

    for class_name in CLASS_NAMES:
        class_dir = training_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        files = _image_files(class_dir)
        counts[class_name] = len(files)
        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
            except (UnidentifiedImageError, OSError, SyntaxError):
                corrupt_files.append(str(path.relative_to(training_dir)))

    total_images = sum(counts.values())
    nonzero_counts = [count for count in counts.values() if count > 0]
    imbalance_ratio = (
        max(nonzero_counts) / min(nonzero_counts) if len(nonzero_counts) >= 2 else None
    )

    return {
        "total_images": total_images,
        "number_of_classes": len(CLASS_NAMES),
        "images_per_class": counts,
        "image_dimensions": dict(dimensions.most_common()),
        "corrupted_files": corrupt_files,
        "potential_imbalance_ratio_max_to_min": imbalance_ratio,
    }


def save_class_distribution(stats: dict) -> Path:
    ensure_directories()
    counts = stats["images_per_class"]
    labels = [DISPLAY_NAMES[name] for name in CLASS_NAMES]
    values = [counts[name] for name in CLASS_NAMES]
    path = PLOTS_DIR / "class_distribution.png"
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Training Set Class Distribution")
    plt.xlabel("MRI Category")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_sample_grid(training_dir: Path = TRAINING_DIR) -> Path | None:
    ensure_directories()
    examples: list[tuple[str, Path]] = []
    for class_name in CLASS_NAMES:
        files = _image_files(training_dir / class_name)
        if files:
            examples.append((class_name, files[0]))
    if not examples:
        return None

    path = PLOTS_DIR / "sample_mri_grid.png"
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
    if len(examples) == 1:
        axes = [axes]
    for axis, (class_name, image_path) in zip(axes, examples):
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(DISPLAY_NAMES[class_name])
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_eda(training_dir: Path = TRAINING_DIR) -> dict:
    """Run EDA and save only artifacts derived from the actual dataset."""
    ensure_directories()
    stats = analyze_dataset(training_dir)
    DATASET_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    save_class_distribution(stats)
    save_sample_grid(training_dir)
    return stats


if __name__ == "__main__":
    try:
        result = run_eda()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"EDA stopped: {exc}") from exc
    print(json.dumps(result, indent=2))
