"""Central project configuration and path management."""
from __future__ import annotations

from pathlib import Path
from src.dataset_utils import find_dataset_root, find_project_root

# 1. Project directory structure
ROOT_DIR: Path = find_project_root()
DEFAULT_DATASET_DIR: Path = ROOT_DIR / "dataset"
MODELS_DIR: Path = ROOT_DIR / "models"
RESULTS_DIR: Path = ROOT_DIR / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
SAMPLE_IMAGES_DIR: Path = ROOT_DIR / "sample_images"

# 2. Runtime dataset resolution functions
def get_dataset_paths(dataset_dir: Path | str | None = None) -> tuple[Path, Path, Path]:
    """Return (dataset_root, training_dir, testing_dir) for given or default path."""
    return find_dataset_root(dataset_dir=dataset_dir, project_root=ROOT_DIR)


def get_training_dir(dataset_dir: Path | str | None = None) -> Path:
    """Return resolved training directory."""
    _, training_dir, _ = get_dataset_paths(dataset_dir)
    return training_dir


def get_testing_dir(dataset_dir: Path | str | None = None) -> Path:
    """Return resolved testing directory."""
    _, _, testing_dir = get_dataset_paths(dataset_dir)
    return testing_dir


# Default resolved dataset paths (fallback to DEFAULT_DATASET_DIR if not found)
_resolved_root, _resolved_train, _resolved_test = get_dataset_paths()
DATASET_DIR: Path = _resolved_root
TRAINING_DIR: Path = _resolved_train
TESTING_DIR: Path = _resolved_test

# 3. Artifact locations
MODEL_PATH: Path = MODELS_DIR / "best_densenet121.keras"
STAGE1_MODEL_PATH: Path = MODELS_DIR / "stage1_best.keras"
STAGE2_MODEL_PATH: Path = MODELS_DIR / "stage2_best.keras"
TRAINING_HISTORY_PATH: Path = RESULTS_DIR / "training_history.json"
EVALUATION_METRICS_PATH: Path = RESULTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_PATH: Path = RESULTS_DIR / "classification_report.json"
DATASET_STATS_PATH: Path = RESULTS_DIR / "dataset_statistics.json"

# 4. Input and class configuration
IMAGE_SIZE: tuple[int, int] = (224, 224)
INPUT_SHAPE: tuple[int, int, int] = (*IMAGE_SIZE, 3)
BATCH_SIZE: int = 16
NUM_CLASSES: int = 4
CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
DISPLAY_NAMES: dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}

# 5. Training hyperparameters
INITIAL_EPOCHS: int = 12
FINE_TUNE_EPOCHS: int = 8
INITIAL_LEARNING_RATE: float = 1e-3
FINE_TUNE_LEARNING_RATE: float = 1e-5
FINE_TUNE_LAYERS: int = 40
VALIDATION_SPLIT: float = 0.20
RANDOM_SEED: int = 42

# 6. Disclaimers
EDUCATIONAL_DISCLAIMER: str = (
    "This system is developed for educational and research purposes only and is not "
    "intended to replace professional medical diagnosis."
)
GRADCAM_DISCLAIMER: str = (
    "Grad-CAM indicates areas that influenced the neural network's prediction. "
    "It does not identify or medically localize a tumor with clinical certainty."
)


def ensure_directories() -> None:
    """Create directories used for generated artifacts."""
    for directory in (MODELS_DIR, RESULTS_DIR, PLOTS_DIR, SAMPLE_IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
