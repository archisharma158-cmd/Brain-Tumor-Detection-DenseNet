"""Central project configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
TRAINING_DIR = DATASET_DIR / "Training"
TESTING_DIR = DATASET_DIR / "Testing"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SAMPLE_IMAGES_DIR = ROOT_DIR / "sample_images"

MODEL_PATH = MODELS_DIR / "best_densenet121.keras"
STAGE1_MODEL_PATH = MODELS_DIR / "stage1_best.keras"
STAGE2_MODEL_PATH = MODELS_DIR / "stage2_best.keras"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.json"
DATASET_STATS_PATH = RESULTS_DIR / "dataset_statistics.json"

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 16
NUM_CLASSES = 4
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 40
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

EDUCATIONAL_DISCLAIMER = (
    "This system is developed for educational and research purposes only and is not "
    "intended to replace professional medical diagnosis."
)
GRADCAM_DISCLAIMER = (
    "Grad-CAM indicates areas that influenced the neural network's prediction. "
    "It does not identify or medically localize a tumor with clinical certainty."
)


def ensure_directories() -> None:
    """Create directories used for generated artifacts."""
    for directory in (MODELS_DIR, RESULTS_DIR, PLOTS_DIR, SAMPLE_IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
