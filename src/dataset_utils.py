"""Robust dataset discovery, validation, and resolution utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

CANONICAL_CLASSES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# Common aliases mapping folder names (lowercased, normalized) to canonical names
CLASS_ALIASES: dict[str, str] = {
    "glioma": "glioma",
    "gliomas": "glioma",
    "glioma_tumor": "glioma",
    "glioma tumor": "glioma",
    "meningioma": "meningioma",
    "meningiomas": "meningioma",
    "meningioma_tumor": "meningioma",
    "meningioma tumor": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no tumor": "notumor",
    "no-tumor": "notumor",
    "notumour": "notumor",
    "no tumour": "notumor",
    "normal": "notumor",
    "pituitary": "pituitary",
    "pituitaries": "pituitary",
    "pituitary_tumor": "pituitary",
    "pituitary tumor": "pituitary",
    "pituitary_tumors": "pituitary",
}


class DatasetValidationError(FileNotFoundError):
    """Raised when the dataset structure is incomplete or invalid."""
    pass


def find_project_root(start_path: Path | str | None = None) -> Path:
    """Find repository root by walking up directories looking for project markers.
    
    Robust across CLI, notebooks, VS Code, and different kernel working directories.
    """
    markers = ("README.md", "src", "notebooks")
    if start_path is not None:
        current = Path(start_path).resolve()
        if current.is_file():
            current = current.parent
    else:
        current = Path.cwd().resolve()

    # Search from current up to root
    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in markers):
            # Verify it has at least 'src' and either 'README.md' or 'notebooks'
            if (parent / "src").is_dir() and ((parent / "README.md").is_file() or (parent / "notebooks").is_dir()):
                return parent

    # Fallback to the parent of this file's directory (src/ -> repo_root)
    return Path(__file__).resolve().parents[1]


def normalize_class_name(name: str) -> str | None:
    """Normalize a class folder name to its canonical identifier.
    
    Returns one of ('glioma', 'meningioma', 'notumor', 'pituitary') or None if unrecognized.
    """
    clean = name.strip().lower().replace("-", " ").replace("_", " ")
    # Try exact match first
    if clean in CLASS_ALIASES:
        return CLASS_ALIASES[clean]
    # Check no-space variant
    no_space = clean.replace(" ", "")
    if no_space in CLASS_ALIASES:
        return CLASS_ALIASES[no_space]
    # Check if canonical name is a prominent substring
    for canonical in CANONICAL_CLASSES:
        if canonical in clean or (canonical == "notumor" and ("no tumor" in clean or "notumor" in clean)):
            return canonical
    return None


def resolve_class_directory(split_dir: Path, canonical_name: str) -> Path | None:
    """Find a directory inside split_dir that corresponds to canonical_name.
    
    Iterates actual directory entries to return the exact Path object on disk.
    """
    if not split_dir.is_dir():
        return None

    # Search children matching normalized canonical name
    for child in split_dir.iterdir():
        if child.is_dir() and normalize_class_name(child.name) == canonical_name:
            return child
    return None


def get_split_class_directories(split_dir: Path) -> dict[str, Path]:
    """Return a mapping of canonical class name -> actual directory path on disk."""
    result: dict[str, Path] = {}
    for canonical in CANONICAL_CLASSES:
        found = resolve_class_directory(split_dir, canonical)
        if found is not None:
            result[canonical] = found
    return result


def _find_split_folder(candidate_dir: Path, split_name: str) -> Path | None:
    """Case-insensitive search for a split folder (e.g. 'Training' or 'Testing')."""
    if not candidate_dir.is_dir():
        return None
    split_lower = split_name.lower()
    for child in candidate_dir.iterdir():
        if child.is_dir() and child.name.lower() == split_lower:
            return child
    return None


def find_dataset_root(
    dataset_dir: Path | str | None = None,
    project_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Locate the dataset root, training directory, and testing directory.
    
    Searches:
      1. Explicit dataset_dir parameter (strictly prioritized; never silently falls back)
      2. BRAIN_TUMOR_DATASET_DIR environment variable
      3. <project_root>/dataset
      4. Nested Kaggle folder inside <project_root>/dataset
    
    Returns:
        (dataset_root, training_dir, testing_dir)
    """
    root = project_root or find_project_root()

    # 1. Explicit parameter
    if dataset_dir is not None:
        explicit_path = Path(dataset_dir)
        if not explicit_path.is_absolute():
            explicit_path = (root / explicit_path).resolve()

        if not explicit_path.exists():
            return explicit_path, explicit_path / "Training", explicit_path / "Testing"

        train_dir = _find_split_folder(explicit_path, "Training")
        test_dir = _find_split_folder(explicit_path, "Testing")
        if train_dir is not None and test_dir is not None:
            return explicit_path, train_dir, test_dir

        # Check subdirectories
        if explicit_path.is_dir():
            for child in explicit_path.iterdir():
                if child.is_dir():
                    nested_train = _find_split_folder(child, "Training")
                    nested_test = _find_split_folder(child, "Testing")
                    if nested_train is not None and nested_test is not None:
                        return child, nested_train, nested_test

        return explicit_path, explicit_path / "Training", explicit_path / "Testing"

    # 2. Environment variable
    env_path_str = os.environ.get("BRAIN_TUMOR_DATASET_DIR")
    if env_path_str:
        env_path = Path(env_path_str).resolve()
        if env_path.exists():
            train_dir = _find_split_folder(env_path, "Training")
            test_dir = _find_split_folder(env_path, "Testing")
            if train_dir is not None and test_dir is not None:
                return env_path, train_dir, test_dir
            for child in env_path.iterdir():
                if child.is_dir():
                    nested_train = _find_split_folder(child, "Training")
                    nested_test = _find_split_folder(child, "Testing")
                    if nested_train is not None and nested_test is not None:
                        return child, nested_train, nested_test
            return env_path, env_path / "Training", env_path / "Testing"

    # 3. Default dataset directory
    default_root = root / "dataset"
    if default_root.exists():
        train_dir = _find_split_folder(default_root, "Training")
        test_dir = _find_split_folder(default_root, "Testing")
        if train_dir is not None and test_dir is not None:
            return default_root, train_dir, test_dir
        if default_root.is_dir():
            for child in default_root.iterdir():
                if child.is_dir():
                    nested_train = _find_split_folder(child, "Training")
                    nested_test = _find_split_folder(child, "Testing")
                    if nested_train is not None and nested_test is not None:
                        return child, nested_train, nested_test

    return default_root, default_root / "Training", default_root / "Testing"


def inspect_dataset_structure(
    dataset_dir: Path | str | None = None,
    project_root: Path | None = None,
    supported_extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> dict:
    """Inspect dataset structure and return diagnostic information."""
    root = project_root or find_project_root()
    dataset_root, training_dir, testing_dir = find_dataset_root(dataset_dir, root)
    extensions = frozenset(ext.lower() for ext in supported_extensions)

    discovered_folders: list[str] = []
    if dataset_root.is_dir():
        discovered_folders = [p.name for p in dataset_root.iterdir() if p.is_dir()]

    def analyze_split(split_dir: Path) -> tuple[dict[str, dict], list[str]]:
        classes_info: dict[str, dict] = {}
        missing_classes: list[str] = []

        if not split_dir.is_dir():
            return classes_info, list(CANONICAL_CLASSES)

        for canonical in CANONICAL_CLASSES:
            class_folder = resolve_class_directory(split_dir, canonical)
            if class_folder is None:
                missing_classes.append(canonical)
                classes_info[canonical] = {
                    "exists": False,
                    "folder_name": None,
                    "path": None,
                    "image_count": 0,
                }
            else:
                image_files = [
                    p for p in class_folder.iterdir()
                    if p.is_file() and p.suffix.lower() in extensions
                ]
                classes_info[canonical] = {
                    "exists": True,
                    "folder_name": class_folder.name,
                    "path": class_folder,
                    "image_count": len(image_files),
                }
                if len(image_files) == 0:
                    missing_classes.append(f"{canonical} (0 images)")

        return classes_info, missing_classes

    training_classes, missing_training = analyze_split(training_dir)
    testing_classes, missing_testing = analyze_split(testing_dir)

    missing_splits: list[str] = []
    if not training_dir.is_dir():
        missing_splits.append(str(training_dir))
    if not testing_dir.is_dir():
        missing_splits.append(str(testing_dir))

    is_valid = (
        len(missing_splits) == 0
        and len(missing_training) == 0
        and len(missing_testing) == 0
    )

    error_message: str | None = None
    if not is_valid:
        error_lines = [
            "Dataset structure is incomplete or invalid.",
            "",
            "Expected directory layout:",
            "  <DATASET_PATH>/",
            "  ├── Training/ (or nested under 'Brain Tumor MRI Dataset/')",
            "  │   ├── glioma/",
            "  │   ├── meningioma/",
            "  │   ├── notumor/",
            "  │   └── pituitary/",
            "  └── Testing/",
            "      ├── glioma/",
            "      ├── meningioma/",
            "      ├── notumor/",
            "      └── pituitary/",
            "",
            f"Resolved dataset root: {dataset_root}",
            f"Directories discovered inside root: {discovered_folders if discovered_folders else '[none]'}",
            "",
        ]

        if missing_splits:
            error_lines.append("Missing split directories:")
            for s in missing_splits:
                error_lines.append(f"  - {s}")

        if missing_training:
            error_lines.append("Training split missing classes or images:")
            for c in missing_training:
                error_lines.append(f"  - {c}")

        if missing_testing:
            error_lines.append("Testing split missing classes or images:")
            for c in missing_testing:
                error_lines.append(f"  - {c}")

        error_lines.extend([
            "",
            "How to fix:",
            "  1. Ensure the 'Brain Tumor MRI Dataset' (by Masoud Nickparvar from Kaggle) is extracted.",
            "  2. Place it at '<project_root>/dataset/' or '<project_root>/dataset/Brain Tumor MRI Dataset/'.",
            "  3. Alternatively, set DATASET_PATH in the notebook or set the BRAIN_TUMOR_DATASET_DIR environment variable.",
        ])
        error_message = "\n".join(error_lines)

    return {
        "project_root": root,
        "dataset_root": dataset_root,
        "training_dir": training_dir,
        "testing_dir": testing_dir,
        "discovered_folders": discovered_folders,
        "training_classes": training_classes,
        "testing_classes": testing_classes,
        "missing_splits": missing_splits,
        "missing_training": missing_training,
        "missing_testing": missing_testing,
        "is_valid": is_valid,
        "error_message": error_message,
    }


def validate_dataset_structure(
    training_dir: Path | None = None,
    testing_dir: Path | None = None,
    dataset_dir: Path | str | None = None,
) -> None:
    """Validate that both Training and Testing splits exist and contain all 4 classes.
    
    Raises DatasetValidationError with detailed diagnostic instructions if invalid.
    """
    if training_dir is not None and testing_dir is not None:
        extensions = DEFAULT_EXTENSIONS
        missing: list[str] = []
        for split_name, split_path in (("Training", training_dir), ("Testing", testing_dir)):
            if not split_path.is_dir():
                missing.append(f"Missing split directory: {split_path}")
                continue
            for canonical in CANONICAL_CLASSES:
                class_dir = resolve_class_directory(split_path, canonical)
                if class_dir is None:
                    missing.append(f"{split_name} missing class folder: '{canonical}'")
                else:
                    count = sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
                    if count == 0:
                        missing.append(f"{split_name}/{canonical} has 0 supported images")
        if missing:
            raise DatasetValidationError(
                "Dataset structure validation failed:\n" + "\n".join(f" - {m}" for m in missing)
            )
        return

    info = inspect_dataset_structure(dataset_dir)
    if not info["is_valid"]:
        raise DatasetValidationError(info["error_message"])
