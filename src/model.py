"""DenseNet121 transfer-learning model construction and fine-tuning helpers."""
from __future__ import annotations

import urllib.error
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.optimizers import Adam

from src.config import (
    FINE_TUNE_LEARNING_RATE,
    INITIAL_LEARNING_RATE,
    INPUT_SHAPE,
    NUM_CLASSES,
)


def build_densenet121_model(weights: str | None = "imagenet") -> tuple[Model, Model]:
    """Build the primary four-class DenseNet121 transfer-learning model.
    
    Raises a clear RuntimeError if ImageNet weights fail to download.
    Never silently falls back to random weights.
    """
    inputs = Input(shape=INPUT_SHAPE, name="mri_input")
    try:
        backbone = DenseNet121(
            weights=weights,
            include_top=False,
            input_shape=INPUT_SHAPE,
        )
    except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to download ImageNet weights for DenseNet121: {exc}\n"
            "An active internet connection is required on the first run to download the pretrained weights "
            "from Keras storage (~30 MB). Transfer learning cannot proceed without pretrained weights. "
            "Please check your network connection and retry."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unable to construct DenseNet121 backbone with weights='{weights}': {exc}"
        ) from exc

    backbone.trainable = False

    x = backbone(inputs, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = BatchNormalization(name="head_batch_norm")(x)
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_256",
    )(x)
    x = Dropout(0.35, name="head_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="class_probabilities")(x)

    model = Model(inputs=inputs, outputs=outputs, name="brain_tumor_densenet121")
    compile_model(model, INITIAL_LEARNING_RATE)
    return model, backbone


def compile_model(model: Model, learning_rate: float = INITIAL_LEARNING_RATE) -> None:
    """Compile a model consistently for multiclass MRI classification."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def find_backbone(model: Model) -> Model:
    """Find the nested DenseNet121 backbone in a model or loaded checkpoint."""
    # 1. Direct layer search
    for layer in model.layers:
        if isinstance(layer, Model) and "densenet" in layer.name.lower():
            return layer

    # 2. Check for any Functional/Model layer that has conv layers
    for layer in model.layers:
        if isinstance(layer, Model) and any("conv" in sub.name.lower() for sub in layer.layers):
            return layer

    # 3. Check if the model itself is the backbone
    if any("conv" in layer.name.lower() for layer in model.layers):
        return model

    raise ValueError("DenseNet121 backbone could not be found in the model.")


def unfreeze_for_fine_tuning(backbone: Model, fine_tune_layers: int = 40) -> None:
    """Unfreeze only upper backbone layers while keeping BatchNorm frozen."""
    if fine_tune_layers <= 0:
        raise ValueError("fine_tune_layers must be greater than zero.")

    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - fine_tune_layers)
    for index, layer in enumerate(backbone.layers):
        if index < cutoff or isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def prepare_for_fine_tuning(model: Model, fine_tune_layers: int = 40) -> Model:
    """Configure a model for low-learning-rate fine-tuning."""
    backbone = find_backbone(model)
    unfreeze_for_fine_tuning(backbone, fine_tune_layers)
    compile_model(model, FINE_TUNE_LEARNING_RATE)
    return model
