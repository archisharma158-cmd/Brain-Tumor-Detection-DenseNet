"""Grad-CAM explainability utilities for the nested DenseNet121 model."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.model import find_backbone
from src.preprocessing import ImageSource, image_to_model_batch, load_rgb_image


def find_last_conv_layer(backbone: tf.keras.Model) -> tf.keras.layers.Layer:
    """Dynamically locate the final 2D convolutional layer in DenseNet121."""
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No convolutional layer was found in the DenseNet121 backbone.")


def make_gradcam_heatmap(
    source: ImageSource,
    model: tf.keras.Model,
    class_index: int | None = None,
) -> np.ndarray:
    """Generate a real Grad-CAM heatmap for one preprocessed MRI image."""
    batch = image_to_model_batch(source)
    backbone = find_backbone(model)
    target_layer = find_last_conv_layer(backbone)

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output],
        name="gradcam_feature_extractor",
    )
    backbone_index = model.layers.index(backbone)
    head_layers = model.layers[backbone_index + 1 :]

    inputs = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(inputs, training=False)
        x: Any = features
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be computed for this model.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activations = conv_outputs[0]
    heatmap = tf.reduce_sum(activations * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def create_heatmap_images(
    source: ImageSource,
    model: tf.keras.Model,
    class_index: int | None = None,
    alpha: float = 0.40,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Return original image, colorized Grad-CAM heatmap, and overlay."""
    original = load_rgb_image(source)
    heatmap = make_gradcam_heatmap(source, model, class_index)
    resized = cv2.resize(heatmap, original.size, interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(np.clip(resized, 0, 1) * 255)
    color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    original_array = np.asarray(original, dtype=np.uint8)
    overlay_array = cv2.addWeighted(original_array, 1.0 - alpha, color_rgb, alpha, 0)
    return original, Image.fromarray(color_rgb), Image.fromarray(overlay_array)
