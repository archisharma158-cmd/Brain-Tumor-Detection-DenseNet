"""Streamlit interface for Brain Tumor Detection using DenseNet121."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT_DIR)

# Ensure the project root has priority over the app/ directory.
# This prevents app/app.py from shadowing the app package.
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)

sys.path.insert(0, ROOT_STR)

import pandas as pd
import plotly.express as px
import streamlit as st

from app.gradcam import create_heatmap_images
from app.predictor import analyze_mri, load_model
from app.utils import build_prediction_report, read_json
from src.config import (
    CLASSIFICATION_REPORT_PATH,
    DATASET_STATS_PATH,
    EDUCATIONAL_DISCLAIMER,
    EVALUATION_METRICS_PATH,
    GRADCAM_DISCLAIMER,
    MODEL_PATH,
    PLOTS_DIR,
)
from src.preprocessing import load_rgb_image

st.set_page_config(
    page_title="Brain Tumor Detection | DenseNet121",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at 15% 10%, #102a43 0, #071521 34%, #040b12 76%); color: #ecf8ff;}
    [data-testid="stSidebar"] {background: #07131f; border-right: 1px solid rgba(80,210,255,.15);}
    .hero {padding: 2.3rem; border: 1px solid rgba(45,212,255,.22); border-radius: 24px;
           background: linear-gradient(135deg, rgba(14,42,68,.94), rgba(5,20,33,.96));
           box-shadow: 0 18px 55px rgba(0,0,0,.25); margin-bottom: 1.2rem;}
    .hero h1 {font-size: 2.7rem; margin-bottom: .4rem;}
    .accent {color: #43d9ff;}
    .muted {color: #a7bfd0;}
    .feature-card {min-height: 130px; padding: 1.15rem; border-radius: 18px;
                   border: 1px solid rgba(75,196,255,.18); background: rgba(8,27,43,.78);}
    .result-card {padding: 1.2rem; border-radius: 18px; border: 1px solid rgba(67,217,255,.24);
                  background: rgba(7,24,39,.90);}
    .disclaimer {padding: .95rem 1.1rem; border-left: 4px solid #f0b429; background: rgba(240,180,41,.08);
                 border-radius: 8px; color: #dce9f2; margin: .8rem 0;}
    div[data-testid="stMetric"] {background: rgba(7,24,39,.82); border: 1px solid rgba(67,217,255,.18);
                                 padding: .7rem; border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str, modified_ns: int):
    """Cache model loading while refreshing if the model file changes."""
    _ = modified_ns
    return load_model(Path(model_path))


def render_disclaimer() -> None:
    st.markdown(f'<div class="disclaimer">⚠️ {EDUCATIONAL_DISCLAIMER}</div>', unsafe_allow_html=True)


def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="muted">AI-ASSISTED BRAIN MRI CLASSIFICATION · ACADEMIC PROJECT</div>
          <h1>Brain Tumor Detection <span class="accent">using DenseNet121</span></h1>
          <p class="muted">This project uses transfer learning with DenseNet121 to classify brain MRI scans into
          Glioma, Meningioma, Pituitary Tumor, or No Tumor.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    cards = [
        ("DenseNet121", "A pretrained convolutional backbone for efficient feature reuse."),
        ("MRI Classification", "Four-category image classification from uploaded brain MRI scans."),
        ("Transfer Learning", "ImageNet features are adapted in feature-extraction and fine-tuning stages."),
        ("Grad-CAM", "Visual explanation of image regions that influenced the network output."),
    ]
    for column, (title, description) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="feature-card"><h4>{title}</h4><p class="muted">{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Project workflow")
    st.code(
        "Brain MRI Dataset → Image Preprocessing → Data Augmentation → DenseNet121 → "
        "Transfer Learning → Fine-Tuning → Classification → Model Evaluation → "
        "MRI Prediction → Grad-CAM Explanation",
        language=None,
    )
    render_disclaimer()


def render_analysis() -> None:
    st.title("MRI Analysis")
    st.caption("Upload one JPG, JPEG, or PNG image. The model output is an educational classification, not a diagnosis.")

    if not MODEL_PATH.is_file():
        st.warning(
            "No trained model is available. Train the DenseNet121 model first.\n\n"
            "Run: `python -m src.train`"
        )
        render_disclaimer()
        return

    uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Choose an MRI image to begin. No prediction is generated until you click **Analyze MRI**.")
        render_disclaimer()
        return

    image_bytes = uploaded.getvalue()
    try:
        preview = load_rgb_image(image_bytes)
        st.image(preview, caption="MRI preview", width=420)
    except ValueError as exc:
        st.error(str(exc))
        return

    if st.button("Analyze MRI", type="primary", use_container_width=True):
        try:
            with st.spinner("Running DenseNet121 classification and Grad-CAM analysis..."):
                model = get_cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
                result = analyze_mri(image_bytes, model)
                original, heatmap, overlay = create_heatmap_images(
                    image_bytes, model, result["class_index"]
                )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            st.error(f"Analysis could not be completed: {exc}")
            return
        except Exception:
            st.error("An unexpected prediction error occurred. Verify the trained model and input image.")
            return

        timestamp = datetime.now().astimezone()
        st.markdown("### Model classification")
        left, right = st.columns(2)
        left.metric("Predicted MRI category", result["predicted_class"])
        right.metric("Model confidence", f"{result['confidence'] * 100:.2f}%")

        probability_frame = pd.DataFrame(
            {
                "MRI category": list(result["probabilities"].keys()),
                "Probability (%)": [value * 100 for value in result["probabilities"].values()],
            }
        )
        st.subheader("Probability distribution")
        chart = px.bar(
            probability_frame,
            x="Probability (%)",
            y="MRI category",
            orientation="h",
            range_x=[0, 100],
        )
        chart.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Grad-CAM explainability")
        image_columns = st.columns(3)
        image_columns[0].image(original, caption="Original MRI", use_container_width=True)
        image_columns[1].image(heatmap, caption="Grad-CAM heatmap", use_container_width=True)
        image_columns[2].image(overlay, caption="Heatmap overlay", use_container_width=True)
        st.info(GRADCAM_DISCLAIMER)

        history_item = {
            "time": timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "file": uploaded.name,
            "classification": result["predicted_class"],
            "confidence": f"{result['confidence'] * 100:.2f}%",
        }
        st.session_state.setdefault("prediction_history", []).append(history_item)

        report = build_prediction_report(uploaded.name, result, timestamp)
        st.download_button(
            "Download educational prediction report",
            data=report,
            file_name="brain_mri_prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    if st.session_state.get("prediction_history"):
        st.subheader("Current-session prediction history")
        st.dataframe(pd.DataFrame(st.session_state["prediction_history"]), use_container_width=True, hide_index=True)
    render_disclaimer()


def render_model_information() -> None:
    st.title("Model Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", "DenseNet121")
    c2.metric("Learning", "Transfer Learning")
    c3.metric("Pretraining", "ImageNet")
    c4.metric("Input", "224 × 224 RGB")

    st.subheader("What is DenseNet?")
    st.write(
        "DenseNet connects each layer to subsequent layers, allowing efficient feature reuse and improved gradient flow. "
        "DenseNet121 is a 121-layer variant used here as the principal image-feature extractor."
    )
    st.subheader("Why transfer learning?")
    st.write(
        "Instead of learning every visual feature from scratch, the project starts from ImageNet-pretrained DenseNet121 weights. "
        "The custom classification head is trained first, then selected upper DenseNet layers are fine-tuned at a lower learning rate."
    )
    st.subheader("Why 224 × 224?")
    st.write(
        "The project standardizes MRI inputs to 224 × 224 pixels with three RGB channels, matching the expected DenseNet121 input format "
        "and keeping training/inference preprocessing consistent."
    )
    st.subheader("Output classes")
    st.write("Glioma · Meningioma · No Tumor · Pituitary Tumor")
    render_disclaimer()


def render_performance() -> None:
    st.title("Performance")
    metrics = read_json(EVALUATION_METRICS_PATH)
    if metrics is None:
        st.info(
            "Model evaluation results will appear after training and evaluation. Run `python -m src.train`, then `python -m src.evaluate`."
        )
    else:
        columns = st.columns(4)
        values = [
            ("Test Accuracy", metrics.get("test_accuracy")),
            ("Macro Precision", metrics.get("macro_precision")),
            ("Macro Recall", metrics.get("macro_recall")),
            ("Macro F1", metrics.get("macro_f1")),
        ]
        for column, (label, value) in zip(columns, values):
            column.metric(label, f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "N/A")
        if metrics.get("number_of_test_images") is not None:
            st.caption(f"Calculated from {metrics['number_of_test_images']} testing images.")

    for filename, caption in (
        ("training_accuracy.png", "Training and validation accuracy"),
        ("training_loss.png", "Training and validation loss"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("class_wise_metrics.png", "Class-wise precision, recall, and F1"),
        ("roc_curves.png", "One-vs-rest ROC curves"),
    ):
        path = PLOTS_DIR / filename
        if path.is_file():
            st.image(str(path), caption=caption, use_container_width=True)

    report = read_json(CLASSIFICATION_REPORT_PATH)
    if report:
        rows = []
        for class_name in ("glioma", "meningioma", "notumor", "pituitary"):
            if class_name in report:
                rows.append({"class": class_name, **report[class_name]})
        if rows:
            st.subheader("Classification report")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_disclaimer()


def render_about() -> None:
    st.title("About Project")
    st.write(
        "This college project demonstrates the complete deep-learning lifecycle for four-class brain MRI image classification: "
        "dataset inspection, preprocessing, conservative augmentation, DenseNet121 transfer learning, selective fine-tuning, "
        "evaluation, prediction, and Grad-CAM explainability."
    )
    stats = read_json(DATASET_STATS_PATH)
    if stats:
        st.subheader("Dataset statistics generated from your local dataset")
        st.json(stats)
    else:
        st.info("Dataset statistics will appear after EDA/training is run on an actual dataset.")
    st.subheader("Author")
    st.write("**Archi Sharma**")
    st.write("GitHub: `archisharma158-cmd`")
    render_disclaimer()


page = st.sidebar.radio(
    "Navigation",
    ("Home", "MRI Analysis", "Model Information", "Performance", "About Project"),
)
st.sidebar.markdown("---")
st.sidebar.caption("DenseNet121 · TensorFlow/Keras · Streamlit · Grad-CAM")

if page == "Home":
    render_home()
elif page == "MRI Analysis":
    render_analysis()
elif page == "Model Information":
    render_model_information()
elif page == "Performance":
    render_performance()
else:
    render_about()
