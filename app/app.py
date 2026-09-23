"""Polished Streamlit interface for Brain Tumor Detection using DenseNet121."""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT_DIR)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

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

# Load logo for page favicon and inline usage
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"
_LOGO_IMAGE = Image.open(_LOGO_PATH) if _LOGO_PATH.is_file() else None
_LOGO_B64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode() if _LOGO_PATH.is_file() else ""
_LOGO_DATA_URL = f"data:image/png;base64,{_LOGO_B64}" if _LOGO_B64 else ""

# Load wallpaper for app background
_WALLPAPER_PATH = Path(__file__).resolve().parent / "assets" / "wallpaper.png"
_WP_B64 = base64.b64encode(_WALLPAPER_PATH.read_bytes()).decode() if _WALLPAPER_PATH.is_file() else ""
_WP_DATA_URL = f"data:image/png;base64,{_WP_B64}" if _WP_B64 else ""

st.set_page_config(
    page_title="Brain MRI AI | DenseNet121",
    page_icon=_LOGO_IMAGE or "🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Clinical UI system
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy-950: #061426;
        --navy-900: #0A1F36;
        --navy-800: #103253;
        --navy-700: #17476F;
        --pink-600: #DB2777;
        --pink-500: #EC4899;
        --pink-400: #F472B6;
        --pink-100: #FCE7F3;
        --pink-50: #FDF2F8;
        --ink: #102235;
        --muted: #5D7186;
        --line: #DDE6EE;
        --surface: #FFFFFF;
        --canvas: #F6F8FC;
        --success: #16856B;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background-color: #EEF8FF;
        background-image: url("__WALLPAPER_URL__");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
    }

    /* Hide default Streamlit top header menu, deploy button, status widget, and decoration line */
    [data-testid="stDecoration"],
    [data-testid="stToolbarActions"],
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"],
    [data-testid="stStatusWidget"],
    [data-testid="stToolbarActionButton"],
    #MainMenu,
    header::before {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
    }

    /* Make the top header and toolbar areas completely transparent, zero height, and non-blocking */
    [data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
        height: 0px !important;
        min-height: 0px !important;
        max-height: 0px !important;
        padding: 0 !important;
        margin: 0 !important;
        backdrop-filter: none !important;
        z-index: 999990 !important;
        overflow: visible !important;
        pointer-events: none !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        height: 0px !important;
        min-height: 0px !important;
        max-height: 0px !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
        pointer-events: none !important;
    }

    /* Clean up excessive top whitespace so content starts naturally near the top */
    .block-container {
        max-width: 1260px;
        padding-top: 1.25rem !important;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--navy-950) !important;
        letter-spacing: -.02em;
    }

    p, li, label, .stMarkdown {
        color: var(--ink);
    }

    /* Sidebar Base */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy-950) 0%, #0A2847 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    [data-testid="stSidebar"] * {
        color: #F8FBFF;
    }

    /* =========================================================
       SIDEBAR NAVIGATION: CLEAN BUTTON-ROW DESIGN (NO RADIO CIRCLES)
       ========================================================= */

    /* Hide the radio input element */
    [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
        position: absolute !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        pointer-events: none !important;
    }

    /* Hide ONLY the radio circular marks (outer circle and inner dot), keeping text fully visible */
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stRadioOption"] > div > div:first-child,
    [data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child,
    [data-testid="stSidebar"] [role="radiogroup"] div[class*="e1mpz0hj4"],
    [data-testid="stSidebar"] [role="radiogroup"] div[class*="e1mpz0hj5"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Sidebar Navigation Container */
    [data-testid="stSidebar"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.48rem !important;
        width: 100% !important;
    }

    /* Ensure the inner option row wrapper is fully visible, aligned, and full width */
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stRadioOption"] > div,
    [data-testid="stSidebar"] [role="radiogroup"] label > div {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Clickable Navigation Menu Row */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 0.70rem 0.95rem !important;
        margin: 0 !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-left: 3.5px solid transparent !important;
        background: rgba(255, 255, 255, 0.045) !important;
        cursor: pointer !important;
        user-select: none !important;
        transition: all 0.20s cubic-bezier(0.22, 1, 0.36, 1) !important;
    }

    /* Inactive text styling: high-contrast, clean, readable clinical ice-white */
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] p {
        display: flex !important;
        align-items: center !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #E2ECF6 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        transition: color 0.18s ease !important;
    }

    /* Hover state: smooth movement & subtle highlight */
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.10) !important;
        border-color: rgba(244, 114, 182, 0.40) !important;
        transform: translateX(4px) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }

    /* Active state: pink highlight and pink left border */
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked),
    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"],
    [data-testid="stSidebar"] [role="radiogroup"] label:has([aria-checked="true"]) {
        background: linear-gradient(90deg, rgba(236, 72, 153, 0.28) 0%, rgba(236, 72, 153, 0.08) 100%) !important;
        border: 1px solid rgba(244, 114, 182, 0.55) !important;
        border-left: 4px solid var(--pink-500) !important;
        box-shadow: 0 4px 20px rgba(219, 39, 119, 0.24) !important;
        transform: translateX(2px) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [role="radiogroup"] label:has([aria-checked="true"]) [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 750 !important;
    }

    /* Custom pink sidebar navigation icon (floating expand trigger) */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 999999 !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        width: 46px !important;
        height: 46px !important;
        min-width: 46px !important;
        min-height: 46px !important;
        max-width: 46px !important;
        max-height: 46px !important;
        border-radius: 13px !important;
        background: linear-gradient(135deg, var(--pink-500), var(--pink-600)) !important;
        border: 1px solid rgba(255, 255, 255, 0.75) !important;
        box-shadow: 0 10px 26px rgba(219, 39, 119, 0.35) !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        outline: none !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        transform: translateY(-1px) scale(1.05) !important;
        box-shadow: 0 14px 32px rgba(219, 39, 119, 0.45) !important;
        background: linear-gradient(135deg, #EC4899, #DB2777) !important;
    }

    [data-testid="stExpandSidebarButton"]:active,
    [data-testid="stSidebarCollapsedControl"]:active,
    [data-testid="collapsedControl"]:active,
    [data-testid="stSidebarCollapsedControl"] button:active,
    [data-testid="collapsedControl"] button:active {
        transform: scale(0.94) !important;
        box-shadow: 0 6px 16px rgba(219, 39, 119, 0.28) !important;
    }

    /* Target inner button if nested within a container */
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        width: 46px !important;
        height: 46px !important;
        border-radius: 13px !important;
        background: transparent !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        pointer-events: auto !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* All icons and text inside the expand button must be crisp white and non-blocking */
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        pointer-events: none !important;
    }

    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="collapsedControl"] span {
        font-size: 24px !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        display: block !important;
        visibility: visible !important;
        width: 22px !important;
        height: 22px !important;
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        pointer-events: none !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        border-radius: 10px !important;
        background: rgba(255,255,255,.08) !important;
        transition: background 0.2s ease !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background: rgba(255,255,255,.18) !important;
    }

    /* Reusable cards */
    .clinical-card {
        background: rgba(255,255,255,.72);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,.45);
        border-radius: 20px;
        box-shadow: 0 14px 36px rgba(6,20,38,.07);
        padding: 1.35rem 1.4rem;
        height: 100%;
    }

    .clinical-card h3, .clinical-card h4 {
        margin-top: 0;
    }

    .section-intro {
        background: rgba(255,255,255,.68);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,.40);
        border-left: 5px solid var(--pink-500);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        color: var(--ink);
        margin: .5rem 0 1.2rem;
        box-shadow: 0 8px 24px rgba(6,20,38,.04);
    }

    .important-note {
        background: linear-gradient(135deg, rgba(253,242,248,.55), rgba(255,255,255,.60));
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(248,201,223,.50);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        color: var(--navy-900);
        margin: .8rem 0 1rem;
    }

    .medical-disclaimer {
        background: rgba(255,248,251,.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(246,198,220,.45);
        border-left: 5px solid var(--pink-500);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        color: #3A2B35;
        margin-top: 1.1rem;
    }

    .eyebrow {
        display: inline-flex;
        gap: .45rem;
        align-items: center;
        padding: .35rem .65rem;
        border-radius: 999px;
        background: var(--pink-100);
        color: #A31359;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #24B47E;
        box-shadow: 0 0 0 4px rgba(36,180,126,.12);
        display: inline-block;
    }

    /* Hero */
    .hero-shell {
        overflow: hidden;
        position: relative;
        border-radius: 28px;
        padding: 2.65rem;
        background:
            radial-gradient(circle at 85% 12%, rgba(244,114,182,.28), transparent 18rem),
            linear-gradient(135deg, var(--navy-950) 0%, #0C2B49 58%, #153B5F 100%);
        box-shadow: 0 24px 58px rgba(6,20,38,.18);
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 1.3rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.35fr .85fr;
        gap: 2.2rem;
        align-items: center;
    }

    .hero-copy h1 {
        color: #FFFFFF !important;
        font-size: clamp(2.4rem, 5vw, 4.6rem);
        line-height: 1.02;
        margin: .8rem 0 1rem;
        max-width: 800px;
    }

    .hero-copy p {
        color: #D9E7F3 !important;
        font-size: 1.08rem;
        line-height: 1.75;
        max-width: 760px;
        margin-bottom: 1.25rem;
    }

    .hero-highlight {
        color: #F9A8D4;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: .65rem;
        margin-top: 1.2rem;
    }

    .hero-badge {
        background: rgba(255,255,255,.09);
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 999px;
        padding: .48rem .72rem;
        font-size: .84rem;
        font-weight: 650;
    }

    .scan-panel {
        background: rgba(255,255,255,.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 20px 50px rgba(0,0,0,.18);
        border: 1px solid rgba(255,255,255,.45);
    }

    .scan-frame {
        min-height: 230px;
        border-radius: 18px;
        background:
            radial-gradient(circle at 50% 44%, rgba(244,114,182,.33) 0 14%, transparent 15%),
            radial-gradient(circle at 50% 50%, rgba(15,47,78,.14) 0 32%, transparent 33%),
            linear-gradient(145deg, #F7FAFD, #E7EEF5);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }

    .scan-frame::before,
    .scan-frame::after {
        content: "";
        position: absolute;
        border-radius: 999px;
        border: 1px solid rgba(16,50,83,.12);
        width: 170px;
        height: 170px;
    }

    .scan-frame::after {
        width: 110px;
        height: 110px;
        border-color: rgba(236,72,153,.25);
    }

    .brain-glyph {
        font-size: 4.4rem;
        z-index: 2;
        filter: drop-shadow(0 10px 16px rgba(16,50,83,.14));
    }

    .scan-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
        color: var(--navy-900);
        font-size: .9rem;
        font-weight: 650;
    }

    .scan-status span:last-child {
        color: var(--success);
    }

    /* Feature cards */
    .feature-icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--pink-100);
        color: var(--pink-600);
        font-size: 1.2rem;
        margin-bottom: .85rem;
    }

    .feature-title {
        font-weight: 800;
        color: var(--navy-950);
        margin-bottom: .35rem;
    }

    .feature-copy {
        color: var(--muted);
        line-height: 1.55;
        font-size: .93rem;
    }

    .step {
        display: flex;
        gap: .8rem;
        align-items: flex-start;
        padding: .85rem 0;
        border-bottom: 1px solid #EDF2F6;
    }

    .step:last-child { border-bottom: none; }

    .step-number {
        min-width: 30px;
        height: 30px;
        border-radius: 10px;
        background: var(--navy-900);
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: .8rem;
    }

    .step strong { color: var(--navy-950); }
    .step small { color: var(--muted); line-height: 1.5; }

    /* Streamlit widgets */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.68);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,.40);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 28px rgba(6,20,38,.055);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy-950) !important;
        font-weight: 850 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border: 0 !important;
        border-radius: 13px !important;
        min-height: 46px;
        font-weight: 800 !important;
        background: linear-gradient(135deg, var(--pink-500), var(--pink-600)) !important;
        color: #FFFFFF !important;
        box-shadow: 0 10px 25px rgba(219,39,119,.20) !important;
        transition: all .18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(219,39,119,.28) !important;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1.5px dashed #F19CC5;
        border-radius: 18px;
        padding: .7rem;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--pink-500);
        background: var(--pink-50);
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid var(--line);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,.40);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(255,255,255,.68);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    .stImage img {
        border-radius: 16px;
    }

    code {
        background: #EEF3F8 !important;
        color: var(--navy-900) !important;
        border-radius: 7px;
    }

    /* =========================================================
   PROFESSIONAL CLINICAL ANIMATION SYSTEM
   ========================================================= */

@keyframes clinicalPageIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes clinicalCardIn {
    from {
        opacity: 0;
        transform: translateY(12px) scale(.992);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes clinicalHeroIn {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes sidebarIn {
    from {
        opacity: .75;
        transform: translateX(-12px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes statusPulse {
    0%, 100% {
        box-shadow: 0 0 0 4px rgba(36,180,126,.12);
    }

    50% {
        box-shadow: 0 0 0 8px rgba(36,180,126,.055);
    }
}

@keyframes menuPulse {
    0%, 100% {
        box-shadow: 0 12px 30px rgba(219,39,119,.28);
    }

    50% {
        box-shadow: 0 12px 34px rgba(219,39,119,.40);
    }
}


/* Smooth page entrance */

.block-container {
    animation: clinicalPageIn .46s
        cubic-bezier(.22, 1, .36, 1) both;
}


/* Sidebar entrance */

[data-testid="stSidebar"] {
    animation: sidebarIn .28s
        cubic-bezier(.22, 1, .36, 1) both;
}


/* Hero entrance */

.hero-shell {
    animation: clinicalHeroIn .58s
        cubic-bezier(.22, 1, .36, 1) both;
}


/* Cards and important elements */

.clinical-card,
div[data-testid="stMetric"],
[data-testid="stFileUploader"],
[data-testid="stAlert"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
.important-note,
.section-intro,
.medical-disclaimer {
    animation: clinicalCardIn .48s
        cubic-bezier(.22, 1, .36, 1) both;
}


/* Clinical card hover */

.clinical-card,
div[data-testid="stMetric"] {
    transition:
        transform .24s cubic-bezier(.22, 1, .36, 1),
        box-shadow .24s ease,
        border-color .24s ease;
}

.clinical-card:hover,
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(236,72,153,.22);
    box-shadow: 0 18px 42px rgba(6,20,38,.10);
}


/* Feature icon */

.feature-icon {
    transition:
        transform .28s cubic-bezier(.22, 1, .36, 1),
        background .28s ease;
}

.clinical-card:hover .feature-icon {
    transform: translateY(-2px) scale(1.06);
    background: #FBCFE8;
}


/* Model-ready indicator */

.status-dot {
    animation: statusPulse 2.6s ease-in-out infinite;
}


/* Navigation button */

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    animation: menuPulse 2.8s ease-in-out infinite;

    transition:
        transform .20s cubic-bezier(.22, 1, .36, 1),
        box-shadow .20s ease !important;
}

[data-testid="stSidebarCollapsedControl"] button:active,
[data-testid="collapsedControl"] button:active {
    transform: scale(.94) !important;
}


/* Sidebar navigation items */

[data-testid="stSidebar"] [role="radiogroup"] label {
    transition:
        transform .20s cubic-bezier(.22, 1, .36, 1),
        background .20s ease,
        border-color .20s ease,
        box-shadow .20s ease !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    transform: translateX(4px) !important;
}


/* Main buttons */

.stButton > button,
.stDownloadButton > button {
    background-size: 180% 180% !important;
    background-position: left center !important;

    transition:
        transform .18s cubic-bezier(.22, 1, .36, 1),
        box-shadow .18s ease,
        background-position .38s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.008);
    background-position: right center !important;
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(1px) scale(.975) !important;

    box-shadow:
        0 5px 14px rgba(219,39,119,.22) !important;

    transition-duration: .07s !important;
}


/* MRI uploader */

[data-testid="stFileUploader"] section {
    transition:
        transform .22s cubic-bezier(.22, 1, .36, 1),
        background .22s ease,
        border-color .22s ease,
        box-shadow .22s ease;
}

[data-testid="stFileUploader"] section:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 30px rgba(16,50,83,.07);
}


/* MRI images and Grad-CAM images */

.stImage img {
    transition:
        transform .28s cubic-bezier(.22, 1, .36, 1),
        box-shadow .28s ease;
}

.stImage img:hover {
    transform: translateY(-2px) scale(1.008);

    box-shadow:
        0 16px 34px rgba(6,20,38,.11);
}


/* Expandable sections */

[data-testid="stExpander"] {
    transition:
        transform .22s ease,
        box-shadow .22s ease;
}

[data-testid="stExpander"]:hover {
    transform: translateY(-1px);
}


/* Loading state */

[data-testid="stSpinner"] {
    animation: clinicalCardIn .28s
        cubic-bezier(.22, 1, .36, 1) both;
}


/* Accessibility */

@media (prefers-reduced-motion: reduce) {

    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
        scroll-behavior: auto !important;
    }
}

    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; }
        .hero-shell { padding: 1.6rem; }
        .scan-panel { margin-top: .4rem; }
    }
    </style>
    """.replace("__WALLPAPER_URL__", _WP_DATA_URL),
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str, modified_ns: int):
    """Cache model loading while refreshing if the model file changes."""
    _ = modified_ns
    return load_model(Path(model_path))


def navigate_to(page_name: str) -> None:
    """Change the sidebar navigation from a button callback."""
    st.session_state["nav_page"] = page_name


def render_disclaimer() -> None:
    st.markdown(
        f'<div class="medical-disclaimer"><strong>Research-use notice</strong><br>{EDUCATIONAL_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, eyebrow: str = "Clinical AI Research Workspace") -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:1.25rem;">
            <div class="eyebrow">{eyebrow}</div>
            <h1 style="margin:.7rem 0 .45rem;">{title}</h1>
            <div class="section-intro" style="margin-top:.45rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_chart(probability_frame: pd.DataFrame):
    chart = px.bar(
        probability_frame,
        x="Probability (%)",
        y="MRI category",
        orientation="h",
        range_x=[0, 100],
        text="Probability (%)",
    )
    chart.update_traces(
        marker_color="#EC4899",
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
    )
    chart.update_layout(
        height=340,
        margin=dict(l=10, r=34, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#102235"),
        xaxis=dict(showgrid=True, gridcolor="#E6EDF3", title="Model probability (%)"),
        yaxis=dict(title=""),
        bargap=.38,
    )
    return chart


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def render_home() -> None:
    metrics = read_json(EVALUATION_METRICS_PATH) or {}
    accuracy = metrics.get("test_accuracy")
    accuracy_badge = f"{accuracy * 100:.1f}% held-out test accuracy" if isinstance(accuracy, (int, float)) else "Evaluation available after testing"
    model_status = "Model ready" if MODEL_PATH.is_file() else "Training required"

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="eyebrow" style="background:rgba(252,231,243,.14); color:#FBCFE8; border:1px solid rgba(251,207,232,.18);">
                        <span class="status-dot"></span> AI-assisted MRI research interface
                    </div>
                    <h1>Brain MRI classification, <span class="hero-highlight">explained visually.</span></h1>
                    <p>
                        A polished DenseNet121 research workspace for four-class brain MRI classification,
                        confidence review, and Grad-CAM explainability — built for academic demonstration,
                        not clinical diagnosis.
                    </p>
                    <div class="hero-badges">
                        <span class="hero-badge">DenseNet121</span>
                        <span class="hero-badge">4 MRI categories</span>
                        <span class="hero-badge">Grad-CAM explanation</span>
                        <span class="hero-badge">{accuracy_badge}</span>
                    </div>
                </div>
                <div class="scan-panel">
                    <div class="scan-frame"><img src="{_LOGO_DATA_URL}" alt="Brain Tumor Detection Logo" style="width:120px;height:120px;z-index:2;filter:drop-shadow(0 10px 16px rgba(16,50,83,.14));" /></div>
                    <div class="scan-status"><span>DenseNet121 inference engine</span><span>● {model_status}</span></div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    action_col, info_col = st.columns([1.0, 1.75], gap="large")
    with action_col:
        st.button(
            "Start MRI Analysis  →",
            type="primary",
            use_container_width=True,
            on_click=navigate_to,
            args=("MRI Analysis",),
        )
    with info_col:
        st.markdown(
            '<div class="important-note"><strong>Designed for clarity.</strong> Important model outputs, research warnings, and result summaries are now placed on high-contrast white or pink clinical surfaces.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### What the system does")
    feature_cols = st.columns(4, gap="medium")
    features = [
        ("MRI", "MRI Classification", "Upload one brain MRI scan and classify it into one of four trained categories."),
        ("AI", "DenseNet121", "Uses ImageNet transfer learning with selective fine-tuning for MRI feature extraction."),
        ("%", "Confidence Review", "Shows the complete four-class probability distribution instead of only one label."),
        ("◎", "Grad-CAM", "Highlights image regions that influenced the model's decision for interpretability."),
    ]
    for column, (icon, title, copy) in zip(feature_cols, features):
        with column:
            st.markdown(
                f"""
                <div class="clinical-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Model snapshot")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Architecture", "DenseNet121")
    metrics_cols[1].metric("MRI categories", "4")
    if isinstance(accuracy, (int, float)):
        metrics_cols[2].metric("Test accuracy", f"{accuracy * 100:.2f}%")
    else:
        metrics_cols[2].metric("Test accuracy", "Pending")
    test_images = metrics.get("number_of_test_images")
    metrics_cols[3].metric("Held-out test images", str(test_images) if test_images is not None else "Pending")

    st.markdown("### How an MRI becomes a result")
    workflow_left, workflow_right = st.columns([1.1, .9], gap="large")
    with workflow_left:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="step"><span class="step-number">1</span><div><strong>Upload MRI</strong><br><small>JPG, JPEG, or PNG input is validated and converted to RGB.</small></div></div>
                <div class="step"><span class="step-number">2</span><div><strong>Preprocess</strong><br><small>The scan is resized to 224×224 and prepared with DenseNet preprocessing.</small></div></div>
                <div class="step"><span class="step-number">3</span><div><strong>Classify</strong><br><small>DenseNet121 produces probabilities for Glioma, Meningioma, No Tumor, and Pituitary Tumor.</small></div></div>
                <div class="step"><span class="step-number">4</span><div><strong>Explain</strong><br><small>Grad-CAM visualizes influential regions in the model's decision.</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with workflow_right:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="eyebrow">Research safeguards</div>
                <h3 style="margin:.8rem 0 .55rem;">Clear medical-AI boundaries</h3>
                <div class="feature-copy" style="font-size:.96rem;">
                    The interface deliberately uses wording such as <strong>model classification</strong>,
                    <strong>model confidence</strong>, and <strong>Grad-CAM influence</strong> rather than presenting results as a medical diagnosis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_disclaimer()


def render_analysis() -> None:
    render_page_header(
        "MRI Analysis",
        "Upload a brain MRI scan, review the model classification and full probability distribution, then inspect Grad-CAM explanation maps.",
        "MRI Inference Workspace",
    )

    if not MODEL_PATH.is_file():
        st.warning("No trained DenseNet model is available. Train the model with `python -m src.train` before using MRI analysis.")
        render_disclaimer()
        return

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown("#### Upload MRI scan")
        uploaded = st.file_uploader(
            "Choose a JPG, JPEG, or PNG MRI image",
            type=["jpg", "jpeg", "png"],
            help="No patient-identifying information is required or stored by this project.",
        )
    with right:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="eyebrow">Before analysis</div>
                <h4 style="margin:.8rem 0 .5rem;">What the model returns</h4>
                <div class="feature-copy">
                    • Predicted MRI category<br>
                    • Model confidence<br>
                    • Four-class probability distribution<br>
                    • Grad-CAM heatmap and overlay<br>
                    • Downloadable educational report
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.info("Choose an MRI image to begin. The model will not run until you click **Analyze MRI**.")
        render_disclaimer()
        return

    image_bytes = uploaded.getvalue()
    try:
        preview = load_rgb_image(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return

    preview_col, action_col = st.columns([1.05, .95], gap="large")
    with preview_col:
        st.markdown("#### MRI preview")
        st.image(preview, caption=f"Selected image · {uploaded.name}", use_container_width=True)
    with action_col:
        st.markdown(
            '<div class="important-note"><strong>Ready for analysis.</strong><br>The image has been validated and prepared for DenseNet121 preprocessing.</div>',
            unsafe_allow_html=True,
        )
        analyze_clicked = st.button("Analyze MRI", type="primary", use_container_width=True)

    if analyze_clicked:
        try:
            with st.spinner("Running DenseNet121 classification and Grad-CAM analysis..."):
                model = get_cached_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
                result = analyze_mri(image_bytes, model)
                original, heatmap, overlay = create_heatmap_images(image_bytes, model, result["class_index"])
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            st.error(f"Analysis could not be completed: {exc}")
            return
        except Exception:
            st.error("An unexpected prediction error occurred. Verify the trained model and input image.")
            return

        timestamp = datetime.now().astimezone()

        st.markdown("### Model result")
        result_left, result_right = st.columns(2)
        result_left.metric("Predicted MRI category", result["predicted_class"])
        result_right.metric("Model confidence", f"{result['confidence'] * 100:.2f}%")

        st.markdown(
            '<div class="important-note"><strong>Interpretation note:</strong> Confidence is the model\'s softmax probability for its selected category. It is not a measure of medical certainty.</div>',
            unsafe_allow_html=True,
        )

        probability_frame = pd.DataFrame(
            {
                "MRI category": list(result["probabilities"].keys()),
                "Probability (%)": [value * 100 for value in result["probabilities"].values()],
            }
        )
        st.markdown("### Probability distribution")
        st.plotly_chart(build_probability_chart(probability_frame), use_container_width=True)

        st.markdown("### Grad-CAM explainability")
        st.markdown(
            '<div class="section-intro">Compare the original scan with the activation heatmap and overlay. Brighter regions indicate areas that contributed more strongly to the network output.</div>',
            unsafe_allow_html=True,
        )
        image_columns = st.columns(3, gap="medium")
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
        with st.expander("Current-session prediction history", expanded=False):
            st.dataframe(
                pd.DataFrame(st.session_state["prediction_history"]),
                use_container_width=True,
                hide_index=True,
            )

    render_disclaimer()


def render_model_information() -> None:
    render_page_header(
        "Model Information",
        "A concise technical view of the architecture, transfer-learning strategy, input requirements, and output categories used by the MRI classifier.",
        "DenseNet121 Architecture",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", "DenseNet121")
    c2.metric("Learning method", "Transfer Learning")
    c3.metric("Pretraining", "ImageNet")
    c4.metric("Input", "224 × 224 RGB")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="eyebrow">Architecture</div>
                <h3 style="margin:.8rem 0 .55rem;">Why DenseNet121?</h3>
                <div class="feature-copy">
                    DenseNet connects layers densely so later layers can reuse earlier feature maps.
                    This improves feature reuse and gradient flow while keeping the classification head compact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="eyebrow">Training strategy</div>
                <h3 style="margin:.8rem 0 .55rem;">Two-stage transfer learning</h3>
                <div class="feature-copy">
                    Stage 1 freezes the DenseNet121 backbone and trains the custom head. Stage 2 selectively
                    unfreezes upper layers and fine-tunes them with a much smaller learning rate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Processing pipeline")
    st.markdown(
        """
        <div class="clinical-card">
            <div class="step"><span class="step-number">1</span><div><strong>Input</strong><br><small>Brain MRI image in JPG, JPEG, or PNG format.</small></div></div>
            <div class="step"><span class="step-number">2</span><div><strong>Standardization</strong><br><small>RGB conversion, 224×224 resize, and DenseNet preprocessing.</small></div></div>
            <div class="step"><span class="step-number">3</span><div><strong>Feature extraction</strong><br><small>ImageNet-pretrained DenseNet121 convolutional backbone.</small></div></div>
            <div class="step"><span class="step-number">4</span><div><strong>Classification</strong><br><small>Four-unit softmax output for Glioma, Meningioma, No Tumor, and Pituitary Tumor.</small></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Output classes")
    st.markdown(
        '<div class="important-note"><strong>Glioma</strong> &nbsp;•&nbsp; <strong>Meningioma</strong> &nbsp;•&nbsp; <strong>No Tumor</strong> &nbsp;•&nbsp; <strong>Pituitary Tumor</strong></div>',
        unsafe_allow_html=True,
    )
    render_disclaimer()


def render_performance() -> None:
    render_page_header(
        "Model Performance",
        "Evaluation metrics and plots shown here are loaded from the real artifacts produced by the held-out Testing split.",
        "Evaluation Dashboard",
    )

    metrics = read_json(EVALUATION_METRICS_PATH)
    if metrics is None:
        st.info("Model evaluation results will appear after running `python -m src.evaluate`.")
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
            st.markdown(
                f'<div class="important-note"><strong>Evaluation population:</strong> {metrics["number_of_test_images"]} held-out testing images.</div>',
                unsafe_allow_html=True,
            )

        auc_values = metrics.get("one_vs_rest_auc")
        if isinstance(auc_values, dict) and auc_values:
            auc_frame = pd.DataFrame(
                {"Class": list(auc_values.keys()), "AUC": list(auc_values.values())}
            )
            with st.expander("View one-vs-rest AUC values", expanded=False):
                st.dataframe(auc_frame, use_container_width=True, hide_index=True)

    st.markdown("### Evaluation visuals")
    plots = [
        ("training_accuracy.png", "Training and validation accuracy"),
        ("training_loss.png", "Training and validation loss"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("class_wise_metrics.png", "Class-wise precision, recall, and F1"),
        ("roc_curves.png", "One-vs-rest ROC curves"),
    ]
    available = [(PLOTS_DIR / filename, caption) for filename, caption in plots if (PLOTS_DIR / filename).is_file()]
    if available:
        for index in range(0, len(available), 2):
            cols = st.columns(2, gap="large")
            for col, item in zip(cols, available[index:index + 2]):
                path, caption = item
                with col:
                    st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info("Evaluation plots will appear here after training/evaluation artifacts are generated.")

    report = read_json(CLASSIFICATION_REPORT_PATH)
    if report:
        rows = []
        for class_name in ("glioma", "meningioma", "notumor", "pituitary"):
            if class_name in report:
                rows.append({"Class": class_name.title(), **report[class_name]})
        if rows:
            st.markdown("### Class-wise report")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_disclaimer()


def render_about() -> None:
    render_page_header(
        "About the Project",
        "A complete academic deep-learning lifecycle for brain MRI image classification — from dataset validation and preprocessing to evaluation, prediction, and explainability.",
        "Academic Medical-AI Project",
    )

    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="eyebrow">Project scope</div>
                <h3 style="margin:.8rem 0 .55rem;">What this project demonstrates</h3>
                <div class="feature-copy">
                    Dataset inspection, conservative augmentation, DenseNet121 transfer learning, selective fine-tuning,
                    held-out evaluation, reusable prediction, Streamlit deployment, and Grad-CAM explainability.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="clinical-card">
                <div class="eyebrow">Author</div>
                <h3 style="margin:.8rem 0 .55rem;">Archi Sharma</h3>
                <div class="feature-copy">GitHub · <strong>archisharma158-cmd</strong><br>Brain Tumor Detection using DenseNet121</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    stats = read_json(DATASET_STATS_PATH)
    if stats:
        st.markdown("### Dataset snapshot")
        s1, s2, s3 = st.columns(3)
        s1.metric("Training images", stats.get("total_images", "N/A"))
        s2.metric("Classes", stats.get("number_of_classes", "N/A"))
        corrupt = stats.get("corrupted_files")
        s3.metric("Corrupted files", len(corrupt) if isinstance(corrupt, list) else "N/A")
        with st.expander("View complete generated dataset statistics", expanded=False):
            st.json(stats)
    else:
        st.info("Dataset statistics appear here after EDA/training has generated them.")

    render_disclaimer()


# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
PAGES = ("Home", "MRI Analysis", "Model Information", "Performance", "About Project")
PAGE_LABELS = {
    "Home": "🏠  Home",
    "MRI Analysis": "🔬  MRI Analysis",
    "Model Information": "🧠  Model Information",
    "Performance": "📊  Performance",
    "About Project": "ℹ️  About Project",
}

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"

st.sidebar.markdown(
    f"""
    <div style="padding:.45rem .2rem 1.1rem;">
        <img src="{_LOGO_DATA_URL}" alt="Logo" style="width:46px;height:46px;border-radius:13px;margin-bottom:.7rem;object-fit:contain;" />
        <div style="font-size:1.12rem;font-weight:850;line-height:1.15;">Brain MRI AI</div>
        <div style="font-size:.78rem;color:#B9CCDC;margin-top:.25rem;">DenseNet121 Research Workspace</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    PAGES,
    key="nav_page",
    format_func=lambda item: PAGE_LABELS[item],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
model_sidebar_status = "Ready" if MODEL_PATH.is_file() else "Not trained"
status_color = "#34D399" if MODEL_PATH.is_file() else "#F87171"
st.sidebar.markdown(
    f"""
    <div style="font-size:.82rem;color:#C8D7E6;line-height:1.75;padding:0.2rem 0.4rem;">
        <div style="color:#FFFFFF;font-weight:750;margin-bottom:0.25rem;letter-spacing:-0.01em;">System status</div>
        <div>Model · <strong style="color:{status_color};">{model_sidebar_status}</strong></div>
        <div>Architecture · <span style="color:#FFFFFF;">DenseNet121</span></div>
        <div>Explainability · <span style="color:#F9A8D4;">Grad-CAM</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
