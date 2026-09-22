"""Convenience launcher for the Streamlit application."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app" / "app.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_PATH)]))
