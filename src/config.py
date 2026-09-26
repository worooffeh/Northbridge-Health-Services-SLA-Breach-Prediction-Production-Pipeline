from __future__ import annotations

import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_settings(path: str | Path | None = None) -> dict:
    path = Path(path or ROOT / "config" / "settings.yaml")
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        settings.setdefault("mlflow", {})["tracking_uri"] = uri
    return settings


def data_dir() -> Path:
    return Path(os.getenv("NORTHBRIDGE_DATA_DIR", ROOT / "data" / "raw"))
