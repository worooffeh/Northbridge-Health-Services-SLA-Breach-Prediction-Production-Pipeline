from __future__ import annotations

import hashlib
import json
from pathlib import Path
import joblib


def save_champion(bundle: dict, model_path: str | Path, metadata_path: str | Path):
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = dict(bundle.get("metadata", {}))
    metadata["artifact_sha256"] = digest
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return digest


def load_champion(model_path: str | Path):
    return joblib.load(model_path)
