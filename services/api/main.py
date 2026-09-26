from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException

from services.api.schemas import PredictionRequest, PredictionResponse
from src.features.online import build_online_frame
from src.models.artifacts import load_champion
from src.models.train import predict_proba

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/champion"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
ROOT_PATH = os.getenv("API_ROOT_PATH", "")

app = FastAPI(
    title="Northbridge SLA Breach API",
    version="1.0.0",
    root_path=ROOT_PATH,
    description="Leakage-aware SLA breach scoring service.",
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None and MODEL_PATH.exists():
        _bundle = load_champion(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_bundle() is not None}


@app.get("/verify")
def verify():
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Champion model artifact not found")
    metadata = bundle["metadata"]
    if METADATA_PATH.exists():
        metadata = {**metadata, **json.loads(METADATA_PATH.read_text(encoding="utf-8"))}
    actual = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    expected = metadata.get("artifact_sha256")
    if expected and actual != expected:
        raise HTTPException(status_code=409, detail="Champion artifact checksum mismatch")
    return {
        "status": "verified",
        "model_name": metadata.get("model_name"),
        "experiment": metadata.get("experiment"),
        "model_version": metadata.get("model_version"),
        "artifact_sha256": actual,
        "trained_through": metadata.get("trained_through"),
        "feature_count": len(bundle["fitted"].categorical) + len(bundle["fitted"].numeric),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Champion model artifact not found")
    frame = build_online_frame(request.model_dump(), bundle["lookups"])
    probability = float(predict_proba(bundle["fitted"], frame)[0])
    threshold = float(bundle["metadata"].get("threshold", 0.5))
    return PredictionResponse(
        breach_probability=probability,
        predicted_breach=probability >= threshold,
        threshold=threshold,
        model_name=bundle["metadata"]["model_name"],
        experiment=bundle["metadata"]["experiment"],
        model_version=bundle["metadata"]["model_version"],
    )
