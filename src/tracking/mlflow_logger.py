from __future__ import annotations

import json
from pathlib import Path


def log_run(experiment_name: str, run_name: str, params: dict, metrics: dict, artifacts: list[str] | None = None, tracking_uri: str | None = None):
    try:
        import mlflow
    except ImportError:
        return {"logged": False, "reason": "mlflow not installed"}

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for artifact in artifacts or []:
            if Path(artifact).exists():
                mlflow.log_artifact(artifact)
        mlflow.set_tag("architecture", "point-in-time leakage-aware")
        return {"logged": True, "run_id": run.info.run_id}
