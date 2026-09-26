from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

from src.config import data_dir, load_settings
from src.data.load import load_source_tables, build_master_table
from src.features.build import build_offline_features
from src.features.historical import fit_serving_lookups
from src.models.experiments import run_experiments, choose_champion
from src.models.specs import EXPERIMENTS
from src.models.train import chronological_split, fit_model, predict_proba
from src.models.evaluate import classification_metrics
from src.models.artifacts import save_champion
from src.tracking.mlflow_logger import log_run


def main():
    settings = load_settings()
    tickets, agents, clients = load_source_tables(data_dir())
    master = build_master_table(tickets, agents, clients)
    features = build_offline_features(master, agents, sla_due_known=settings["prediction"]["sla_due_known"])
    train, valid, test = chronological_split(features)

    results = run_experiments(train, valid, random_state=settings["project"]["random_state"])
    out = Path("reports/experiments/validation_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    champion = choose_champion(results)

    # Refit selected architecture on Train + Validation, then evaluate the untouched Test set once.
    train_valid = pd.concat([train, valid]).sort_values(["CreatedAt", "TicketID"])
    spec = EXPERIMENTS[champion["experiment"]]
    fitted = fit_model(champion["model"], train_valid, spec["categorical"], spec["numeric"], settings["project"]["random_state"])
    test_p = predict_proba(fitted, test)
    test_metrics = classification_metrics(test["SLABreached"], test_p)

    # Serving lookups use only the deployment training history.
    cutoff = test["CreatedAt"].min()
    safe_history = master.loc[master["ResolvedAt"].notna() & (master["ResolvedAt"] < cutoff)].copy()
    lookups = fit_serving_lookups(safe_history)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "model_version": version,
        "experiment": champion["experiment"],
        "model_name": champion["model"],
        "threshold": 0.50,
        "validation_metrics": {k: float(champion[k]) for k in ["roc_auc", "pr_auc", "lift_at_10pct", "brier"]},
        "test_metrics": test_metrics,
        "trained_through": str(train_valid["CreatedAt"].max()),
        "test_period_start": str(test["CreatedAt"].min()),
        "test_period_end": str(test["CreatedAt"].max()),
        "prediction_point": settings["prediction"]["point"],
    }
    bundle = {"fitted": fitted, "lookups": lookups, "metadata": metadata}
    digest = save_champion(bundle, "models/champion/model.joblib", "models/champion/metadata.json")
    metadata["artifact_sha256"] = digest
    Path("models/champion/metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    Path("reports/experiments/test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("Champion:", champion["experiment"], "/", champion["model"])
    print("Validation ROC-AUC:", round(champion["roc_auc"], 4))
    print("Test metrics:", test_metrics)

    log_run(
        experiment_name=settings["mlflow"]["experiment_name"],
        run_name=f'{champion["experiment"]}-{champion["model"]}',
        params={"experiment": champion["experiment"], "model": champion["model"], "prediction_point": settings["prediction"]["point"]},
        metrics={f"test_{k}": v for k, v in test_metrics.items()},
        artifacts=[str(out), "models/champion/metadata.json"],
        tracking_uri=settings["mlflow"].get("tracking_uri"),
    )


if __name__ == "__main__":
    main()
