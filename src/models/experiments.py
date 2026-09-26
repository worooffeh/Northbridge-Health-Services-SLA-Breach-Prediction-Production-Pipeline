from __future__ import annotations

import pandas as pd
from .specs import EXPERIMENTS
from .train import fit_model, predict_proba
from .evaluate import classification_metrics

MODEL_NAMES = ["Logistic Regression", "XGBoost", "CatBoost"]


def run_experiments(train_df: pd.DataFrame, validation_df: pd.DataFrame, random_state=42) -> pd.DataFrame:
    rows = []
    for exp_name, spec in EXPERIMENTS.items():
        for model_name in MODEL_NAMES:
            fitted = fit_model(model_name, train_df, spec["categorical"], spec["numeric"], random_state)
            p = predict_proba(fitted, validation_df)
            metrics = classification_metrics(validation_df["SLABreached"], p)
            rows.append({
                "experiment": exp_name,
                "model": model_name,
                "n_features": len(spec["categorical"]) + len(spec["numeric"]),
                **metrics,
            })
    return pd.DataFrame(rows)


def choose_champion(results: pd.DataFrame) -> pd.Series:
    return results.sort_values(["roc_auc", "pr_auc", "lift_at_10pct"], ascending=False).iloc[0]
