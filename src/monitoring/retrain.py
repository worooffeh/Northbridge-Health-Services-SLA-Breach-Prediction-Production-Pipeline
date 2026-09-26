from __future__ import annotations

import pandas as pd


def retraining_gate(drift: pd.DataFrame | None, reference_metrics: dict | None = None, current_metrics: dict | None = None, critical_psi: float = 0.25, auc_drop: float = 0.05, pr_drop: float = 0.05) -> dict:
    reasons = []
    if drift is not None and not drift.empty and (drift["psi"] >= critical_psi).any():
        reasons.append("critical feature drift")
    if reference_metrics and current_metrics:
        if current_metrics.get("roc_auc", 1) <= reference_metrics.get("roc_auc", 1) - auc_drop:
            reasons.append("ROC-AUC degradation")
        if current_metrics.get("pr_auc", 1) <= reference_metrics.get("pr_auc", 1) - pr_drop:
            reasons.append("PR-AUC degradation")
    return {"retrain": bool(reasons), "reasons": reasons or ["no retraining trigger"]}
