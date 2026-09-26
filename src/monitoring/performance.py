from __future__ import annotations

from src.models.evaluate import classification_metrics


def performance_snapshot(y_true, probability, threshold=0.5):
    return classification_metrics(y_true, probability, threshold)


def retraining_decision(reference_metrics: dict, current_metrics: dict, auc_drop=0.05, pr_drop=0.05) -> dict:
    auc_delta = current_metrics.get("roc_auc", 0) - reference_metrics.get("roc_auc", 0)
    pr_delta = current_metrics.get("pr_auc", 0) - reference_metrics.get("pr_auc", 0)
    trigger = auc_delta <= -auc_drop or pr_delta <= -pr_drop
    return {
        "retrain": bool(trigger),
        "roc_auc_change": float(auc_delta),
        "pr_auc_change": float(pr_delta),
        "reason": "performance degradation" if trigger else "within tolerance",
    }
