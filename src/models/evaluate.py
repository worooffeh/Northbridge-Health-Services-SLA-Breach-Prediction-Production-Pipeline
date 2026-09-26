from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, brier_score_loss,
)


def classification_metrics(y_true, probability, threshold: float = 0.50) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(probability)
    pred = (p >= threshold).astype(int)
    top_n = max(1, int(np.ceil(len(y) * 0.10)))
    top = np.argsort(-p)[:top_n]
    base = y.mean()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision_050": float(precision_score(y, pred, zero_division=0)),
        "recall_050": float(recall_score(y, pred, zero_division=0)),
        "f1_050": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "lift_at_10pct": float(y[top].mean() / base) if base else float("nan"),
    }
