from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def global_explanation(fitted, sample_df: pd.DataFrame, output_dir: str | Path) -> dict:
    """Create a portable explanation summary. SHAP is used where directly supported."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = sample_df[fitted.categorical + fitted.numeric].copy()

    summary = {"model": fitted.model_name, "method": None, "top_features": []}
    try:
        import shap
        if fitted.model_name == "CatBoost":
            for col in fitted.categorical:
                X[col] = X[col].fillna("__missing__").astype(str)
            med = getattr(fitted.model, "_northbridge_numeric_medians", {})
            for col in fitted.numeric:
                X[col] = X[col].fillna(med.get(col, X[col].median()))
            values = fitted.model.get_feature_importance(type="ShapValues", data=None)
            # If native SHAP values require Pool, fall through to feature importance.
            raise RuntimeError("Use portable CatBoost feature importance fallback")
        else:
            prep = fitted.model.named_steps["preprocess"]
            estimator = fitted.model.named_steps["model"]
            Xt = prep.transform(X)
            names = prep.get_feature_names_out()
            if fitted.model_name == "XGBoost":
                explainer = shap.TreeExplainer(estimator)
                sv = explainer.shap_values(Xt[: min(250, Xt.shape[0])])
                importance = np.abs(sv).mean(axis=0)
                order = np.argsort(importance)[::-1][:20]
                summary["method"] = "SHAP TreeExplainer"
                summary["top_features"] = [{"feature": str(names[i]), "importance": float(importance[i])} for i in order]
            else:
                coef = np.abs(estimator.coef_[0])
                order = np.argsort(coef)[::-1][:20]
                summary["method"] = "absolute logistic coefficient"
                summary["top_features"] = [{"feature": str(names[i]), "importance": float(coef[i])} for i in order]
    except Exception:
        if fitted.model_name == "CatBoost":
            imp = fitted.model.get_feature_importance()
            names = fitted.categorical + fitted.numeric
            order = np.argsort(imp)[::-1][:20]
            summary["method"] = "CatBoost feature importance fallback"
            summary["top_features"] = [{"feature": names[i], "importance": float(imp[i])} for i in order]
        else:
            summary["method"] = "explanation unavailable"

    (output_dir / "global_explanation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
